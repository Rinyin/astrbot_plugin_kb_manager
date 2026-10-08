"""Behaviour tests for ``backend.NativeKBBackend``.

The tests use in-memory doubles that keep real state for the three native
stores (KB metadata database, per-KB text/FTS storage and the vector storage)
instead of only asserting that a mock method was called. Ordering-sensitive
operations record an event log so deletion order can be asserted directly.

Run with the plugin development interpreter:

    data/plugin_data/astrbot_plugin_kb_manager/dev/venv/Scripts/python.exe \
        -m pytest data/plugins/astrbot_plugin_kb_manager/tests/test_backend.py
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

# Import the plugin as a package, mirroring how AstrBot loads plugins. Only the
# parent directory of ``data/plugins`` goes on sys.path; the plugin directory
# itself must never be injected, so absolute in-package imports stay visible.
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
UPSTREAM_ROOT = WORKSPACE_ROOT / "upstream" / "AstrBot"
DATA_DIR = WORKSPACE_ROOT / "data"
for _path in (str(UPSTREAM_ROOT), str(DATA_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# AstrBot resolves all runtime paths from ASTRBOT_ROOT; point it at a throwaway
# directory before the first astrbot import so no real data is ever touched.
os.environ.setdefault("ASTRBOT_ROOT", tempfile.mkdtemp(prefix="kb_backend_root_"))

import astrbot.api  # noqa: E402,F401  (import order matters for upstream modules)
from plugins.astrbot_plugin_kb_manager.backend import NativeKBBackend  # noqa: E402
from plugins.astrbot_plugin_kb_manager.common import KBError  # noqa: E402

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Event log + doubles with real backing state
# ---------------------------------------------------------------------------


class EventLog:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def record(self, name: str, detail: object = None) -> None:
        self.events.append((name, detail))

    def first(self, name: str) -> int:
        for index, (event_name, _detail) in enumerate(self.events):
            if event_name == name:
                return index
        raise AssertionError(f"event not recorded: {name}")

    def count(self, name: str) -> int:
        return sum(1 for event_name, _detail in self.events if event_name == name)


class FakeUploadError(Exception):
    """Mirrors ``KnowledgeBaseUploadError`` shape used by the backend."""

    def __init__(self, user_message, stage="metadata", details=None):
        super().__init__(user_message)
        self.user_message = user_message
        self.stage = stage
        self.details = details or {}


class FakeEmbeddingProvider:
    def __init__(self, provider_id, model="embed-model", dim=8):
        self.provider_config = {"id": provider_id, "model": model}
        self.model_name = model
        self._dim = dim

    def get_dim(self):
        return self._dim

    def meta(self):
        return SimpleNamespace(id=self.provider_config["id"])


class FakeChatProvider:
    def __init__(self, provider_id, model="chat-model"):
        self.provider_config = {"id": provider_id, "model": model}
        self.model_name = model

    def meta(self):
        return SimpleNamespace(id=self.provider_config["id"])


class FakeKB:
    def __init__(
        self,
        kb_id,
        kb_name,
        embedding_provider_id="emb-1",
        chunk_size=512,
        chunk_overlap=50,
    ):
        self.kb_id = kb_id
        self.kb_name = kb_name
        self.description = None
        self.emoji = "📚"
        self.embedding_provider_id = embedding_provider_id
        self.rerank_provider_id = None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k_dense = 50
        self.top_k_sparse = 50
        self.top_m_final = 5
        self.doc_count = 0
        self.chunk_count = 0
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)


class FakeDocument:
    def __init__(
        self,
        doc_id,
        kb_id,
        doc_name,
        file_type="txt",
        file_size=0,
        chunk_count=0,
    ):
        self.doc_id = doc_id
        self.kb_id = kb_id
        self.doc_name = doc_name
        self.file_type = file_type
        self.file_size = file_size
        self.file_path = ""
        self.chunk_count = chunk_count
        self.media_count = 0
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)


class FakeMedia:
    def __init__(self, media_id, doc_id, kb_id, file_path):
        self.media_id = media_id
        self.doc_id = doc_id
        self.kb_id = kb_id
        self.file_path = str(file_path)
        self.media_type = "image"
        self.file_name = Path(file_path).name
        self.file_size = 1
        self.mime_type = "image/png"


class FakeEmbeddingStorage:
    def __init__(self, log: EventLog):
        self.log = log
        self.payloads: dict[int, str] = {}
        self.fail_delete = False

    async def insert_batch(self, vectors, ids):
        for int_id, payload in zip(ids, vectors):
            self.payloads[int(int_id)] = payload

    async def delete(self, ids):
        int_ids = [int(i) for i in ids]
        self.log.record("vector_delete", tuple(int_ids))
        if self.fail_delete:
            raise RuntimeError("vector delete failed")
        for int_id in int_ids:
            self.payloads.pop(int_id, None)


class FakeDocumentStorage:
    def __init__(self, log: EventLog):
        self.log = log
        self.rows: dict[str, dict] = {}
        self._next_id = 1
        self.fail_delete_doc_ids: set[str] = set()

    async def insert_documents_batch(self, doc_ids, texts, metadatas):
        result = []
        for doc_id, text, metadata in zip(doc_ids, texts, metadatas):
            int_id = self._next_id
            self._next_id += 1
            self.rows[doc_id] = {
                "id": int_id,
                "doc_id": doc_id,
                "text": text,
                "metadata": json.dumps(metadata),
            }
            result.append(int_id)
        return result

    async def get_document_by_doc_id(self, doc_id):
        row = self.rows.get(doc_id)
        return dict(row) if row is not None else None

    async def get_documents(self, metadata_filters, ids=None, offset=0, limit=100):
        wanted = None if ids is None else {int(i) for i in ids if int(i) != -1}
        result = []
        for row in self.rows.values():
            metadata = json.loads(row["metadata"])
            if any(
                metadata.get(key) != value for key, value in metadata_filters.items()
            ):
                continue
            if wanted is not None and row["id"] not in wanted:
                continue
            result.append(dict(row))
        if offset is not None:
            result = result[offset:]
        if limit is not None:
            result = result[:limit]
        return result

    async def delete_document_by_doc_id(self, doc_id):
        self.log.record("text_delete", doc_id)
        if doc_id in self.fail_delete_doc_ids:
            raise RuntimeError("text delete failed")
        self.rows.pop(doc_id, None)

    async def delete_documents(self, metadata_filters):
        removed = []
        for doc_id, row in list(self.rows.items()):
            metadata = json.loads(row["metadata"])
            if all(
                metadata.get(key) == value for key, value in metadata_filters.items()
            ):
                removed.append(doc_id)
                self.rows.pop(doc_id, None)
        self.log.record("text_delete_batch", tuple(removed))

    async def count_documents(self, metadata_filters=None):
        filters = metadata_filters or {}
        total = 0
        for row in self.rows.values():
            metadata = json.loads(row["metadata"])
            if all(metadata.get(key) == value for key, value in filters.items()):
                total += 1
        return total


class FakeVecDB:
    def __init__(self, log: EventLog):
        self.log = log
        self.document_storage = FakeDocumentStorage(log)
        self.embedding_storage = FakeEmbeddingStorage(log)
        self.fail_insert = False
        self.fail_insert_after_write = False
        self.closed = False

    async def close(self):
        self.closed = True

    async def insert_batch(
        self,
        contents,
        metadatas,
        ids=None,
        batch_size=32,
        tasks_limit=3,
        max_retries=3,
        progress_callback=None,
        embedding_contents=None,
    ):
        if self.fail_insert:
            raise RuntimeError("embedding service unavailable")
        if embedding_contents is None:
            embedding_contents = contents
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in contents]
        if not (len(contents) == len(metadatas) == len(ids) == len(embedding_contents)):
            raise ValueError("insert_batch length mismatch")
        int_ids = await self.document_storage.insert_documents_batch(
            ids,
            contents,
            metadatas,
        )
        await self.embedding_storage.insert_batch(embedding_contents, int_ids)
        if self.fail_insert_after_write:
            # Mirrors a native compensation failure: the new row/vector is
            # already committed even though the call raises.
            raise RuntimeError("compensation failed after write")
        if progress_callback is not None:
            await progress_callback(len(contents), len(contents))
        return int_ids

    async def delete_documents(self, metadata_filters):
        docs = await self.document_storage.get_documents(
            metadata_filters,
            offset=None,
            limit=None,
        )
        await self.embedding_storage.delete([doc["id"] for doc in docs])
        await self.document_storage.delete_documents(metadata_filters)

    async def count_documents(self, metadata_filter=None):
        return await self.document_storage.count_documents(metadata_filter or {})


class FakeSession:
    def __init__(self, db: FakeKBDatabase):
        self._db = db

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    def add(self, obj):
        self._db.added.append(obj)

    async def execute(self, statement):
        self._db.execute_native(statement)

    async def commit(self):
        if self._db.fail_commit:
            raise RuntimeError("commit failed")
        return None

    async def refresh(self, obj):
        return None


class FakeKBDatabase:
    def __init__(self, log: EventLog):
        self.log = log
        self.kbs: dict[str, FakeKB] = {}
        self.documents: dict[str, FakeDocument] = {}
        self.media: dict[str, FakeMedia] = {}
        self.added: list[object] = []
        self.fail_commit = False
        self.fail_media_purge = False

    def execute_native(self, statement):
        """Interpret the backend's ``delete(KBMedia)`` purge statement.

        The backend uses the real native ``KBMedia`` model, so the double only
        needs to honour the table name and the ``kb_id`` filter; any other
        statement is ignored.
        """

        table = getattr(getattr(statement, "table", None), "name", None)
        if table != "kb_media":
            return
        if self.fail_media_purge:
            raise RuntimeError("media purge failed")
        kb_id = next(
            (
                value
                for key, value in statement.compile().params.items()
                if key.startswith("kb_id")
            ),
            None,
        )
        if kb_id is None:
            return
        for media_id in [
            record.media_id for record in self.media.values() if record.kb_id == kb_id
        ]:
            self.media.pop(media_id, None)
        self.log.record("media_purge", kb_id)

    async def get_kb_by_id(self, kb_id):
        return self.kbs.get(kb_id)

    async def get_kb_by_name(self, kb_name):
        for kb in self.kbs.values():
            if kb.kb_name == kb_name:
                return kb
        return None

    async def get_document_by_id(self, doc_id):
        return self.documents.get(doc_id)

    async def list_documents_by_kb(self, kb_id, offset=0, limit=100, search=None):
        docs = [doc for doc in self.documents.values() if doc.kb_id == kb_id]
        if search:
            docs = [doc for doc in docs if search in (doc.doc_name or "")]
        docs.sort(key=lambda doc: doc.created_at, reverse=True)
        return docs[offset : offset + limit]

    async def count_documents_by_kb(self, kb_id, search=None):
        docs = [doc for doc in self.documents.values() if doc.kb_id == kb_id]
        if search:
            docs = [doc for doc in docs if search in (doc.doc_name or "")]
        return len(docs)

    async def list_media_by_doc(self, doc_id):
        return [record for record in self.media.values() if record.doc_id == doc_id]

    async def delete_document_by_id(self, doc_id, vec_db):
        self.log.record("metadata_delete", doc_id)
        for media_id in [
            record.media_id for record in self.media.values() if record.doc_id == doc_id
        ]:
            self.media.pop(media_id, None)
        self.documents.pop(doc_id, None)
        await vec_db.delete_documents({"kb_doc_id": doc_id})

    async def update_kb_stats(self, kb_id, vec_db):
        kb = self.kbs.get(kb_id)
        if kb is None:
            return
        kb.doc_count = len(
            [doc for doc in self.documents.values() if doc.kb_id == kb_id],
        )
        kb.chunk_count = await vec_db.count_documents({"kb_id": kb_id})

    def get_db(self):
        return FakeSession(self)


def _chunk_text(text, chunk_size, chunk_overlap):
    text = text.strip()
    if not text:
        return []
    step = max(1, chunk_size - chunk_overlap)
    return [text[index : index + chunk_size] for index in range(0, len(text), step)]


class FakeKBHelper:
    def __init__(self, kb, kb_db, vec_db, kb_dir):
        self.kb = kb
        self.kb_db = kb_db
        self.vec_db = vec_db
        self.kb_dir = Path(kb_dir)
        self.init_error = None
        self.upload_hook = None
        self.upload_calls = []
        self.fail_refresh_kb = False
        self.fail_refresh_document = False
        self.fail_get_chunk_count = False

    async def upload_document(
        self,
        file_name,
        file_content,
        file_type,
        chunk_size=512,
        chunk_overlap=50,
        batch_size=32,
        tasks_limit=3,
        max_retries=3,
        progress_callback=None,
        pre_chunked_text=None,
    ):
        self.upload_calls.append(
            {
                "file_name": file_name,
                "file_type": file_type,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            },
        )
        if self.upload_hook is not None:
            return await self.upload_hook(
                file_name,
                file_content,
                file_type,
                chunk_size,
                chunk_overlap,
                progress_callback,
            )
        text = (file_content or b"").decode("utf-8", errors="replace")
        chunks = _chunk_text(text, chunk_size, chunk_overlap)
        doc_id = str(uuid.uuid4())
        metadatas = [
            {"kb_id": self.kb.kb_id, "kb_doc_id": doc_id, "chunk_index": index}
            for index in range(len(chunks))
        ]
        ids = [str(uuid.uuid4()) for _ in chunks]
        title = Path(file_name).stem.strip()
        embedding_contents = [
            f"{title}\n\n{chunk}" if title else chunk for chunk in chunks
        ]
        await self.vec_db.insert_batch(
            contents=chunks,
            metadatas=metadatas,
            ids=ids,
            embedding_contents=embedding_contents,
        )
        doc = FakeDocument(
            doc_id,
            self.kb.kb_id,
            file_name,
            file_type,
            len(file_content or b""),
            len(chunks),
        )
        self.kb_db.documents[doc_id] = doc
        await self.kb_db.update_kb_stats(self.kb.kb_id, self.vec_db)
        await self.refresh_kb()
        return doc

    async def list_documents(self, offset=0, limit=100, search=None):
        return await self.kb_db.list_documents_by_kb(
            self.kb.kb_id,
            offset,
            limit,
            search=search,
        )

    async def count_documents(self, search=None):
        return await self.kb_db.count_documents_by_kb(self.kb.kb_id, search=search)

    async def get_document(self, doc_id):
        return await self.kb_db.get_document_by_id(doc_id)

    async def get_chunks_by_doc_id(self, doc_id, offset=0, limit=100):
        rows = await self.vec_db.document_storage.get_documents(
            {"kb_doc_id": doc_id},
            offset=offset,
            limit=limit,
        )
        result = []
        for row in rows:
            metadata = json.loads(row["metadata"])
            result.append(
                {
                    "chunk_id": row["doc_id"],
                    "doc_id": metadata["kb_doc_id"],
                    "kb_id": metadata["kb_id"],
                    "chunk_index": metadata["chunk_index"],
                    "content": row["text"],
                    "char_count": len(row["text"]),
                },
            )
        return result

    async def get_chunk_count_by_doc_id(self, doc_id):
        if self.fail_get_chunk_count:
            raise RuntimeError("chunk count unavailable")
        return await self.vec_db.count_documents({"kb_doc_id": doc_id})

    async def refresh_kb(self):
        if self.fail_refresh_kb:
            raise RuntimeError("kb refresh failed")
        kb = await self.kb_db.get_kb_by_id(self.kb.kb_id)
        if kb is not None:
            self.kb = kb

    async def refresh_document(self, doc_id):
        if self.fail_refresh_document:
            raise RuntimeError("document refresh failed")
        doc = await self.get_document(doc_id)
        if doc is None:
            raise ValueError(f"document not found: {doc_id}")
        doc.chunk_count = await self.get_chunk_count_by_doc_id(doc_id)


class FakeManager:
    def __init__(self, kb_db, kb_root):
        self.kb_db = kb_db
        self.kb_root = Path(kb_root)
        self.log = kb_db.log
        self.kb_insts: dict[str, FakeKBHelper] = {}
        self.retrieve_calls = []
        self.retrieve_payload = None
        self.delete_kb_calls = []
        self.fail_delete_kb = False
        self.delete_kb_exc = None

    def add_kb(self, kb, helper):
        self.kb_db.kbs[kb.kb_id] = kb
        self.kb_insts[kb.kb_id] = helper

    async def get_kb(self, kb_id):
        return self.kb_insts.get(kb_id)

    async def get_kb_by_name(self, kb_name):
        for helper in self.kb_insts.values():
            if helper.kb.kb_name == kb_name:
                return helper
        return None

    async def list_kbs(self):
        return [helper.kb for helper in self.kb_insts.values()]

    async def create_kb(
        self,
        kb_name,
        description=None,
        emoji=None,
        embedding_provider_id=None,
        chunk_size=512,
        chunk_overlap=50,
        **kwargs,
    ):
        if await self.get_kb_by_name(kb_name) is not None:
            raise ValueError(f"知识库名称 '{kb_name}' 已存在")
        kb = FakeKB(
            str(uuid.uuid4()),
            kb_name,
            embedding_provider_id=embedding_provider_id or "",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        kb.description = description
        kb_dir = self.kb_root / kb.kb_id
        kb_dir.mkdir(parents=True, exist_ok=True)
        helper = FakeKBHelper(kb, self.kb_db, FakeVecDB(self.log), kb_dir)
        self.add_kb(kb, helper)
        return helper

    async def delete_kb(self, kb_id):
        self.delete_kb_calls.append(kb_id)
        self.log.record("manager_delete_kb", kb_id)
        if self.delete_kb_exc is not None:
            raise self.delete_kb_exc
        if self.fail_delete_kb:
            return False
        helper = self.kb_insts.pop(kb_id, None)
        self.kb_db.kbs.pop(kb_id, None)
        return helper is not None

    async def retrieve(self, query, kb_names, top_k_fusion=20, top_m_final=5):
        self.retrieve_calls.append(
            {
                "query": query,
                "kb_names": list(kb_names),
                "top_k_fusion": top_k_fusion,
                "top_m_final": top_m_final,
            },
        )
        if self.retrieve_payload is not None:
            return self.retrieve_payload
        results = []
        for helper in self.kb_insts.values():
            if helper.kb.kb_name not in kb_names:
                continue
            rows = await helper.vec_db.document_storage.get_documents(
                {},
                offset=None,
                limit=None,
            )
            for row in rows:
                metadata = json.loads(row["metadata"])
                if query and query not in row["text"]:
                    continue
                doc = await helper.get_document(metadata["kb_doc_id"])
                results.append(
                    {
                        "chunk_id": row["doc_id"],
                        "doc_id": metadata["kb_doc_id"],
                        "kb_id": metadata["kb_id"],
                        "kb_name": helper.kb.kb_name,
                        "doc_name": doc.doc_name if doc else "",
                        "chunk_index": metadata["chunk_index"],
                        "content": row["text"],
                        "score": 0.75,
                        "char_count": len(row["text"]),
                    },
                )
        if not results:
            return None
        return {"context_text": "", "results": results[:top_m_final]}


class FakeContext:
    def __init__(self, manager, embedding_providers=None, chat_providers=None):
        self.kb_manager = manager
        self._embedding_providers = (
            embedding_providers if isinstance(embedding_providers, list) else []
        )
        self.provider_manager = SimpleNamespace(
            embedding_provider_insts=list(self._embedding_providers),
            provider_insts=list(chat_providers or []),
            inst_map={},
        )

    def get_all_embedding_providers(self):
        return list(self._embedding_providers)

    def get_provider_by_id(self, provider_id):
        for provider in self._embedding_providers:
            if provider.provider_config.get("id") == provider_id:
                return provider
        return None


# ---------------------------------------------------------------------------
# Fixtures and small helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def env(tmp_path):
    log = EventLog()
    db = FakeKBDatabase(log)
    manager = FakeManager(db, tmp_path / "kb")
    providers = [FakeEmbeddingProvider("emb-1")]
    context = FakeContext(manager, providers)
    backend = NativeKBBackend(context, default_embedding_provider_id="")
    return SimpleNamespace(
        backend=backend,
        manager=manager,
        db=db,
        context=context,
        log=log,
        providers=providers,
        tmp_path=tmp_path,
    )


async def _create_kb(env, name="kb1", **kwargs):
    result = await env.backend.create_kb(name=name, **kwargs)
    return result["kb"]["kb_id"]


async def _add_doc(env, kb_id, filename="doc.txt", content=b"alpha beta gamma"):
    result = await env.backend.add_document(kb_id, filename, content)
    return result["document"]["doc_id"]


def _assert_json_clean(payload):
    dumped = json.dumps(payload)
    assert "file_path" not in dumped
    assert "kb_dir" not in dumped
    return dumped


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def test_initialize_validates_existing_manager_and_is_idempotent(env):
    assert await env.backend.initialize() == {"initialized": True}
    assert await env.backend.initialize() == {"initialized": True}
    assert env.manager.delete_kb_calls == []


async def test_initialize_without_manager_raises_backend_unavailable():
    backend = NativeKBBackend(SimpleNamespace())
    with pytest.raises(KBError) as excinfo:
        await backend.initialize()
    assert excinfo.value.code == "backend_unavailable"


# ---------------------------------------------------------------------------
# KB CRUD and embedding model selection
# ---------------------------------------------------------------------------


async def test_create_kb_falls_back_to_unique_embedding_provider(env):
    kb_id = await _create_kb(env, name="unique")
    assert env.manager.kb_insts[kb_id].kb.embedding_provider_id == "emb-1"


async def test_create_kb_explicit_provider_wins_over_default(env, tmp_path):
    env.providers.append(FakeEmbeddingProvider("emb-2"))
    backend = NativeKBBackend(env.context, default_embedding_provider_id="emb-1")
    result = await backend.create_kb(name="explicit", embedding_provider_id="emb-2")
    assert result["kb"]["embedding_provider_id"] == "emb-2"


async def test_create_kb_unavailable_default_falls_back_to_unique(env):
    backend = NativeKBBackend(env.context, default_embedding_provider_id="missing")
    result = await backend.create_kb(name="fallback")
    assert result["kb"]["embedding_provider_id"] == "emb-1"


async def test_create_kb_without_provider_returns_candidates(env):
    context = FakeContext(env.manager, [])
    backend = NativeKBBackend(context)
    with pytest.raises(KBError) as excinfo:
        await backend.create_kb(name="nope")
    error = excinfo.value
    assert error.code == "embedding_provider_missing"
    assert error.details["candidates"] == []


async def test_create_kb_multiple_providers_requires_explicit_choice(env):
    env.providers.append(FakeEmbeddingProvider("emb-2"))
    with pytest.raises(KBError) as excinfo:
        await env.backend.create_kb(name="ambiguous")
    assert excinfo.value.code == "embedding_provider_missing"
    assert {item["id"] for item in excinfo.value.details["candidates"]} == {
        "emb-1",
        "emb-2",
    }


async def test_create_kb_never_treats_chat_model_as_embedding(env):
    context = FakeContext(
        env.manager,
        [],
        chat_providers=[FakeChatProvider("chat-1")],
    )
    backend = NativeKBBackend(context)
    with pytest.raises(KBError) as excinfo:
        await backend.create_kb(name="chatless", embedding_provider_id="chat-1")
    assert excinfo.value.code == "embedding_provider_missing"


async def test_create_kb_name_conflict(env):
    await _create_kb(env, name="dup")
    with pytest.raises(KBError) as excinfo:
        await _create_kb(env, name="dup")
    assert excinfo.value.code == "name_conflict"


async def test_create_kb_validates_inputs(env):
    with pytest.raises(KBError):
        await env.backend.create_kb(name="")
    with pytest.raises(KBError):
        await env.backend.create_kb(name="bad", chunk_size=10, chunk_overlap=10)
    with pytest.raises(KBError):
        await env.backend.create_kb(name="typed", description=5)
    with pytest.raises(KBError):
        await env.backend.create_kb(name="typed2", embedding_provider_id=123)


async def test_create_kb_validates_native_name_constraints(env):
    for bad in ("   ", "x" * 101, "bad\nname", "bad\x00name", 123, None):
        with pytest.raises(KBError) as excinfo:
            await env.backend.create_kb(name=bad)
        assert excinfo.value.code == "invalid_argument"

    exact = "y" * 100
    kb_id = await _create_kb(env, name=exact)
    assert env.db.kbs[kb_id].kb_name == exact

    trimmed_id = await _create_kb(env, name="  padded  ")
    assert env.db.kbs[trimmed_id].kb_name == "padded"

    with pytest.raises(KBError) as excinfo:
        await env.backend.create_kb(name=" padded ")
    assert excinfo.value.code == "name_conflict"


async def test_list_kbs_returns_kbs_and_embedding_providers(env):
    env.providers.append(FakeEmbeddingProvider("emb-2"))
    backend = NativeKBBackend(env.context, default_embedding_provider_id="emb-1")
    await backend.create_kb(name="listed")
    result = await backend.list_kbs()
    assert [item["name"] for item in result["kbs"]] == ["listed"]
    assert {item["id"] for item in result["embedding_providers"]} == {"emb-1", "emb-2"}
    assert result["default_embedding_provider_id"] == "emb-1"
    _assert_json_clean(result)


async def test_update_kb_only_touches_given_fields(env):
    kb_id = await _create_kb(env, name="upd", chunk_size=128, chunk_overlap=16)
    result = await env.backend.update_kb(kb_id, description="hello")
    assert result["kb"]["description"] == "hello"
    assert result["kb"]["chunk_size"] == 128
    assert result["kb"]["chunk_overlap"] == 16
    result = await env.backend.update_kb(kb_id, chunk_size=256)
    assert result["kb"]["chunk_size"] == 256
    assert result["kb"]["chunk_overlap"] == 16
    assert env.db.kbs[kb_id].description == "hello"
    assert env.db.added  # persisted through the native session
    with pytest.raises(KBError):
        await env.backend.update_kb(kb_id, chunk_size=8, chunk_overlap=8)


# ---------------------------------------------------------------------------
# Documents: list / read / search
# ---------------------------------------------------------------------------


async def test_list_documents_search_and_pagination(env):
    kb_id = await _create_kb(env)
    for name in ("alpha.txt", "beta.txt", "gamma.txt"):
        await _add_doc(env, kb_id, filename=name)
    result = await env.backend.list_documents(kb_id, offset=0, limit=2)
    assert result["total"] == 3
    assert len(result["documents"]) == 2
    assert result["offset"] == 0 and result["limit"] == 2
    narrowed = await env.backend.list_documents(kb_id, search="beta")
    assert narrowed["total"] == 1
    assert narrowed["documents"][0]["filename"] == "beta.txt"
    _assert_json_clean(result)


async def test_list_and_read_pagination_validation(env):
    kb_id = await _create_kb(env)
    doc_id = await _add_doc(env, kb_id)
    with pytest.raises(KBError):
        await env.backend.list_documents(kb_id, offset=-1)
    with pytest.raises(KBError):
        await env.backend.list_documents(kb_id, limit=0)
    with pytest.raises(KBError):
        await env.backend.list_documents(kb_id, limit=101)
    with pytest.raises(KBError):
        await env.backend.read_document(kb_id, doc_id, limit=101)
    with pytest.raises(KBError):
        await env.backend.list_documents(kb_id, search=123)


async def test_read_document_sorts_by_chunk_index_then_pages(env):
    kb_id = await _create_kb(env)
    helper = env.manager.kb_insts[kb_id]
    doc_id = "seeded-doc"
    env.db.documents[doc_id] = FakeDocument(doc_id, kb_id, "seed.txt", chunk_count=3)
    for chunk_index in (2, 0, 1):
        await helper.vec_db.insert_batch(
            contents=[f"content-{chunk_index}"],
            metadatas=[
                {"kb_id": kb_id, "kb_doc_id": doc_id, "chunk_index": chunk_index},
            ],
            ids=[f"chunk-{chunk_index}"],
        )
    first = await env.backend.read_document(kb_id, doc_id, offset=0, limit=2)
    assert first["total"] == 3
    assert [chunk["index"] for chunk in first["chunks"]] == [0, 1]
    assert [chunk["content"] for chunk in first["chunks"]] == ["content-0", "content-1"]
    second = await env.backend.read_document(kb_id, doc_id, offset=2, limit=2)
    assert [chunk["index"] for chunk in second["chunks"]] == [2]
    _assert_json_clean(first)


async def test_read_document_enforces_kb_ownership(env):
    kb1 = await _create_kb(env, name="kb-one")
    kb2 = await _create_kb(env, name="kb-two")
    foreign_doc = await _add_doc(env, kb2)
    with pytest.raises(KBError) as excinfo:
        await env.backend.read_document(kb1, foreign_doc)
    assert excinfo.value.code == "document_not_found"
    result = await env.backend.read_document(kb2, foreign_doc)
    assert result["document"]["kb_id"] == kb2


async def test_search_filters_foreign_hits_and_keeps_native_score(env):
    kb1 = await _create_kb(env, name="kb-one")
    kb2 = await _create_kb(env, name="kb-two")
    doc1 = await _add_doc(env, kb1, content=b"alpha shared")
    doc2 = await _add_doc(env, kb2, content=b"alpha shared")
    rows = await env.manager.kb_insts[kb1].vec_db.document_storage.get_documents(
        {"kb_doc_id": doc1},
        offset=None,
        limit=None,
    )
    chunk1 = rows[0]["doc_id"]
    env.manager.retrieve_payload = {
        "results": [
            {
                "kb_id": kb1,
                "doc_id": doc1,
                "chunk_id": chunk1,
                "content": "alpha shared",
                "score": 0.875,
                "doc_name": "doc.txt",
                "chunk_index": 0,
            },
            {
                "kb_id": kb1,
                "doc_id": doc2,  # document actually lives in kb2
                "chunk_id": "forged",
                "content": "alpha shared",
                "score": 0.5,
            },
            {
                "kb_id": kb2,  # not requested
                "doc_id": doc2,
                "chunk_id": "other",
                "content": "alpha shared",
                "score": 0.25,
            },
        ],
    }
    result = await env.backend.search("alpha", [kb1], top_k=3)
    assert [hit["chunk_id"] for hit in result["results"]] == [chunk1]
    assert result["results"][0]["score"] == 0.875
    assert result["results"][0]["index"] == 0
    assert env.manager.retrieve_calls[-1]["top_m_final"] == 3
    _assert_json_clean(result)


async def test_search_runs_native_hybrid_pipeline(env):
    kb_id = await _create_kb(env, name="hybrid")
    await _add_doc(env, kb_id, content=b"delta epsilon zeta")
    result = await env.backend.search("epsilon", [kb_id], top_k=5)
    assert result["results"]
    assert result["results"][0]["content"] == "delta epsilon zeta"
    assert result["results"][0]["score"] == 0.75


async def test_search_validates_arguments(env):
    kb_id = await _create_kb(env)
    with pytest.raises(KBError):
        await env.backend.search("", [kb_id])
    with pytest.raises(KBError):
        await env.backend.search("q", [])
    with pytest.raises(KBError):
        await env.backend.search("q", [kb_id], top_k=0)
    with pytest.raises(KBError):
        await env.backend.search("q", [kb_id], top_k=51)
    with pytest.raises(KBError):
        await env.backend.search("q", ["missing-kb"])


# ---------------------------------------------------------------------------
# Document writes
# ---------------------------------------------------------------------------


async def test_add_document_lands_in_all_three_stores(env):
    kb_id = await _create_kb(env, name="adder", chunk_size=16, chunk_overlap=4)
    result = await env.backend.add_document(
        kb_id,
        "notes.txt",
        b"alpha beta gamma delta",
    )
    doc_id = result["document"]["doc_id"]
    assert result["document"]["filename"] == "notes.txt"
    assert result["chunk_count"] >= 1
    stored = env.db.documents[doc_id]
    assert stored.kb_id == kb_id
    helper = env.manager.kb_insts[kb_id]
    rows = await helper.vec_db.document_storage.get_documents(
        {"kb_doc_id": doc_id},
        offset=None,
        limit=None,
    )
    assert rows
    for row in rows:
        metadata = json.loads(row["metadata"])
        assert metadata["kb_id"] == kb_id
        assert helper.vec_db.embedding_storage.payloads[row["id"]].startswith(
            "notes\n\n",
        )
    assert helper.upload_calls[-1]["chunk_size"] == 16
    assert helper.upload_calls[-1]["chunk_overlap"] == 4
    _assert_json_clean(result)


async def test_add_document_rejects_empty_or_unsupported_input(env):
    kb_id = await _create_kb(env)
    with pytest.raises(KBError):
        await env.backend.add_document(kb_id, "", b"x")
    with pytest.raises(KBError):
        await env.backend.add_document(kb_id, "doc.txt", b"")
    with pytest.raises(KBError):
        await env.backend.add_document(kb_id, "binary.bin", b"\x00\x01")


async def test_add_document_partial_when_committed_then_stats_fail(env):
    kb_id = await _create_kb(env, name="partial")
    helper = env.manager.kb_insts[kb_id]

    async def hook(
        file_name,
        file_content,
        file_type,
        chunk_size,
        chunk_overlap,
        progress_callback,
    ):
        text = file_content.decode("utf-8", errors="replace")
        chunks = _chunk_text(text, chunk_size, chunk_overlap) or [text]
        doc_id = "committed-doc"
        metadatas = [
            {"kb_id": kb_id, "kb_doc_id": doc_id, "chunk_index": index}
            for index in range(len(chunks))
        ]
        ids = [str(uuid.uuid4()) for _ in chunks]
        await helper.vec_db.insert_batch(
            contents=chunks,
            metadatas=metadatas,
            ids=ids,
        )
        helper.kb_db.documents[doc_id] = FakeDocument(
            doc_id,
            kb_id,
            file_name,
            file_type,
            len(file_content),
            len(chunks),
        )
        raise FakeUploadError(
            "statistics refresh failed",
            stage="metadata",
            details={"doc_id": doc_id},
        )

    helper.upload_hook = hook
    with pytest.raises(KBError) as excinfo:
        await env.backend.add_document(kb_id, "stats.txt", b"committed content")
    error = excinfo.value
    assert error.partial is True
    assert error.details["doc_id"] == "committed-doc"
    assert "committed-doc" in env.db.documents  # was really committed


async def test_add_document_failure_without_commit_is_not_partial(env):
    kb_id = await _create_kb(env, name="plainfail")
    helper = env.manager.kb_insts[kb_id]

    async def hook(*_args, **_kwargs):
        raise FakeUploadError("storage exploded", stage="storage", details={})

    helper.upload_hook = hook
    with pytest.raises(KBError) as excinfo:
        await env.backend.add_document(kb_id, "fail.txt", b"will not commit")
    assert excinfo.value.partial is False
    assert not env.db.documents


async def test_replace_document_imports_new_then_removes_old(env):
    kb_id = await _create_kb(env, name="replacer", chunk_size=12, chunk_overlap=0)
    old_doc = await _add_doc(env, kb_id, filename="old.txt", content=b"old content")
    old_rows = await env.manager.kb_insts[kb_id].vec_db.document_storage.get_documents(
        {"kb_doc_id": old_doc},
        offset=None,
        limit=None,
    )
    old_chunk_ids = {row["doc_id"] for row in old_rows}
    result = await env.backend.replace_document(
        kb_id,
        old_doc,
        "new.txt",
        b"brand new content",
    )
    new_doc = result["document"]["doc_id"]
    assert new_doc != old_doc
    assert result["old_doc_id"] == old_doc
    assert result["chunk_count"] >= 1
    assert old_doc not in env.db.documents
    assert new_doc in env.db.documents
    rows = await env.manager.kb_insts[kb_id].vec_db.document_storage.get_documents(
        {"kb_doc_id": new_doc},
        offset=None,
        limit=None,
    )
    assert rows and all(row["doc_id"] not in old_chunk_ids for row in rows)
    assert not await env.manager.kb_insts[kb_id].vec_db.document_storage.get_documents(
        {"kb_doc_id": old_doc},
        offset=None,
        limit=None,
    )
    _assert_json_clean(result)


async def test_replace_document_committed_error_keeps_old_document(env):
    kb_id = await _create_kb(env, name="replacer2")
    old_doc = await _add_doc(env, kb_id, filename="old.txt", content=b"old content")
    helper = env.manager.kb_insts[kb_id]
    old_chunks = await helper.get_chunks_by_doc_id(old_doc, offset=None, limit=None)

    async def hook(
        file_name,
        file_content,
        file_type,
        chunk_size,
        chunk_overlap,
        progress_callback,
    ):
        text = file_content.decode("utf-8", errors="replace")
        chunks = _chunk_text(text, chunk_size, chunk_overlap) or [text]
        doc_id = "new-committed"
        metadatas = [
            {"kb_id": kb_id, "kb_doc_id": doc_id, "chunk_index": index}
            for index in range(len(chunks))
        ]
        ids = [str(uuid.uuid4()) for _ in chunks]
        await helper.vec_db.insert_batch(
            contents=chunks,
            metadatas=metadatas,
            ids=ids,
        )
        helper.kb_db.documents[doc_id] = FakeDocument(
            doc_id,
            kb_id,
            file_name,
            file_type,
            len(file_content),
            len(chunks),
        )
        raise FakeUploadError(
            "statistics refresh failed",
            stage="metadata",
            details={"doc_id": doc_id},
        )

    helper.upload_hook = hook
    with pytest.raises(KBError) as excinfo:
        await env.backend.replace_document(kb_id, old_doc, "new.txt", b"new content")
    error = excinfo.value
    assert error.partial is True
    assert error.details["new_doc_id"] == "new-committed"
    assert error.details["old_doc_id"] == old_doc
    # Both documents are preserved: the committed new one and the untouched old.
    assert old_doc in env.db.documents
    assert "new-committed" in env.db.documents
    remaining = await helper.get_chunks_by_doc_id(old_doc, offset=None, limit=None)
    assert len(remaining) == len(old_chunks)


async def test_replace_document_enforces_ownership(env):
    kb1 = await _create_kb(env, name="own-one")
    kb2 = await _create_kb(env, name="own-two")
    foreign = await _add_doc(env, kb2)
    with pytest.raises(KBError) as excinfo:
        await env.backend.replace_document(kb1, foreign, "x.txt", b"x")
    assert excinfo.value.code == "document_not_found"


async def test_delete_document_removes_vectors_before_metadata(env):
    kb_id = await _create_kb(env, name="deleter")
    doc_id = await _add_doc(env, kb_id)
    env.log.events.clear()
    result = await env.backend.delete_document(kb_id, doc_id)
    assert result == {"deleted": True, "doc_id": doc_id}
    assert env.log.first("vector_delete") < env.log.first("text_delete_batch")
    assert env.log.first("text_delete_batch") < env.log.first("metadata_delete")
    assert doc_id not in env.db.documents
    assert (
        await env.manager.kb_insts[kb_id].vec_db.count_documents(
            {"kb_id": kb_id},
        )
        == 0
    )


async def test_delete_document_partial_when_native_delete_fails(env):
    kb_id = await _create_kb(env, name="delfail")
    doc_id = await _add_doc(env, kb_id)
    env.manager.kb_insts[kb_id].vec_db.embedding_storage.fail_delete = True
    with pytest.raises(KBError) as excinfo:
        await env.backend.delete_document(kb_id, doc_id)
    error = excinfo.value
    assert error.partial is True
    assert error.details["doc_id"] == doc_id
    assert doc_id in env.db.documents  # metadata untouched


async def test_delete_document_enforces_ownership(env):
    kb1 = await _create_kb(env, name="own-doc-one")
    kb2 = await _create_kb(env, name="own-doc-two")
    foreign = await _add_doc(env, kb2)
    with pytest.raises(KBError) as excinfo:
        await env.backend.delete_document(kb1, foreign)
    assert excinfo.value.code == "document_not_found"


async def test_delete_document_cleans_only_media_inside_kb_dir(env):
    kb_id = await _create_kb(env, name="media")
    doc_id = await _add_doc(env, kb_id)
    helper = env.manager.kb_insts[kb_id]
    inside = helper.kb_dir / "medias" / doc_id / "a.png"
    inside.parent.mkdir(parents=True, exist_ok=True)
    inside.write_bytes(b"inside")
    outside = env.tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    env.db.media["m-in"] = FakeMedia("m-in", doc_id, kb_id, inside)
    env.db.media["m-out"] = FakeMedia("m-out", doc_id, kb_id, outside)
    await env.backend.delete_document(kb_id, doc_id)
    assert not inside.exists()
    assert outside.exists()
    assert env.db.media == {}


# ---------------------------------------------------------------------------
# Whole-KB deletion
# ---------------------------------------------------------------------------


async def test_delete_kb_purges_documents_media_and_vectors(env):
    kb_id = await _create_kb(env, name="purge")
    doc_id = await _add_doc(env, kb_id)
    helper = env.manager.kb_insts[kb_id]
    media_file = helper.kb_dir / "medias" / doc_id / "m.png"
    media_file.parent.mkdir(parents=True, exist_ok=True)
    media_file.write_bytes(b"m")
    env.db.media["m1"] = FakeMedia("m1", doc_id, kb_id, media_file)
    result = await env.backend.delete_kb(kb_id)
    assert result["deleted"] is True
    assert result["documents_deleted"] == 1
    assert kb_id not in env.manager.kb_insts
    assert kb_id not in env.db.kbs
    assert env.db.documents == {}
    assert env.db.media == {}
    assert env.manager.delete_kb_calls == [kb_id]
    assert not media_file.exists()


async def test_delete_kb_purges_orphan_media_without_touching_other_kb(env):
    kb_one = await _create_kb(env, name="orphan-one")
    kb_two = await _create_kb(env, name="orphan-two")
    helper_one = env.manager.kb_insts[kb_one]
    helper_two = env.manager.kb_insts[kb_two]
    orphan_one = helper_one.kb_dir / "medias" / "gone-doc" / "a.png"
    orphan_one.parent.mkdir(parents=True, exist_ok=True)
    orphan_one.write_bytes(b"a")
    orphan_two = helper_two.kb_dir / "medias" / "gone-doc" / "b.png"
    orphan_two.parent.mkdir(parents=True, exist_ok=True)
    orphan_two.write_bytes(b"b")
    env.db.media["orphan-a"] = FakeMedia("orphan-a", "gone-doc", kb_one, orphan_one)
    env.db.media["orphan-b"] = FakeMedia("orphan-b", "gone-doc", kb_two, orphan_two)
    env.log.events.clear()
    result = await env.backend.delete_kb(kb_one)
    assert result["deleted"] is True
    assert "orphan-a" not in env.db.media
    assert "orphan-b" in env.db.media
    assert env.db.media["orphan-b"].kb_id == kb_two
    assert env.log.first("media_purge") < env.log.first("manager_delete_kb")


async def test_delete_kb_media_cleanup_failure_is_partial(env):
    kb_id = await _create_kb(env, name="media-purge-fail")
    env.db.fail_media_purge = True
    with pytest.raises(KBError) as excinfo:
        await env.backend.delete_kb(kb_id)
    error = excinfo.value
    assert error.partial is True
    assert error.details["kb_id"] == kb_id
    assert error.details["stage"] == "media_cleanup"
    assert env.manager.delete_kb_calls == []
    assert kb_id in env.manager.kb_insts


async def test_delete_kb_stops_before_manager_on_cleanup_failure(env):
    kb_id = await _create_kb(env, name="purgefail")
    doc_id = await _add_doc(env, kb_id)
    env.manager.kb_insts[kb_id].vec_db.embedding_storage.fail_delete = True
    with pytest.raises(KBError) as excinfo:
        await env.backend.delete_kb(kb_id)
    error = excinfo.value
    assert error.partial is True
    assert doc_id in error.details["remaining_document_ids"]
    assert env.manager.delete_kb_calls == []
    assert kb_id in env.manager.kb_insts


# ---------------------------------------------------------------------------
# Chunk writes
# ---------------------------------------------------------------------------


async def test_add_chunk_appends_with_next_index_and_title_embedding(env):
    kb_id = await _create_kb(env, name="chunky", chunk_size=8, chunk_overlap=0)
    doc_id = await _add_doc(env, kb_id, filename="notes.txt", content=b"one two")
    helper = env.manager.kb_insts[kb_id]
    before = await helper.get_chunks_by_doc_id(doc_id, offset=None, limit=None)
    result = await env.backend.add_chunk(kb_id, doc_id, "delta content")
    assert result["chunk"]["index"] == len(before)
    assert result["chunk"]["doc_id"] == doc_id
    row = await helper.vec_db.document_storage.get_document_by_doc_id(
        result["chunk"]["chunk_id"],
    )
    assert row is not None
    assert row["text"] == "delta content"
    assert helper.vec_db.embedding_storage.payloads[row["id"]] == (
        "notes\n\ndelta content"
    )
    with pytest.raises(KBError):
        await env.backend.add_chunk(kb_id, doc_id, "")


async def test_update_chunk_replaces_id_keeps_index_and_ordering(env):
    kb_id = await _create_kb(env, name="updchunk", chunk_size=8, chunk_overlap=0)
    doc_id = await _add_doc(env, kb_id, filename="notes.txt", content=b"one two three")
    helper = env.manager.kb_insts[kb_id]
    chunks = await helper.get_chunks_by_doc_id(doc_id, offset=None, limit=None)
    target = chunks[0]
    old_row = await helper.vec_db.document_storage.get_document_by_doc_id(
        target["chunk_id"],
    )
    old_int_id = old_row["id"]
    env.log.events.clear()
    result = await env.backend.update_chunk(
        kb_id,
        doc_id,
        target["chunk_id"],
        "replacement text",
    )
    new_chunk_id = result["chunk"]["chunk_id"]
    assert new_chunk_id != target["chunk_id"]
    assert result["replaced_chunk_id"] == target["chunk_id"]
    assert result["chunk"]["index"] == target["chunk_index"]
    assert (
        await helper.vec_db.document_storage.get_document_by_doc_id(
            target["chunk_id"],
        )
        is None
    )
    new_row = await helper.vec_db.document_storage.get_document_by_doc_id(new_chunk_id)
    assert new_row is not None
    assert new_row["text"] == "replacement text"
    assert helper.vec_db.embedding_storage.payloads[new_row["id"]] == (
        "notes\n\nreplacement text"
    )
    assert old_int_id not in helper.vec_db.embedding_storage.payloads
    # old removal deletes the vector before the text row
    assert env.log.first("vector_delete") < env.log.first("text_delete")


async def test_update_chunk_insert_failure_keeps_old_content(env):
    kb_id = await _create_kb(env, name="updkeep", chunk_size=8, chunk_overlap=0)
    doc_id = await _add_doc(env, kb_id, filename="notes.txt", content=b"one two three")
    helper = env.manager.kb_insts[kb_id]
    chunks = await helper.get_chunks_by_doc_id(doc_id, offset=None, limit=None)
    target = chunks[0]
    helper.vec_db.fail_insert = True
    env.log.events.clear()
    with pytest.raises(KBError) as excinfo:
        await env.backend.update_chunk(kb_id, doc_id, target["chunk_id"], "new text")
    assert excinfo.value.partial is False
    survivor = await helper.vec_db.document_storage.get_document_by_doc_id(
        target["chunk_id"],
    )
    assert survivor is not None
    assert survivor["text"] == target["content"]
    assert env.log.count("vector_delete") == 0
    assert env.log.count("text_delete") == 0


async def test_update_chunk_enforces_doc_and_kb_ownership(env):
    kb1 = await _create_kb(env, name="chunk-one")
    kb2 = await _create_kb(env, name="chunk-two")
    doc1 = await _add_doc(env, kb1)
    doc2 = await _add_doc(env, kb2)
    helper2 = env.manager.kb_insts[kb2]
    foreign_chunk = (await helper2.get_chunks_by_doc_id(doc2, offset=None, limit=None))[
        0
    ]["chunk_id"]
    with pytest.raises(KBError) as excinfo:
        await env.backend.update_chunk(kb1, doc1, foreign_chunk, "x")
    assert excinfo.value.code == "chunk_not_found"
    with pytest.raises(KBError) as excinfo:
        await env.backend.delete_chunk(kb2, doc2, "missing-chunk")
    assert excinfo.value.code == "chunk_not_found"


async def test_delete_chunk_removes_vector_before_text(env):
    kb_id = await _create_kb(env, name="delchunk", chunk_size=8, chunk_overlap=0)
    doc_id = await _add_doc(env, kb_id, content=b"one two three")
    helper = env.manager.kb_insts[kb_id]
    chunks = await helper.get_chunks_by_doc_id(doc_id, offset=None, limit=None)
    target = chunks[0]
    env.log.events.clear()
    result = await env.backend.delete_chunk(kb_id, doc_id, target["chunk_id"])
    assert result == {"deleted": True, "chunk_id": target["chunk_id"]}
    assert env.log.first("vector_delete") < env.log.first("text_delete")
    assert (
        await helper.vec_db.document_storage.get_document_by_doc_id(
            target["chunk_id"],
        )
        is None
    )
    refreshed = env.db.documents[doc_id]
    assert refreshed.chunk_count == len(chunks) - 1


async def test_delete_chunk_rejects_wrong_document_or_kb(env):
    kb_id = await _create_kb(env, name="delchunk2")
    doc_one = await _add_doc(env, kb_id, filename="one.txt")
    doc_two = await _add_doc(env, kb_id, filename="two.txt")
    helper = env.manager.kb_insts[kb_id]
    chunk_two = (await helper.get_chunks_by_doc_id(doc_two, offset=None, limit=None))[
        0
    ]["chunk_id"]
    with pytest.raises(KBError) as excinfo:
        await env.backend.delete_chunk(kb_id, doc_one, chunk_two)
    assert excinfo.value.code == "chunk_not_found"


async def test_snapshot_payloads_are_json_serializable(env):
    kb_id = await _create_kb(env, name="json")
    doc_id = await _add_doc(env, kb_id)
    payloads = [
        await env.backend.initialize(),
        await env.backend.list_kbs(),
        await env.backend.list_documents(kb_id),
        await env.backend.read_document(kb_id, doc_id),
        await env.backend.search("alpha", [kb_id]),
        await env.backend.add_chunk(kb_id, doc_id, "extra"),
        await env.backend.update_kb(kb_id, description="x"),
    ]
    for payload in payloads:
        _assert_json_clean(payload)


async def test_cancellation_propagates(env):
    kb_id = await _create_kb(env, name="cancel")
    helper = env.manager.kb_insts[kb_id]

    async def hook(*_args, **_kwargs):
        raise asyncio.CancelledError

    helper.upload_hook = hook
    with pytest.raises(asyncio.CancelledError):
        await env.backend.add_document(kb_id, "cancel.txt", b"x")


# ---------------------------------------------------------------------------
# Fix batch regressions: statistics partials, vec_db lifecycle, validation
# ---------------------------------------------------------------------------


async def test_initialize_reports_missing_capabilities(env):
    backend = NativeKBBackend(SimpleNamespace(kb_manager=SimpleNamespace()))
    with pytest.raises(KBError) as excinfo:
        await backend.initialize()
    assert excinfo.value.code == "backend_unavailable"
    assert "get_kb" in excinfo.value.details["missing"]


async def test_add_document_rejects_unsafe_filenames(env):
    kb_id = await _create_kb(env, name="names")
    unsafe = (
        "a/b.txt",
        "a\\b.txt",
        "bad\x00name.txt",
        "line\nbreak.txt",
        "x" * 300,
        ".",
        "..",
    )
    for filename in unsafe:
        with pytest.raises(KBError) as excinfo:
            await env.backend.add_document(kb_id, filename, b"content")
        assert excinfo.value.code == "invalid_argument"


async def test_upload_error_does_not_leak_exception_text(env):
    kb_id = await _create_kb(env, name="leak")
    helper = env.manager.kb_insts[kb_id]
    secret = "http://internal.host/fetch?token=SUPERSECRET"

    async def hook(*_args, **_kwargs):
        raise ValueError(f"failed to fetch {secret}")

    helper.upload_hook = hook
    with pytest.raises(KBError) as excinfo:
        await env.backend.add_document(kb_id, "safe.txt", b"data")
    error = excinfo.value
    assert "SUPERSECRET" not in error.message
    assert "SUPERSECRET" not in json.dumps(error.details)
    assert error.details["type"] == "ValueError"


async def test_add_document_releases_replaced_vec_db_on_success(env):
    kb_id = await _create_kb(env, name="vecrel-ok")
    helper = env.manager.kb_insts[kb_id]
    old_vec_db = helper.vec_db
    new_vec_db = FakeVecDB(env.log)

    async def hook(
        file_name,
        file_content,
        file_type,
        chunk_size,
        chunk_overlap,
        progress_callback,
    ):
        helper.vec_db = new_vec_db  # mimic native _ensure_vec_db replacement
        doc = FakeDocument(
            "replaced-doc",
            kb_id,
            file_name,
            file_type,
            len(file_content),
            1,
        )
        env.db.documents["replaced-doc"] = doc
        return doc

    helper.upload_hook = hook
    result = await env.backend.add_document(kb_id, "doc.txt", b"hello")
    assert result["document"]["doc_id"] == "replaced-doc"
    assert old_vec_db.closed is True
    assert new_vec_db.closed is False


async def test_add_document_releases_replaced_vec_db_on_failure(env):
    kb_id = await _create_kb(env, name="vecrel-fail")
    helper = env.manager.kb_insts[kb_id]
    old_vec_db = helper.vec_db
    new_vec_db = FakeVecDB(env.log)

    async def hook(*_args, **_kwargs):
        helper.vec_db = new_vec_db
        raise FakeUploadError("storage exploded", stage="storage", details={})

    helper.upload_hook = hook
    with pytest.raises(KBError):
        await env.backend.add_document(kb_id, "doc.txt", b"hello")
    assert old_vec_db.closed is True
    assert new_vec_db.closed is False


async def test_update_kb_commit_failure_restores_in_memory_state(env):
    kb_id = await _create_kb(
        env,
        name="updfail",
        chunk_size=128,
        chunk_overlap=16,
    )
    env.db.fail_commit = True
    with pytest.raises(KBError) as excinfo:
        await env.backend.update_kb(kb_id, description="changed", chunk_size=256)
    assert excinfo.value.partial is False
    kb = env.manager.kb_insts[kb_id].kb
    assert kb.description is None
    assert kb.chunk_size == 128
    assert kb.chunk_overlap == 16
    env.db.fail_commit = False
    result = await env.backend.update_kb(kb_id, description="ok")
    assert result["kb"]["description"] == "ok"


async def test_add_chunk_stats_failure_is_partial_with_chunk_id(env):
    kb_id = await _create_kb(env, name="chunkstat", chunk_size=8, chunk_overlap=0)
    doc_id = await _add_doc(env, kb_id, content=b"one two")
    helper = env.manager.kb_insts[kb_id]
    helper.fail_refresh_document = True
    with pytest.raises(KBError) as excinfo:
        await env.backend.add_chunk(kb_id, doc_id, "new chunk")
    error = excinfo.value
    assert error.partial is True
    new_chunk_id = error.details["chunk_id"]
    row = await helper.vec_db.document_storage.get_document_by_doc_id(new_chunk_id)
    assert row is not None
    assert row["text"] == "new chunk"


async def test_update_chunk_stats_failure_is_partial_with_ids(env):
    kb_id = await _create_kb(env, name="updstat", chunk_size=8, chunk_overlap=0)
    doc_id = await _add_doc(env, kb_id, content=b"one two three")
    helper = env.manager.kb_insts[kb_id]
    target = (await helper.get_chunks_by_doc_id(doc_id, offset=None, limit=None))[0]
    helper.fail_refresh_kb = True
    with pytest.raises(KBError) as excinfo:
        await env.backend.update_chunk(kb_id, doc_id, target["chunk_id"], "changed")
    error = excinfo.value
    assert error.partial is True
    assert error.details["replaced_chunk_id"] == target["chunk_id"]
    new_chunk_id = error.details["new_chunk_id"]
    assert (
        await helper.vec_db.document_storage.get_document_by_doc_id(target["chunk_id"])
        is None
    )
    assert (
        await helper.vec_db.document_storage.get_document_by_doc_id(new_chunk_id)
        is not None
    )


async def test_delete_chunk_stats_failure_is_partial_with_chunk_id(env):
    kb_id = await _create_kb(env, name="delstat", chunk_size=8, chunk_overlap=0)
    doc_id = await _add_doc(env, kb_id, content=b"one two three")
    helper = env.manager.kb_insts[kb_id]
    target = (await helper.get_chunks_by_doc_id(doc_id, offset=None, limit=None))[0]
    helper.fail_refresh_document = True
    with pytest.raises(KBError) as excinfo:
        await env.backend.delete_chunk(kb_id, doc_id, target["chunk_id"])
    error = excinfo.value
    assert error.partial is True
    assert error.details["chunk_id"] == target["chunk_id"]
    assert (
        await helper.vec_db.document_storage.get_document_by_doc_id(target["chunk_id"])
        is None
    )


async def test_add_document_count_failure_is_partial(env):
    kb_id = await _create_kb(env, name="countfail")
    helper = env.manager.kb_insts[kb_id]
    helper.fail_get_chunk_count = True
    with pytest.raises(KBError) as excinfo:
        await env.backend.add_document(kb_id, "doc.txt", b"content here")
    error = excinfo.value
    assert error.partial is True
    doc_id = error.details["doc_id"]
    assert doc_id in env.db.documents


async def test_replace_document_count_failure_is_partial(env):
    kb_id = await _create_kb(env, name="replcount")
    old_doc = await _add_doc(env, kb_id, filename="old.txt", content=b"old content")
    helper = env.manager.kb_insts[kb_id]
    helper.fail_get_chunk_count = True
    with pytest.raises(KBError) as excinfo:
        await env.backend.replace_document(kb_id, old_doc, "new.txt", b"new content")
    error = excinfo.value
    assert error.partial is True
    assert error.details["old_doc_id"] == old_doc
    new_doc = error.details["doc_id"]
    assert old_doc not in env.db.documents
    assert new_doc in env.db.documents


async def test_delete_kb_manager_exception_is_partial(env):
    kb_id = await _create_kb(env, name="delkbexc")
    await _add_doc(env, kb_id)
    env.manager.delete_kb_exc = RuntimeError("boom")
    with pytest.raises(KBError) as excinfo:
        await env.backend.delete_kb(kb_id)
    error = excinfo.value
    assert error.partial is True
    assert error.details["kb_id"] == kb_id
    assert error.details["stage"] == "delete_kb"
    assert not [doc for doc in env.db.documents.values() if doc.kb_id == kb_id]


async def test_delete_kb_manager_false_is_partial_with_stage(env):
    kb_id = await _create_kb(env, name="delkbfalse")
    env.manager.fail_delete_kb = True
    with pytest.raises(KBError) as excinfo:
        await env.backend.delete_kb(kb_id)
    error = excinfo.value
    assert error.partial is True
    assert error.details["stage"] == "delete_kb"
    assert error.details["kb_id"] == kb_id


async def test_add_chunk_residual_insert_is_partial_with_new_id(env):
    kb_id = await _create_kb(env, name="residual", chunk_size=8, chunk_overlap=0)
    doc_id = await _add_doc(env, kb_id, content=b"one two")
    helper = env.manager.kb_insts[kb_id]
    helper.vec_db.fail_insert_after_write = True
    with pytest.raises(KBError) as excinfo:
        await env.backend.add_chunk(kb_id, doc_id, "residual chunk")
    error = excinfo.value
    assert error.partial is True
    assert error.details["confirmed_residual"] is True
    new_chunk_id = error.details["new_chunk_id"]
    row = await helper.vec_db.document_storage.get_document_by_doc_id(new_chunk_id)
    assert row is not None
    assert row["text"] == "residual chunk"


# ---------------------------------------------------------------------------
# Real native integration (real SQLite/FTS5/FAISS, throwaway ASTRBOT_ROOT)
# ---------------------------------------------------------------------------


class _DeterministicEmbeddingProvider:
    """Minimal embedding provider good enough for the real FAISS pipeline."""

    def __init__(self, provider_id="fake-emb", dim=8):
        self.provider_config = {"id": provider_id, "model": "deterministic"}
        self.model_name = "deterministic"
        self._dim = dim
        self.fail = False

    def get_dim(self):
        return self._dim

    def _vector(self, text):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[index] / 255.0 for index in range(self._dim)]

    async def get_embedding(self, text):
        if self.fail:
            raise RuntimeError("embedding backend down")
        return self._vector(text)

    async def get_embeddings(self, texts):
        if self.fail:
            raise RuntimeError("embedding backend down")
        return [self._vector(text) for text in texts]

    async def get_embeddings_batch(
        self,
        texts,
        batch_size=16,
        tasks_limit=3,
        max_retries=3,
        progress_callback=None,
    ):
        vectors = await self.get_embeddings(texts)
        if progress_callback is not None:
            await progress_callback(len(texts), len(texts))
        return vectors


class _FakeProviderManager:
    def __init__(self, provider):
        self.provider = provider

    async def get_provider_by_id(self, provider_id):
        if provider_id == self.provider.provider_config["id"]:
            return self.provider
        return None


async def test_real_native_create_add_replace_delete_roundtrip(tmp_path, monkeypatch):
    kb_mgr_mod = importlib.import_module("astrbot.core.knowledge_base.kb_mgr")
    kb_root = tmp_path / "knowledge_base"
    kb_root.mkdir(parents=True, exist_ok=True)
    # Redirect the native paths so nothing touches the workspace data dir.
    monkeypatch.setattr(kb_mgr_mod, "FILES_PATH", str(kb_root), raising=True)
    monkeypatch.setattr(kb_mgr_mod, "DB_PATH", kb_root / "kb.db", raising=True)

    provider = _DeterministicEmbeddingProvider()
    manager = kb_mgr_mod.KnowledgeBaseManager(_FakeProviderManager(provider))
    await manager.initialize()
    context = SimpleNamespace(
        kb_manager=manager,
        get_all_embedding_providers=lambda: [provider],
        provider_manager=SimpleNamespace(embedding_provider_insts=[provider]),
    )
    backend = NativeKBBackend(context, default_embedding_provider_id="fake-emb")

    helper1 = await manager.create_kb(
        kb_name="real-one",
        embedding_provider_id="fake-emb",
        chunk_size=64,
        chunk_overlap=8,
    )
    kb1 = helper1.kb.kb_id
    initial_vec_db = helper1.vec_db
    assert initial_vec_db.document_storage.engine is not None

    # Embedding failure still must release the replaced vec_db (WinError-32 cause).
    provider.fail = True
    with pytest.raises(KBError):
        await backend.add_document(kb1, "fail.txt", b"content that fails embedding")
    provider.fail = False
    assert initial_vec_db.document_storage.engine is None
    attempt_vec_db = helper1.vec_db
    assert attempt_vec_db is not initial_vec_db
    assert attempt_vec_db.document_storage.engine is not None

    added = await backend.add_document(kb1, "note.txt", b"alpha beta gamma delta")
    doc_id = added["document"]["doc_id"]
    assert added["chunk_count"] >= 1
    assert attempt_vec_db.document_storage.engine is None
    assert helper1.vec_db.document_storage.engine is not None

    helper2 = await manager.create_kb(
        kb_name="real-two",
        embedding_provider_id="fake-emb",
        chunk_size=64,
        chunk_overlap=8,
    )
    kb2 = helper2.kb.kb_id
    await backend.add_document(kb2, "other.txt", b"epsilon zeta eta theta")
    search = await backend.search("alpha", [kb1, kb2], top_k=5)
    assert search["results"]
    assert all(hit["kb_id"] in {kb1, kb2} for hit in search["results"])

    pre_replace_vec_db = helper1.vec_db
    replaced = await backend.replace_document(
        kb1, doc_id, "note2.txt", b"omega psi chi"
    )
    new_doc = replaced["document"]["doc_id"]
    assert new_doc != doc_id
    assert replaced["old_doc_id"] == doc_id
    assert pre_replace_vec_db.document_storage.engine is None
    assert helper1.vec_db.document_storage.engine is not None

    chunk = await backend.add_chunk(kb1, new_doc, "extra chunk")
    chunk_id = chunk["chunk"]["chunk_id"]
    updated = await backend.update_chunk(kb1, new_doc, chunk_id, "replaced chunk")
    assert updated["replaced_chunk_id"] == chunk_id
    await backend.delete_chunk(kb1, new_doc, updated["chunk"]["chunk_id"])

    with pytest.raises(KBError) as excinfo:
        await backend.delete_document(kb2, new_doc)
    assert excinfo.value.code == "document_not_found"

    # On Windows an unclosed doc.db makes these raise WinError 32.
    assert (await backend.delete_kb(kb2))["deleted"] is True
    assert (await backend.delete_kb(kb1))["deleted"] is True
    assert not (kb_root / kb1).exists()
    assert not (kb_root / kb2).exists()
    await manager.terminate()


async def test_real_delete_kb_purges_orphan_media_rows(tmp_path, monkeypatch):
    """Real SQLite regression: KBMedia rows of the deleted KB must not survive.

    Two knowledge bases are created and each receives a ``KBMedia`` whose
    ``doc_id`` belongs to no document, so document-level cleanup can never
    reach it. Deleting one KB must remove only that KB's media rows and its
    directory while the other KB keeps both.
    """

    from sqlalchemy import select

    kb_mgr_mod = importlib.import_module("astrbot.core.knowledge_base.kb_mgr")
    models_mod = importlib.import_module("astrbot.core.knowledge_base.models")
    kb_root = tmp_path / "knowledge_base"
    kb_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(kb_mgr_mod, "FILES_PATH", str(kb_root), raising=True)
    monkeypatch.setattr(kb_mgr_mod, "DB_PATH", kb_root / "kb.db", raising=True)

    provider = _DeterministicEmbeddingProvider()
    manager = kb_mgr_mod.KnowledgeBaseManager(_FakeProviderManager(provider))
    await manager.initialize()
    context = SimpleNamespace(
        kb_manager=manager,
        get_all_embedding_providers=lambda: [provider],
        provider_manager=SimpleNamespace(embedding_provider_insts=[provider]),
    )
    backend = NativeKBBackend(context, default_embedding_provider_id="fake-emb")

    async def media_kb_ids(kb_id):
        async with manager.kb_db.get_db() as session:
            result = await session.execute(
                select(models_mod.KBMedia).where(models_mod.KBMedia.kb_id == kb_id)
            )
            return {row.kb_id for row in result.scalars().all()}

    try:
        helper_one = await manager.create_kb(
            kb_name="orphan-one",
            embedding_provider_id="fake-emb",
        )
        helper_two = await manager.create_kb(
            kb_name="orphan-two",
            embedding_provider_id="fake-emb",
        )
        kb_one = helper_one.kb.kb_id
        kb_two = helper_two.kb.kb_id

        # Orphaned media: doc_id points at a document that does not exist, so a
        # document-level cleanup would never see these rows.
        async with manager.kb_db.get_db() as session:
            session.add(
                models_mod.KBMedia(
                    doc_id="missing-doc-one",
                    kb_id=kb_one,
                    media_type="image",
                    file_name="a.png",
                    file_path=str(kb_root / kb_one / "medias" / "a.png"),
                    file_size=1,
                    mime_type="image/png",
                ),
            )
            session.add(
                models_mod.KBMedia(
                    doc_id="missing-doc-two",
                    kb_id=kb_two,
                    media_type="image",
                    file_name="b.png",
                    file_path=str(kb_root / kb_two / "medias" / "b.png"),
                    file_size=1,
                    mime_type="image/png",
                ),
            )
            await session.commit()

        assert await media_kb_ids(kb_one) == {kb_one}
        assert await media_kb_ids(kb_two) == {kb_two}

        assert (await backend.delete_kb(kb_one))["deleted"] is True

        assert await media_kb_ids(kb_one) == set()
        assert await media_kb_ids(kb_two) == {kb_two}
        assert not (kb_root / kb_one).exists()
        assert (kb_root / kb_two).exists()
    finally:
        await manager.terminate()
