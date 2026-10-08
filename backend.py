"""NativeKBBackend: drives AstrBot's native knowledge base subsystem.

The backend never re-implements storage. It resolves the framework-owned
``Context.kb_manager`` and drives the native ``KnowledgeBaseManager`` /
``KBHelper`` / ``FaissVecDB`` objects through their public async methods,
returning JSON-serialisable ``dict`` payloads or raising :class:`KBError`.

Cross-store writes (the KB metadata database, the per-KB SQLite text/FTS store
and the FAISS index) cannot share a transaction, so this module follows the
native ordering instead of pretending otherwise:

* new content is written and verified before old content is removed;
* chunk deletion removes the vector before the text/FTS row;
* document deletion removes vectors before metadata rows;
* KB statistics are refreshed after every structural change;
* partially completed writes raise ``KBError(partial=True)`` with the known
  resource ids instead of claiming an atomic rollback.

Only the standard library and ``common`` are imported at module import time so
the unit tests can run without the AstrBot package installed. Native module
imports are avoided entirely by driving duck-typed objects.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from astrbot.api import logger as _logger

from .common import (
    CODE_BACKEND_UNAVAILABLE,
    CODE_CHUNK_NOT_FOUND,
    CODE_DOCUMENT_NOT_FOUND,
    CODE_EMBEDDING_FAILED,
    CODE_EMBEDDING_PROVIDER_MISSING,
    CODE_INTERNAL,
    CODE_INVALID_ARGUMENT,
    CODE_KB_NOT_FOUND,
    CODE_NAME_CONFLICT,
    KBError,
)

__all__ = ["NativeKBBackend"]

# File extensions accepted by the native parsers in AstrBot v4.28.2.
_SUPPORTED_FILE_TYPES = frozenset(
    {"adoc", "docx", "epub", "markdown", "md", "pdf", "rst", "txt", "xls", "xlsx"}
)

_DOCUMENT_BATCH = 100


def _require_non_empty_str(name: str, value: Any) -> str:
    """Validate that ``value`` is a non-empty string."""

    if not isinstance(value, str) or not value:
        raise KBError(
            CODE_INVALID_ARGUMENT,
            f"{name} must be a non-empty string",
            details={"argument": name, "type": type(value).__name__},
        )
    return value


def _require_bytes(name: str, value: Any) -> bytes:
    """Validate that ``value`` is non-empty bytes."""

    if not isinstance(value, (bytes, bytearray)) or not value:
        raise KBError(
            CODE_INVALID_ARGUMENT,
            f"{name} must be non-empty bytes",
            details={"argument": name, "type": type(value).__name__},
        )
    return bytes(value)


def _validate_page(offset: Any, limit: Any) -> None:
    """Validate ``offset >= 0`` and ``1 <= limit <= 100`` (contract 3.3)."""

    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise KBError(
            CODE_INVALID_ARGUMENT,
            "offset must be an integer >= 0",
            details={"argument": "offset"},
        )
    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or limit < 1
        or limit > 100
    ):
        raise KBError(
            CODE_INVALID_ARGUMENT,
            "limit must be an integer between 1 and 100",
            details={"argument": "limit"},
        )


def _validate_top_k(top_k: Any) -> None:
    """Validate ``1 <= top_k <= 50`` (contract 3.4)."""

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1 or top_k > 50:
        raise KBError(
            CODE_INVALID_ARGUMENT,
            "top_k must be an integer between 1 and 50",
            details={"argument": "top_k"},
        )


def _validate_chunk_config(chunk_size: Any, chunk_overlap: Any) -> None:
    """Validate the native chunking constraint ``overlap < size``."""

    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size < 1
    ):
        raise KBError(
            CODE_INVALID_ARGUMENT,
            "chunk_size must be a positive integer",
            details={"argument": "chunk_size"},
        )
    if (
        not isinstance(chunk_overlap, int)
        or isinstance(chunk_overlap, bool)
        or chunk_overlap < 0
    ):
        raise KBError(
            CODE_INVALID_ARGUMENT,
            "chunk_overlap must be a non-negative integer",
            details={"argument": "chunk_overlap"},
        )
    if chunk_overlap >= chunk_size:
        raise KBError(
            CODE_INVALID_ARGUMENT,
            "chunk_overlap must be smaller than chunk_size",
            details={"argument": "chunk_overlap"},
        )


def _file_type(filename: str) -> str:
    """Map a filename to the native ``file_type`` token (no leading dot)."""

    suffix = Path(filename).suffix.lower().lstrip(".")
    if suffix in _SUPPORTED_FILE_TYPES:
        return suffix
    if not suffix:
        return "txt"
    raise KBError(
        CODE_INVALID_ARGUMENT,
        "unsupported file format",
        details={"extension": suffix},
    )


def _to_epoch_ms(value: Any) -> int | None:
    """Convert a datetime / number to Unix epoch milliseconds (UTC)."""

    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return int(value.timestamp() * 1000)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return None


def _provider_id(provider: Any) -> str | None:
    """Best-effort provider id extraction that never raises."""

    config = getattr(provider, "provider_config", None)
    if isinstance(config, dict) and config.get("id"):
        return str(config["id"])
    try:
        meta = provider.meta()
    except Exception:
        meta = None
    meta_id = getattr(meta, "id", None)
    if meta_id:
        return str(meta_id)
    provider_attr = getattr(provider, "id", None)
    if isinstance(provider_attr, str) and provider_attr:
        return provider_attr
    return None


def _provider_name(provider: Any) -> str:
    """Best-effort human-readable provider label."""

    name = getattr(provider, "model_name", "") or ""
    if not name:
        config = getattr(provider, "provider_config", None)
        if isinstance(config, dict):
            name = str(config.get("model", "") or "")
    return name or (_provider_id(provider) or "unknown")


def _upload_error(exc: BaseException) -> KBError:
    """Normalise an upload failure without leaking caller-controlled text.

    Only the native ``KnowledgeBaseUploadError.user_message`` template is
    reused; arbitrary exception text (which may embed URLs, tokens or document
    fragments) is never copied into the error payload. ``stage``, exception
    type and the known ``doc_id`` are preserved for diagnosis.
    """

    details = getattr(exc, "details", None)
    stage = getattr(exc, "stage", None)
    user_message = getattr(exc, "user_message", None)
    message = (
        user_message
        if isinstance(user_message, str) and user_message
        else "document import failed"
    )
    code = CODE_EMBEDDING_FAILED if stage == "embedding" else CODE_INTERNAL
    safe: dict[str, Any] = {"type": type(exc).__name__}
    if isinstance(details, dict) and details.get("doc_id"):
        safe["doc_id"] = str(details["doc_id"])
    if stage:
        safe["stage"] = str(stage)
    return KBError(code, message, details=safe)


def _validate_filename(filename: str) -> str:
    """Reject path separators, control characters and over-long names."""

    _require_non_empty_str("filename", filename)
    if len(filename) > 255:
        raise KBError(
            CODE_INVALID_ARGUMENT,
            "filename must not exceed 255 characters",
            details={"argument": "filename", "length": len(filename)},
        )
    if "/" in filename or "\\" in filename:
        raise KBError(
            CODE_INVALID_ARGUMENT,
            "filename must not contain path separators",
            details={"argument": "filename"},
        )
    if filename in {".", ".."} or any(
        ord(char) < 32 or ord(char) == 127 for char in filename
    ):
        raise KBError(
            CODE_INVALID_ARGUMENT,
            "filename must not contain control characters or be a path segment",
            details={"argument": "filename"},
        )
    return filename


def _validate_kb_name(name: Any) -> str:
    """Validate the native ``kb_name`` constraint and return the trimmed name.

    The native ``KnowledgeBase.kb_name`` column is a non-null 100-character
    string, so names must be non-empty after trimming, at most 100 characters
    and free of control characters.
    """

    if not isinstance(name, str) or not name.strip():
        raise KBError(
            CODE_INVALID_ARGUMENT,
            "name must be a non-empty string",
            details={"argument": "name", "type": type(name).__name__},
        )
    if any(ord(char) < 32 or ord(char) == 127 for char in name):
        raise KBError(
            CODE_INVALID_ARGUMENT,
            "name must not contain control characters",
            details={"argument": "name"},
        )
    trimmed = name.strip()
    if len(trimmed) > 100:
        raise KBError(
            CODE_INVALID_ARGUMENT,
            "name must not exceed 100 characters",
            details={"argument": "name", "length": len(trimmed)},
        )
    return trimmed


def _embedding_text(doc_name: str, content: str) -> str:
    """Mirror native uploads: prepend the document title to embedding text."""

    title = Path(doc_name or "").stem.strip()
    return f"{title}\n\n{content}" if title else content


class NativeKBBackend:
    """Contract implementation for the plugin's native KB operations."""

    def __init__(
        self,
        context: Any,
        default_embedding_provider_id: str = "",
    ) -> None:
        if not isinstance(default_embedding_provider_id, str):
            raise KBError(
                CODE_INVALID_ARGUMENT,
                "default_embedding_provider_id must be a string",
                details={"argument": "default_embedding_provider_id"},
            )
        self._context = context
        self._default_embedding_provider_id = default_embedding_provider_id
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> dict[str, Any]:
        """Validate the framework-owned manager without re-creating anything.

        The backend must not initialise or close shared framework resources;
        it only proves that ``Context.kb_manager`` exposes the native surface
        it drives later. Safe to call repeatedly.
        """

        manager = self._manager()
        required_manager = (
            "get_kb",
            "get_kb_by_name",
            "create_kb",
            "delete_kb",
            "list_kbs",
            "retrieve",
        )
        missing = [
            name
            for name in required_manager
            if not callable(getattr(manager, name, None))
        ]
        kb_db = getattr(manager, "kb_db", None)
        required_kb_db = (
            "get_kb_by_id",
            "get_db",
            "update_kb_stats",
            "list_documents_by_kb",
            "delete_document_by_id",
        )
        if kb_db is not None:
            missing.extend(
                f"kb_db.{name}"
                for name in required_kb_db
                if not callable(getattr(kb_db, name, None))
            )
        else:
            missing.append("kb_db")
        if missing:
            raise KBError(
                CODE_BACKEND_UNAVAILABLE,
                "knowledge base manager is not available",
                details={"type": type(manager).__name__, "missing": missing},
            )
        self._initialized = True
        return {"initialized": True}

    # ------------------------------------------------------------------
    # Knowledge base CRUD
    # ------------------------------------------------------------------

    async def list_kbs(self) -> dict[str, Any]:
        """List every loaded KB plus the selectable embedding providers."""

        manager = self._manager()
        kbs = [self._kb_summary(kb) for kb in (await manager.list_kbs()) or []]
        providers = []
        for provider in self._embedding_providers():
            provider_id = _provider_id(provider)
            if provider_id:
                providers.append(
                    {"id": provider_id, "name": _provider_name(provider)},
                )
        return {
            "kbs": kbs,
            "embedding_providers": providers,
            "default_embedding_provider_id": (
                self._default_embedding_provider_id or None
            ),
        }

    async def create_kb(
        self,
        name: str,
        description: str = "",
        embedding_provider_id: str = "",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> dict[str, Any]:
        """Create a KB through ``KnowledgeBaseManager.create_kb``."""

        name = _validate_kb_name(name)
        if not isinstance(description, str):
            raise KBError(
                CODE_INVALID_ARGUMENT,
                "description must be a string",
                details={"argument": "description"},
            )
        if not isinstance(embedding_provider_id, str):
            raise KBError(
                CODE_INVALID_ARGUMENT,
                "embedding_provider_id must be a string",
                details={"argument": "embedding_provider_id"},
            )
        _validate_chunk_config(chunk_size, chunk_overlap)
        manager = self._manager()
        existing = await manager.get_kb_by_name(name)
        if existing is not None:
            raise KBError(
                CODE_NAME_CONFLICT,
                "knowledge base name already exists",
                details={"name": name},
            )
        resolved_id = self._resolve_embedding_provider_id(embedding_provider_id)
        try:
            helper = await manager.create_kb(
                kb_name=name,
                description=description or None,
                embedding_provider_id=resolved_id,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        except asyncio.CancelledError:
            raise
        except ValueError as exc:
            message = str(exc)
            if "已存在" in message or "exist" in message.lower():
                raise KBError(
                    CODE_NAME_CONFLICT,
                    "knowledge base name already exists",
                    details={"name": name},
                ) from exc
            raise KBError(
                CODE_EMBEDDING_PROVIDER_MISSING,
                "failed to initialise the embedding provider",
                details={"embedding_provider_id": resolved_id},
            ) from exc
        except Exception as exc:
            _logger.error("create_kb failed", exc_info=True)
            raise KBError(
                CODE_INTERNAL,
                "failed to create the knowledge base",
                details={"type": type(exc).__name__},
            ) from exc
        return {"kb": self._kb_summary(helper.kb)}

    async def update_kb(
        self,
        kb_id: str,
        description: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> dict[str, Any]:
        """Update description / chunking; only ``None`` fields are skipped."""

        helper = await self._get_kb(kb_id)
        if description is not None and not isinstance(description, str):
            raise KBError(
                CODE_INVALID_ARGUMENT,
                "description must be a string or None",
                details={"argument": "description"},
            )
        kb = helper.kb
        effective_size = kb.chunk_size if chunk_size is None else chunk_size
        effective_overlap = kb.chunk_overlap if chunk_overlap is None else chunk_overlap
        _validate_chunk_config(effective_size, effective_overlap)
        previous = {
            "description": kb.description,
            "chunk_size": kb.chunk_size,
            "chunk_overlap": kb.chunk_overlap,
        }
        if description is not None:
            kb.description = description or None
        if chunk_size is not None:
            kb.chunk_size = chunk_size
        if chunk_overlap is not None:
            kb.chunk_overlap = chunk_overlap
        kb_db = self._kb_db(helper)
        committed = False
        try:
            async with kb_db.get_db() as session:
                session.add(kb)
                await session.commit()
                committed = True
                refresh = getattr(session, "refresh", None)
                if callable(refresh):
                    await refresh(kb)
        except asyncio.CancelledError:
            if not committed:
                kb.description = previous["description"]
                kb.chunk_size = previous["chunk_size"]
                kb.chunk_overlap = previous["chunk_overlap"]
            raise
        except Exception as exc:
            if not committed:
                # The shared in-memory KB must not keep values that were never
                # persisted.
                kb.description = previous["description"]
                kb.chunk_size = previous["chunk_size"]
                kb.chunk_overlap = previous["chunk_overlap"]
            _logger.error("update_kb persistence failed: kb=%s", kb_id, exc_info=True)
            raise KBError(
                CODE_INTERNAL,
                "failed to persist the knowledge base update",
                details={
                    "kb_id": kb_id,
                    "committed": committed,
                    "type": type(exc).__name__,
                },
                partial=committed,
            ) from exc
        return {"kb": self._kb_summary(kb)}

    async def delete_kb(self, kb_id: str) -> dict[str, Any]:
        """Cascade documents/media records, then delete through the manager."""

        helper = await self._get_kb(kb_id)
        vec_db = self._require_vec_db(helper)
        kb_db = self._kb_db(helper)
        failures: list[str] = []
        deleted_count = 0
        while True:
            docs = await kb_db.list_documents_by_kb(kb_id, 0, _DOCUMENT_BATCH)
            if not docs:
                break
            progressed = False
            for doc in list(docs):
                try:
                    await self._delete_document_data(
                        helper,
                        vec_db,
                        doc.doc_id,
                        refresh_stats=False,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    _logger.error(
                        "failed to delete document during KB cleanup: kb=%s doc=%s",
                        kb_id,
                        doc.doc_id,
                        exc_info=True,
                    )
                    failures.append(doc.doc_id)
                else:
                    deleted_count += 1
                    progressed = True
            if not progressed:
                break
        if failures:
            raise KBError(
                CODE_INTERNAL,
                "failed to clean up all documents before deleting the KB",
                details={
                    "kb_id": kb_id,
                    "stage": "document_cleanup",
                    "remaining_document_ids": failures,
                },
                partial=True,
            )
        try:
            await self._purge_kb_media(kb_db, kb_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _logger.error(
                "failed to purge media records before KB deletion: kb=%s",
                kb_id,
                exc_info=True,
            )
            raise KBError(
                CODE_INTERNAL,
                "failed to clean up media records before deleting the KB",
                details={
                    "kb_id": kb_id,
                    "stage": "media_cleanup",
                    "type": type(exc).__name__,
                },
                partial=True,
            ) from exc
        try:
            deleted = await self._manager().delete_kb(kb_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _logger.error(
                "native delete_kb failed after document cleanup: kb=%s",
                kb_id,
                exc_info=True,
            )
            raise KBError(
                CODE_INTERNAL,
                "documents were cleaned up but KB deletion failed",
                details={
                    "kb_id": kb_id,
                    "stage": "delete_kb",
                    "type": type(exc).__name__,
                },
                partial=True,
            ) from exc
        if not deleted:
            raise KBError(
                CODE_INTERNAL,
                "native delete_kb returned False after cleanup",
                details={"kb_id": kb_id, "stage": "delete_kb"},
                partial=True,
            )
        return {"deleted": True, "kb_id": kb_id, "documents_deleted": deleted_count}

    # ------------------------------------------------------------------
    # Document reads
    # ------------------------------------------------------------------

    async def list_documents(
        self,
        kb_id: str,
        offset: int = 0,
        limit: int = 20,
        search: str = "",
    ) -> dict[str, Any]:
        """List document summaries with native ordering and totals."""

        _validate_page(offset, limit)
        if not isinstance(search, str):
            raise KBError(
                CODE_INVALID_ARGUMENT,
                "search must be a string",
                details={"argument": "search"},
            )
        helper = await self._get_kb(kb_id)
        search_term = search or None
        docs = await helper.list_documents(
            offset=offset,
            limit=limit,
            search=search_term,
        )
        total = await helper.count_documents(search=search_term)
        return {
            "documents": [self._doc_summary(doc) for doc in docs or []],
            "total": total,
            "offset": offset,
            "limit": limit,
        }

    async def read_document(
        self,
        kb_id: str,
        doc_id: str,
        offset: int = 0,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Read chunks sorted by ``chunk_index`` before pagination."""

        _validate_page(offset, limit)
        helper = await self._get_kb(kb_id)
        doc = await self._require_document(helper, kb_id, doc_id)
        raw_chunks = await helper.get_chunks_by_doc_id(
            doc_id,
            offset=None,
            limit=None,
        )
        owned = [
            chunk
            for chunk in raw_chunks or []
            if chunk.get("doc_id") == doc_id and chunk.get("kb_id") == kb_id
        ]
        # The default native storage order is unordered; sort the full set
        # before slicing so pagination is stable.
        owned.sort(key=lambda chunk: int(chunk.get("chunk_index", 0)))
        page = owned[offset : offset + limit]
        return {
            "document": self._doc_summary(doc),
            "chunks": [self._chunk_view(chunk) for chunk in page],
            "total": len(owned),
            "offset": offset,
            "limit": limit,
        }

    async def search(
        self,
        query: str,
        kb_ids: list[str],
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Run the native hybrid retrieval and keep only owned hits."""

        _require_non_empty_str("query", query)
        _validate_top_k(top_k)
        if not isinstance(kb_ids, list) or not kb_ids:
            raise KBError(
                CODE_INVALID_ARGUMENT,
                "kb_ids must be a non-empty list",
                details={"argument": "kb_ids"},
            )
        manager = self._manager()
        helpers: dict[str, Any] = {}
        names: list[str] = []
        for kb_id in kb_ids:
            helper = await self._get_kb(kb_id)
            helpers[kb_id] = helper
            names.append(helper.kb.kb_name)
        try:
            payload = await manager.retrieve(
                query=query,
                kb_names=names,
                top_k_fusion=20,
                top_m_final=top_k,
            )
        except asyncio.CancelledError:
            raise
        except ValueError as exc:
            raise KBError(
                CODE_BACKEND_UNAVAILABLE,
                "none of the requested knowledge bases are usable",
                details={"kb_ids": list(kb_ids)},
            ) from exc
        raw_results = (payload or {}).get("results") or []
        doc_cache: dict[str, Any] = {}
        hits: list[dict[str, Any]] = []
        for item in raw_results:
            kb_id = item.get("kb_id")
            doc_id = item.get("doc_id")
            chunk_id = item.get("chunk_id")
            if kb_id not in helpers or not doc_id or not chunk_id:
                continue
            helper = helpers[kb_id]
            if doc_id not in doc_cache:
                doc_cache[doc_id] = await helper.get_document(doc_id)
            doc = doc_cache[doc_id]
            if doc is None or getattr(doc, "kb_id", None) != kb_id:
                continue
            hits.append(
                {
                    "kb_id": kb_id,
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "content": item.get("content", ""),
                    "score": float(item.get("score", 0.0)),
                    "filename": item.get("doc_name") or getattr(doc, "doc_name", None),
                    "index": item.get("chunk_index"),
                },
            )
        return {"query": query, "results": hits}

    # ------------------------------------------------------------------
    # Document writes
    # ------------------------------------------------------------------

    async def add_document(
        self,
        kb_id: str,
        filename: str,
        content: bytes,
        progress: Any = None,
    ) -> dict[str, Any]:
        """Import a document through the native upload pipeline."""

        _validate_filename(filename)
        _require_bytes("content", content)
        helper = await self._get_kb(kb_id)
        file_type = _file_type(filename)
        doc, partial_error = await self._upload_document(
            helper,
            filename,
            content,
            file_type,
            progress,
        )
        if partial_error is not None:
            raise partial_error
        chunk_count = await self._count_chunks(
            helper,
            kb_id,
            doc.doc_id,
        )
        return {
            "document": self._doc_summary(doc),
            "chunk_count": chunk_count,
        }

    async def replace_document(
        self,
        kb_id: str,
        doc_id: str,
        filename: str,
        content: bytes,
        progress: Any = None,
    ) -> dict[str, Any]:
        """Import the new document first, then remove the old one."""

        _validate_filename(filename)
        _require_bytes("content", content)
        helper = await self._get_kb(kb_id)
        old_doc = await self._require_document(helper, kb_id, doc_id)
        file_type = _file_type(filename)
        new_doc, partial_error = await self._upload_document(
            helper,
            filename,
            content,
            file_type,
            progress,
        )
        if partial_error is not None:
            # The uploaded document was committed, but the native pipeline
            # reported a post-commit failure. Keep the old document intact and
            # report the ambiguous state instead of continuing the replacement.
            partial_error.details = {
                **(partial_error.details or {}),
                "old_doc_id": old_doc.doc_id,
                "new_doc_id": new_doc.doc_id,
            }
            raise partial_error
        vec_db = self._require_vec_db(helper)
        try:
            await self._delete_document_data(helper, vec_db, old_doc.doc_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _logger.error(
                "replace_document imported new doc but failed to remove old: "
                "old=%s new=%s",
                old_doc.doc_id,
                new_doc.doc_id,
                exc_info=True,
            )
            raise KBError(
                CODE_INTERNAL,
                "new document imported but the old document could not be removed",
                details={
                    "old_doc_id": old_doc.doc_id,
                    "new_doc_id": new_doc.doc_id,
                    "type": type(exc).__name__,
                },
                partial=True,
            ) from exc
        chunk_count = await self._count_chunks(
            helper,
            kb_id,
            new_doc.doc_id,
            extra_details={"old_doc_id": old_doc.doc_id},
        )
        return {
            "document": self._doc_summary(new_doc),
            "chunk_count": chunk_count,
            "old_doc_id": old_doc.doc_id,
        }

    async def delete_document(self, kb_id: str, doc_id: str) -> dict[str, Any]:
        """Delete vectors first, then metadata, then media files."""

        helper = await self._get_kb(kb_id)
        doc = await self._require_document(helper, kb_id, doc_id)
        vec_db = self._require_vec_db(helper)
        try:
            await self._delete_document_data(helper, vec_db, doc.doc_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _logger.error(
                "failed to delete document: kb=%s doc=%s",
                kb_id,
                doc.doc_id,
                exc_info=True,
            )
            raise KBError(
                CODE_INTERNAL,
                "failed to delete the document",
                details={
                    "kb_id": kb_id,
                    "doc_id": doc.doc_id,
                    "type": type(exc).__name__,
                },
                partial=True,
            ) from exc
        return {"deleted": True, "doc_id": doc.doc_id}

    # ------------------------------------------------------------------
    # Chunk writes
    # ------------------------------------------------------------------

    async def add_chunk(
        self,
        kb_id: str,
        doc_id: str,
        content: str,
    ) -> dict[str, Any]:
        """Append a chunk that is immediately searchable."""

        _require_non_empty_str("content", content)
        helper = await self._get_kb(kb_id)
        doc = await self._require_document(helper, kb_id, doc_id)
        vec_db = self._require_vec_db(helper)
        chunks = await helper.get_chunks_by_doc_id(doc_id, offset=None, limit=None)
        next_index = (
            max(
                (int(chunk.get("chunk_index", 0)) for chunk in chunks or []),
                default=-1,
            )
            + 1
        )
        new_chunk_id = str(uuid.uuid4())
        metadata = {
            "kb_id": kb_id,
            "kb_doc_id": doc_id,
            "chunk_index": next_index,
        }
        await self._insert_chunk(
            vec_db,
            content,
            metadata,
            new_chunk_id,
            _embedding_text(doc.doc_name, content),
            error_details={"kb_id": kb_id, "doc_id": doc_id},
        )
        await self._refresh_after_chunk_change(
            helper,
            vec_db,
            doc_id,
            resource_ids={"kb_id": kb_id, "doc_id": doc_id, "chunk_id": new_chunk_id},
        )
        return {
            "chunk": {
                "chunk_id": new_chunk_id,
                "doc_id": doc_id,
                "index": next_index,
                "content": content,
            },
        }

    async def update_chunk(
        self,
        kb_id: str,
        doc_id: str,
        chunk_id: str,
        content: str,
    ) -> dict[str, Any]:
        """Replace a chunk with a new UUID, then remove the old one."""

        _require_non_empty_str("content", content)
        helper = await self._get_kb(kb_id)
        doc = await self._require_document(helper, kb_id, doc_id)
        vec_db = self._require_vec_db(helper)
        old_row = await self._require_chunk(vec_db, kb_id, doc_id, chunk_id)
        old_index = self._chunk_index(old_row)
        new_chunk_id = str(uuid.uuid4())
        metadata = {
            "kb_id": kb_id,
            "kb_doc_id": doc_id,
            "chunk_index": old_index,
        }
        await self._insert_chunk(
            vec_db,
            content,
            metadata,
            new_chunk_id,
            _embedding_text(doc.doc_name, content),
            error_details={
                "kb_id": kb_id,
                "doc_id": doc_id,
                "replaced_chunk_id": chunk_id,
            },
        )
        try:
            current = await vec_db.document_storage.get_document_by_doc_id(chunk_id)
            if current is not None:
                await vec_db.embedding_storage.delete([current["id"]])
                await vec_db.document_storage.delete_document_by_doc_id(chunk_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _logger.error(
                "update_chunk stored new chunk but failed to remove old: new=%s old=%s",
                new_chunk_id,
                chunk_id,
                exc_info=True,
            )
            raise KBError(
                CODE_INTERNAL,
                "new chunk stored but the previous chunk could not be removed",
                details={
                    "kb_id": kb_id,
                    "doc_id": doc_id,
                    "new_chunk_id": new_chunk_id,
                    "replaced_chunk_id": chunk_id,
                    "type": type(exc).__name__,
                },
                partial=True,
            ) from exc
        await self._refresh_after_chunk_change(
            helper,
            vec_db,
            doc_id,
            resource_ids={
                "kb_id": kb_id,
                "doc_id": doc_id,
                "new_chunk_id": new_chunk_id,
                "replaced_chunk_id": chunk_id,
            },
        )
        return {
            "chunk": {
                "chunk_id": new_chunk_id,
                "doc_id": doc_id,
                "index": old_index,
                "content": content,
            },
            "replaced_chunk_id": chunk_id,
        }

    async def delete_chunk(
        self,
        kb_id: str,
        doc_id: str,
        chunk_id: str,
    ) -> dict[str, Any]:
        """Remove the vector first, then the text/FTS row."""

        helper = await self._get_kb(kb_id)
        await self._require_document(helper, kb_id, doc_id)
        vec_db = self._require_vec_db(helper)
        row = await self._require_chunk(vec_db, kb_id, doc_id, chunk_id)
        try:
            await vec_db.embedding_storage.delete([row["id"]])
            await vec_db.document_storage.delete_document_by_doc_id(chunk_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _logger.error(
                "failed to delete chunk: kb=%s doc=%s chunk=%s",
                kb_id,
                doc_id,
                chunk_id,
                exc_info=True,
            )
            raise KBError(
                CODE_INTERNAL,
                "failed to delete the chunk",
                details={
                    "kb_id": kb_id,
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "type": type(exc).__name__,
                },
                partial=True,
            ) from exc
        await self._refresh_after_chunk_change(
            helper,
            vec_db,
            doc_id,
            resource_ids={
                "kb_id": kb_id,
                "doc_id": doc_id,
                "chunk_id": chunk_id,
            },
        )
        return {"deleted": True, "chunk_id": chunk_id}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _manager(self) -> Any:
        manager = getattr(self._context, "kb_manager", None)
        if manager is None:
            raise KBError(
                CODE_BACKEND_UNAVAILABLE,
                "context.kb_manager is not available",
            )
        return manager

    def _embedding_providers(self) -> list[Any]:
        getter = getattr(self._context, "get_all_embedding_providers", None)
        if callable(getter):
            try:
                providers = list(getter() or [])
            except Exception:
                providers = []
            if providers:
                return providers
        provider_manager = getattr(self._context, "provider_manager", None)
        return list(getattr(provider_manager, "embedding_provider_insts", []) or [])

    def _resolve_embedding_provider_id(self, explicit: str) -> str:
        by_id: dict[str, Any] = {}
        for provider in self._embedding_providers():
            provider_id = _provider_id(provider)
            if provider_id and provider_id not in by_id:
                by_id[provider_id] = provider
        candidates = [
            {"id": provider_id, "name": _provider_name(provider)}
            for provider_id, provider in by_id.items()
        ]

        def _missing(reason: str) -> KBError:
            return KBError(
                CODE_EMBEDDING_PROVIDER_MISSING,
                reason,
                details={
                    "requested": explicit or None,
                    "default": self._default_embedding_provider_id or None,
                    "candidates": candidates,
                },
            )

        if explicit:
            if explicit in by_id:
                return explicit
            raise _missing("the requested embedding provider is not available")
        default_id = self._default_embedding_provider_id
        if default_id and default_id in by_id:
            return default_id
        if len(by_id) == 1:
            return next(iter(by_id))
        if not by_id:
            raise _missing("no embedding provider is available")
        raise _missing(
            "multiple embedding providers are available; specify embedding_provider_id",
        )

    async def _get_kb(self, kb_id: str) -> Any:
        _require_non_empty_str("kb_id", kb_id)
        helper = await self._manager().get_kb(kb_id)
        if helper is None:
            raise KBError(
                CODE_KB_NOT_FOUND,
                "knowledge base not found",
                details={"kb_id": kb_id},
            )
        return helper

    @staticmethod
    def _kb_db(helper: Any) -> Any:
        kb_db = getattr(helper, "kb_db", None)
        if kb_db is None:
            raise KBError(
                CODE_BACKEND_UNAVAILABLE,
                "knowledge base metadata database is not available",
                details={"kb_id": getattr(getattr(helper, "kb", None), "kb_id", None)},
            )
        return kb_db

    @staticmethod
    def _require_vec_db(helper: Any) -> Any:
        vec_db = getattr(helper, "vec_db", None)
        if vec_db is None:
            raise KBError(
                CODE_BACKEND_UNAVAILABLE,
                "knowledge base vector storage is not initialised",
                details={"kb_id": getattr(getattr(helper, "kb", None), "kb_id", None)},
            )
        return vec_db

    async def _require_document(
        self,
        helper: Any,
        kb_id: str,
        doc_id: str,
    ) -> Any:
        _require_non_empty_str("doc_id", doc_id)
        doc = await helper.get_document(doc_id)
        if doc is None or getattr(doc, "kb_id", None) != kb_id:
            raise KBError(
                CODE_DOCUMENT_NOT_FOUND,
                "document not found in the requested knowledge base",
                details={"kb_id": kb_id, "doc_id": doc_id},
            )
        return doc

    @staticmethod
    async def _require_chunk(
        vec_db: Any,
        kb_id: str,
        doc_id: str,
        chunk_id: str,
    ) -> dict[str, Any]:
        _require_non_empty_str("chunk_id", chunk_id)
        row = await vec_db.document_storage.get_document_by_doc_id(chunk_id)
        metadata: Any = row.get("metadata") if isinstance(row, dict) else None
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except ValueError:
                metadata = None
        if (
            not isinstance(row, dict)
            or not isinstance(metadata, dict)
            or metadata.get("kb_doc_id") != doc_id
            or metadata.get("kb_id") != kb_id
        ):
            raise KBError(
                CODE_CHUNK_NOT_FOUND,
                "chunk not found in the requested document",
                details={"kb_id": kb_id, "doc_id": doc_id, "chunk_id": chunk_id},
            )
        return row

    @staticmethod
    def _chunk_index(row: dict[str, Any]) -> int:
        metadata = row.get("metadata")
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except ValueError:
                metadata = {}
        if isinstance(metadata, dict):
            return int(metadata.get("chunk_index", 0))
        return 0

    async def _upload_document(
        self,
        helper: Any,
        filename: str,
        content: bytes,
        file_type: str,
        progress: Any,
    ) -> tuple[Any, KBError | None]:
        """Run the native upload and release any replaced vector store.

        Native ``KBHelper.upload_document`` calls ``_ensure_vec_db`` which
        builds a brand-new ``FaissVecDB`` on every invocation and overwrites
        ``helper.vec_db`` without closing the previous instance. On Windows
        that leaks an open ``doc.db`` handle, so a later KB removal fails with
        ``WinError 32``. We snapshot the previous instance and close only the
        replaced one once the call returns, including error paths.

        Returns ``(document, partial_error)``. ``partial_error`` is set only
        when the native pipeline raised after the document metadata was
        already committed and verified; the caller decides how to proceed.
        """

        kb = helper.kb
        previous_vec_db = getattr(helper, "vec_db", None)
        try:
            try:
                doc = await helper.upload_document(
                    file_name=filename,
                    file_content=content,
                    file_type=file_type,
                    chunk_size=kb.chunk_size,
                    chunk_overlap=kb.chunk_overlap,
                    progress_callback=progress,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _logger.error(
                    "native document upload failed: kb=%s type=%s",
                    kb.kb_id,
                    type(exc).__name__,
                    exc_info=True,
                )
                error = _upload_error(exc)
                committed_id = None
                if isinstance(error.details, dict):
                    committed_id = error.details.get("doc_id")
                if committed_id:
                    doc = await helper.get_document(committed_id)
                    if doc is not None and getattr(doc, "kb_id", None) == kb.kb_id:
                        return doc, KBError(
                            error.code,
                            error.message,
                            details=error.details,
                            partial=True,
                        )
                raise error from exc
            return doc, None
        finally:
            await self._release_replaced_vec_db(helper, previous_vec_db)

    @staticmethod
    async def _release_replaced_vec_db(helper: Any, previous: Any) -> None:
        """Close the previous vec_db only when the helper swapped it out."""

        if previous is None:
            return
        if getattr(helper, "vec_db", None) is previous:
            return
        close = getattr(previous, "close", None)
        if not callable(close):
            return
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
        except asyncio.CancelledError:
            # Best-effort cleanup while a cancellation unwinds; never mask the
            # original CancelledError raised by the caller.
            return
        except Exception:
            _logger.warning(
                "failed to close replaced vector store: kb=%s",
                getattr(getattr(helper, "kb", None), "kb_id", None),
                exc_info=True,
            )

    async def _count_chunks(
        self,
        helper: Any,
        kb_id: str,
        doc_id: str,
        extra_details: dict[str, Any] | None = None,
    ) -> int:
        """Read the post-write chunk count, reporting partial on failure."""

        try:
            return await helper.get_chunk_count_by_doc_id(doc_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _logger.error(
                "document write applied but chunk count could not be read: "
                "kb=%s doc=%s",
                kb_id,
                doc_id,
                exc_info=True,
            )
            details: dict[str, Any] = {"kb_id": kb_id, "doc_id": doc_id}
            if extra_details:
                details.update(extra_details)
            details["type"] = type(exc).__name__
            raise KBError(
                CODE_INTERNAL,
                "document write applied but chunk count could not be read",
                details=details,
                partial=True,
            ) from exc

    async def _insert_chunk(
        self,
        vec_db: Any,
        content: str,
        metadata: dict[str, Any],
        chunk_id: str,
        embedding_content: str,
        error_details: dict[str, Any],
    ) -> None:
        """Insert a chunk, distinguishing a clean failure from residual data.

        ``FaissVecDB.insert_batch`` writes the text row and the vector in two
        separate commits. If it raises, the newly assigned ``chunk_id`` may
        still exist, so the failure must not be reported as "nothing written".
        """

        try:
            await vec_db.insert_batch(
                contents=[content],
                metadatas=[metadata],
                ids=[chunk_id],
                embedding_contents=[embedding_content],
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _logger.error(
                "failed to store chunk: kb=%s doc=%s",
                error_details.get("kb_id"),
                error_details.get("doc_id"),
                exc_info=True,
            )
            residual: Any = None
            lookup_failed = False
            try:
                residual = await vec_db.document_storage.get_document_by_doc_id(
                    chunk_id,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                lookup_failed = True
            if residual is not None or lookup_failed:
                raise KBError(
                    CODE_EMBEDDING_FAILED,
                    "failed to store the new chunk; it may already be indexed",
                    details={
                        **error_details,
                        "new_chunk_id": chunk_id,
                        "confirmed_residual": residual is not None,
                        "type": type(exc).__name__,
                    },
                    partial=True,
                ) from exc
            raise KBError(
                CODE_EMBEDDING_FAILED,
                "failed to store the new chunk",
                details={
                    **error_details,
                    "new_chunk_id": chunk_id,
                    "type": type(exc).__name__,
                },
            ) from exc

    async def _refresh_after_chunk_change(
        self,
        helper: Any,
        vec_db: Any,
        doc_id: str,
        resource_ids: dict[str, Any],
    ) -> None:
        """Refresh native statistics, reporting partial on a post-write error."""

        try:
            kb_db = self._kb_db(helper)
            await kb_db.update_kb_stats(kb_id=helper.kb.kb_id, vec_db=vec_db)
            await helper.refresh_kb()
            await helper.refresh_document(doc_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _logger.error(
                "chunk change applied but statistics refresh failed: %s",
                resource_ids,
                exc_info=True,
            )
            raise KBError(
                CODE_INTERNAL,
                "chunk change applied but statistics refresh failed",
                details={**resource_ids, "type": type(exc).__name__},
                partial=True,
            ) from exc

    async def _purge_kb_media(self, kb_db: Any, kb_id: str) -> None:
        """Delete every ``kb_media`` row owned by ``kb_id``.

        Native ``KnowledgeBaseManager.delete_kb`` removes the KB row and the KB
        directory but never purges ``kb_media``; document-level deletion removes
        media only for documents that still exist. Media rows whose ``doc_id``
        has already disappeared therefore survive a whole-KB deletion, so they
        are removed here first. The statement is scoped to ``kb_id`` and never
        touches records of another knowledge base.

        Args:
            kb_db: Native metadata database (or an equivalent double).
            kb_id: Knowledge base whose media records must be removed.

        Raises:
            Exception: Propagated to the caller, which maps it to a partial
                ``KBError`` with the ``media_cleanup`` stage.
        """

        from astrbot.core.knowledge_base.models import KBMedia
        from sqlalchemy import delete

        async with kb_db.get_db() as session:
            await session.execute(delete(KBMedia).where(KBMedia.kb_id == kb_id))
            commit = getattr(session, "commit", None)
            if callable(commit):
                await commit()

    async def _delete_document_data(
        self,
        helper: Any,
        vec_db: Any,
        doc_id: str,
        refresh_stats: bool = True,
    ) -> None:
        """Vectors first, then metadata rows, then media files on disk."""

        kb_db = self._kb_db(helper)
        media = await self._list_media(kb_db, doc_id)
        await vec_db.delete_documents(metadata_filters={"kb_doc_id": doc_id})
        await kb_db.delete_document_by_id(doc_id, vec_db)
        self._cleanup_media_files(helper, media)
        if refresh_stats:
            await kb_db.update_kb_stats(kb_id=helper.kb.kb_id, vec_db=vec_db)
            await helper.refresh_kb()

    @staticmethod
    async def _list_media(kb_db: Any, doc_id: str) -> list[Any]:
        lister = getattr(kb_db, "list_media_by_doc", None)
        if not callable(lister):
            return []
        try:
            return list(await lister(doc_id) or [])
        except asyncio.CancelledError:
            raise
        except Exception:
            _logger.warning("failed to list media for document %s", doc_id)
            return []

    @staticmethod
    def _cleanup_media_files(helper: Any, media: list[Any]) -> None:
        kb_dir = getattr(helper, "kb_dir", None)
        if kb_dir is None:
            return
        try:
            root = Path(kb_dir).resolve()
        except OSError:
            return
        for record in media or []:
            file_path = getattr(record, "file_path", None)
            if not file_path:
                continue
            try:
                resolved = Path(file_path).resolve()
            except OSError:
                continue
            if not resolved.is_relative_to(root):
                continue
            try:
                if resolved.is_file():
                    resolved.unlink()
            except OSError:
                _logger.warning("failed to remove media file for %s", root.name)

    @staticmethod
    def _doc_summary(doc: Any) -> dict[str, Any]:
        return {
            "doc_id": doc.doc_id,
            "kb_id": doc.kb_id,
            "filename": doc.doc_name,
            "size": doc.file_size,
            "chunk_count": doc.chunk_count,
            "created_at": _to_epoch_ms(getattr(doc, "created_at", None)),
            "updated_at": _to_epoch_ms(getattr(doc, "updated_at", None)),
        }

    @staticmethod
    def _kb_summary(kb: Any) -> dict[str, Any]:
        return {
            "kb_id": kb.kb_id,
            "name": kb.kb_name,
            "description": kb.description,
            "embedding_provider_id": kb.embedding_provider_id,
            "chunk_size": kb.chunk_size,
            "chunk_overlap": kb.chunk_overlap,
            "document_count": kb.doc_count,
            "chunk_count": kb.chunk_count,
            "created_at": _to_epoch_ms(getattr(kb, "created_at", None)),
            "updated_at": _to_epoch_ms(getattr(kb, "updated_at", None)),
        }

    @staticmethod
    def _chunk_view(chunk: dict[str, Any]) -> dict[str, Any]:
        return {
            "chunk_id": chunk.get("chunk_id"),
            "doc_id": chunk.get("doc_id"),
            "index": int(chunk.get("chunk_index", 0)),
            "content": chunk.get("content", ""),
        }
