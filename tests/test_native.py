"""Native plugin loading and end-to-end acceptance tests (batch A · native).

These tests load the plugin through the *real* AstrBot v4.28.2 ``PluginManager``
in an isolated child process, then drive the real 17 LLM tools end to end over
real SQLite / FTS5 / FAISS storage.

Boundaries:

* ``ASTRBOT_ROOT`` points at a throwaway directory created under ``tmp_path``;
  the plugin's necessary files are copied to
  ``<root>/data/plugins/astrbot_plugin_kb_manager`` and the workspace ``data``
  directory is never written.
* The child process adds the temporary root and ``upstream/AstrBot`` to
  ``sys.path`` and imports ``astrbot.api`` first (upstream import order).
* The embedding provider is a deterministic fake object (no network / keys).
  Web fetching is isolated with a deterministic ``SourceDocument`` double
  rather than claiming a real public fetch.
* ``main.py`` is owned by another task; when it is absent the test fails with a
  clear fixture-status message instead of faking it.

Run with the plugin development interpreter::

    <venv>/Scripts/python.exe -m pytest tests/test_native.py -q
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
UPSTREAM_ROOT = WORKSPACE_ROOT / "upstream" / "AstrBot"
PLUGIN_SRC = Path(__file__).resolve().parents[1]

# Files the real plugin needs inside the temporary ASTRBOT_ROOT.
_COPY_FILES = (
    "main.py",
    "backend.py",
    "sources.py",
    "jobs.py",
    "common.py",
    "metadata.yaml",
    "_conf_schema.json",
    "requirements.txt",
)

_CHILD_SOURCE = r'''
import asyncio
import functools
import hashlib
import inspect
import json
import os
import shutil
import sys
import threading
import traceback
from pathlib import Path
from types import SimpleNamespace

TMP_ROOT, UPSTREAM, PLUGIN_SRC = sys.argv[1], sys.argv[2], sys.argv[3]

# ASTRBOT_ROOT must be set before the first astrbot import so every global
# path constant resolves inside the throwaway root.
os.environ["ASTRBOT_ROOT"] = TMP_ROOT
os.environ["ASTRBOT_RELOAD"] = "0"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for _p in (TMP_ROOT, UPSTREAM):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import astrbot.api  # noqa: E402,F401  (import before other native modules)
from astrbot.api.message_components import File  # noqa: E402
from astrbot.core import astrbot_config, db_helper, sp  # noqa: E402
from astrbot.core.knowledge_base.kb_mgr import KnowledgeBaseManager  # noqa: E402
from astrbot.core.provider.register import llm_tools  # noqa: E402
from astrbot.core.star.context import Context  # noqa: E402
from astrbot.core.star.star import star_map  # noqa: E402
from astrbot.core.star.star_handler import star_handlers_registry  # noqa: E402
from astrbot.core.star.star_manager import PluginManager  # noqa: E402

MODULE = "data.plugins.astrbot_plugin_kb_manager.main"
TOOL_NAMES = [
    "kbm_list_kbs", "kbm_create_kb", "kbm_update_kb", "kbm_delete_kb",
    "kbm_list_documents", "kbm_read_document", "kbm_search", "kbm_add_text",
    "kbm_list_attachments", "kbm_import_attachment", "kbm_import_url",
    "kbm_replace_document", "kbm_delete_document", "kbm_add_chunk",
    "kbm_update_chunk", "kbm_delete_chunk", "kbm_job_status",
]

# Frozen tool contract (required parameters), independent of main.py internals.
TOOL_CONTRACT = {
    "kbm_list_kbs": (),
    "kbm_create_kb": ("request_id", "name"),
    "kbm_update_kb": ("request_id", "kb_id", "changes"),
    "kbm_delete_kb": ("request_id", "kb_id"),
    "kbm_list_documents": ("kb_id",),
    "kbm_read_document": ("kb_id", "doc_id"),
    "kbm_search": ("query", "kb_ids"),
    "kbm_add_text": ("request_id", "kb_id", "filename", "content"),
    "kbm_list_attachments": (),
    "kbm_import_attachment": ("request_id", "kb_id", "attachment_id"),
    "kbm_import_url": ("request_id", "kb_id", "url"),
    "kbm_replace_document": (
        "request_id", "kb_id", "doc_id", "filename", "content",
    ),
    "kbm_delete_document": ("request_id", "kb_id", "doc_id"),
    "kbm_add_chunk": ("request_id", "kb_id", "doc_id", "content"),
    "kbm_update_chunk": ("request_id", "kb_id", "doc_id", "chunk_id", "content"),
    "kbm_delete_chunk": ("request_id", "kb_id", "doc_id", "chunk_id"),
    "kbm_job_status": ("job_id",),
}
ENVELOPE_KEYS = {"status", "job_id", "data", "error"}
TERMINAL_STATUSES = {"succeeded", "failed", "partial", "interrupted"}
NON_TERMINAL_STATUSES = {"queued", "running"}

REPORT = {
    "ok": False,
    "steps": [],
    "error": None,
    "traceback": None,
    "leftover_threads": [],
    "exit_note": None,
}

# Resources opened by the run, tracked so failure paths still clean them up.
_OPEN = {"plugin": None, "kb_manager": None}


def step(name, **info):
    REPORT["steps"].append({"name": name, **info})


class DeterministicEmbeddingProvider:
    provider_config = {"id": "e2e-embedding", "model": "deterministic"}
    model_name = "deterministic"
    _dim = 16

    def get_dim(self):
        return self._dim

    def _vec(self, text):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i] / 255.0 for i in range(self._dim)]

    async def get_embedding(self, text):
        return self._vec(text)

    async def get_embeddings(self, texts):
        return [self._vec(t) for t in texts]

    async def get_embeddings_batch(
        self,
        texts,
        batch_size=16,
        tasks_limit=3,
        max_retries=3,
        progress_callback=None,
    ):
        out = [self._vec(t) for t in texts]
        if progress_callback is not None:
            await progress_callback(len(texts), len(texts))
        return out

    async def test(self):
        return None


class FakeProviderManager:
    def __init__(self, embedding):
        self.embedding_provider_insts = [embedding]
        self.provider_insts = []
        self.stt_provider_insts = []
        self.tts_provider_insts = []
        self.inst_map = {embedding.provider_config["id"]: embedding}
        self.llm_tools = llm_tools

    async def get_provider_by_id(self, provider_id):
        return self.inst_map.get(provider_id)

    def get_using_provider(self, *args, **kwargs):
        return None

    async def get_using_provider_async(self, *args, **kwargs):
        return None


class FakeEvent:
    def __init__(self, umo, sender_id, admin, messages=None):
        self.unified_msg_origin = umo
        self._sender_id = sender_id
        self._admin = admin
        self._messages = list(messages or [])
        self.message_obj = SimpleNamespace(message=self._messages, message_str="")

    def set_messages(self, messages):
        self._messages = list(messages)
        self.message_obj = SimpleNamespace(message=self._messages, message_str="")

    def get_sender_id(self):
        return self._sender_id

    def is_admin(self):
        return self._admin

    def get_messages(self):
        return list(self._messages)

    def get_group_id(self):
        return ""

    def get_platform_name(self):
        return "e2e-platform"

    def get_platform_id(self):
        return "e2e-platform"

    def get_self_id(self):
        return "bot-1"

    def stop_event(self):
        self._stopped = True

    def get_extra(self, key, default=None):
        return default


def tool_function(tool):
    """Return the raw coroutine function behind a tool handler."""

    handler = tool.handler
    if isinstance(handler, functools.partial):
        return handler.func
    return handler


def tool_signature(tool):
    """Derive (allowed, required) parameter names from the real signature."""

    signature = inspect.signature(tool_function(tool))
    allowed, required = set(), set()
    for name, parameter in signature.parameters.items():
        if name in ("self", "event"):
            continue
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        allowed.add(name)
        if parameter.default is inspect.Parameter.empty:
            required.add(name)
    return allowed, required


async def call_tool(tool, event, contracts, bag):
    """Call a tool strictly and validate its envelope.

    Unknown or missing parameters are assertion failures. The tool must return
    a ``str`` holding a JSON object with exactly the four envelope keys; there
    is no lenient fallback.
    """

    allowed, required = tool_signature(tool)
    expected = set(contracts[tool.name])
    assert required == expected, (
        f"{tool.name}: signature required {sorted(required)} != "
        f"contract {sorted(expected)}"
    )
    unknown = set(bag) - allowed
    assert not unknown, f"{tool.name}: unknown parameters {sorted(unknown)}"
    missing = required - set(bag)
    assert not missing, f"{tool.name}: missing required {sorted(missing)}"
    result = tool.handler(event, **bag)
    if hasattr(result, "__aiter__"):
        last = None
        async for item in result:
            last = item
        result = last
    else:
        result = await result
    assert isinstance(result, str), (
        f"{tool.name}: tool must return a str, got {type(result).__name__}"
    )
    try:
        env = json.loads(result)
    except ValueError as exc:
        raise AssertionError(
            f"{tool.name}: result is not JSON: {result[:200]!r}"
        ) from exc
    assert isinstance(env, dict), f"{tool.name}: envelope is not an object"
    assert set(env) == ENVELOPE_KEYS, (
        f"{tool.name}: envelope keys {sorted(env)} != {sorted(ENVELOPE_KEYS)}"
    )
    assert env["status"] in TERMINAL_STATUSES | NON_TERMINAL_STATUSES, (
        f"{tool.name}: invalid status {env['status']!r}"
    )
    return env


def envelope_ok(env):
    return isinstance(env, dict) and env.get("status") == "succeeded"


async def _safe_cleanup():
    """Best-effort release of every resource the run opened."""

    plugin = _OPEN.get("plugin")
    if plugin is not None:
        try:
            await plugin.terminate()
        except Exception:
            traceback.print_exc()
    kb_manager = _OPEN.get("kb_manager")
    if kb_manager is not None:
        try:
            await kb_manager.terminate()
        except Exception:
            traceback.print_exc()
    try:
        await sp.close()
    except Exception:
        traceback.print_exc()
    engine = getattr(db_helper, "engine", None)
    if engine is not None:
        try:
            await engine.dispose()
        except Exception:
            traceback.print_exc()


async def run():
    try:
        await _run_impl()
        REPORT["ok"] = True
    finally:
        await _safe_cleanup()


def main():
    asyncio.run(run())


async def _run_impl():
    embedding = DeterministicEmbeddingProvider()
    provider_manager = FakeProviderManager(embedding)
    kb_manager = KnowledgeBaseManager(provider_manager)
    _OPEN["kb_manager"] = kb_manager

    await db_helper.initialize()
    await sp.initialize()
    await kb_manager.initialize()

    context = Context(
        event_queue=asyncio.Queue(),
        config=astrbot_config,
        db=db_helper,
        provider_manager=provider_manager,
        platform_manager=SimpleNamespace(get_insts=lambda: []),
        conversation_manager=SimpleNamespace(),
        message_history_manager=SimpleNamespace(),
        persona_manager=SimpleNamespace(),
        astrbot_config_mgr=SimpleNamespace(get_conf=lambda umo=None: astrbot_config),
        knowledge_base_manager=kb_manager,
        cron_manager=SimpleNamespace(),
    )
    step("context_built")

    plugin_manager = PluginManager(context, astrbot_config)
    loaded, load_error = await plugin_manager.load(
        specified_dir_name="astrbot_plugin_kb_manager"
    )
    if not loaded or star_map.get(MODULE) is None:
        raise AssertionError(f"plugin load failed: {load_error!r}")
    metadata = star_map[MODULE]
    plugin = metadata.star_cls
    _OPEN["plugin"] = plugin
    step("plugin_loaded", plugin_name=metadata.name, cls=type(plugin).__name__)

    from data.plugins.astrbot_plugin_kb_manager.common import (  # noqa: PLC0415
        SOURCE_URL,
        SourceDocument,
        encode_scope,
    )

    def plugin_managers(instance):
        backend_ = getattr(instance, "_backend", None)
        sources_ = getattr(instance, "_sources", None)
        jobs_ = getattr(instance, "_jobs", None)
        if backend_ is None or sources_ is None or jobs_ is None:
            raise AssertionError("plugin managers not found on instance")
        return backend_, sources_, jobs_

    backend, source_manager, job_manager = plugin_managers(plugin)
    step(
        "managers_bound",
        source_closed=source_manager._closed,
        jobs_closed=job_manager._closed,
    )

    contracts = TOOL_CONTRACT

    def collect_tools():
        found = {name: llm_tools.get_func(name) for name in TOOL_NAMES}
        missing_names = [name for name, tool in found.items() if tool is None]
        if missing_names:
            raise AssertionError(f"missing tools: {missing_names}")
        count = sum(
            1
            for tool in llm_tools.func_list
            if getattr(tool, "handler_module_path", None) == MODULE
        )
        return found, count

    tools, count_first = collect_tools()
    if count_first != len(TOOL_NAMES):
        raise AssertionError(f"expected {len(TOOL_NAMES)} plugin tools, found {count_first}")
    step("tools_registered", count=count_first)

    # Independent schema checks: required names come from the real signatures
    # cross-checked against the frozen tool contract, never from main internals.
    for name, tool in tools.items():
        allowed, required = tool_signature(tool)
        contract = set(contracts[name])
        assert required == contract, (
            f"{name}: signature required {sorted(required)} != "
            f"contract {sorted(contract)}"
        )
        assert contract <= allowed, (
            f"{name}: contract params absent from signature: "
            f"{sorted(contract - allowed)}"
        )
        props = tool.parameters.get("properties", {})
        assert contract <= set(props), (
            f"{name}: required params missing from schema: "
            f"{sorted(contract - set(props))}"
        )
        if contract:
            assert props, f"{name} has an empty schema for its required params"
        rendered = json.dumps(tool.parameters)
        assert "unified_msg_origin" not in rendered, f"{name} leaks the event origin"
        assert props.get("scope") is None, f"{name} exposes scope"
        handler = tool.handler
        assert isinstance(handler, functools.partial) and handler.args, (
            f"{name} handler is not a bound plugin method"
        )
        assert handler.args[0] is plugin, (
            f"{name} handler is bound to a stale plugin instance"
        )
    update_props = tools["kbm_update_kb"].parameters.get("properties", {})
    if "changes" in update_props:
        allowed_changes = update_props["changes"].get("properties", {})
        assert set(allowed_changes) <= {"description", "chunk_size", "chunk_overlap"}, (
            f"update_kb changes exposes {sorted(allowed_changes)}"
        )
    step("schemas_checked")

    admin = FakeEvent("kbm:e2e:admin", "admin-1001", True)
    non_admin = FakeEvent("kbm:e2e:user", "user-2002", False)
    scope = encode_scope(admin.unified_msg_origin, admin.get_sender_id())

    KB_NAME = "native-e2e"
    TEXT = "The zebra marker NATIVEMARKERONE lives in this document."
    MARKER = "NATIVEMARKERONE"

    async def call(tool_name, event, **bag):
        env = await call_tool(tools[tool_name], event, contracts, bag)
        # Writes may return queued/running on a slow machine; poll the real
        # status tool until terminal so assertions see the final outcome.
        if (
            tool_name != "kbm_job_status"
            and env["status"] in NON_TERMINAL_STATUSES
            and env["job_id"]
        ):
            for _ in range(400):
                await asyncio.sleep(0.05)
                env = await call_tool(
                    tools["kbm_job_status"],
                    event,
                    contracts,
                    {"job_id": env["job_id"]},
                )
                if env["status"] in TERMINAL_STATUSES:
                    break
        return env

    # --- create library ---------------------------------------------------
    created = await call(
        "kbm_create_kb",
        admin,
        name=KB_NAME,
        description="native e2e",
        embedding_provider_id="e2e-embedding",
        request_id="req-create",
    )
    assert envelope_ok(created), created
    listed = await call("kbm_list_kbs", admin)
    assert envelope_ok(listed), listed
    kb_id = None
    for item in listed["data"]["kbs"]:
        if item.get("name") == KB_NAME:
            kb_id = item["kb_id"]
    assert kb_id, f"created KB not listed: {listed['data']}"
    step("kb_created", kb_id=kb_id)

    # --- admin writes text ------------------------------------------------
    add_bag = dict(
        request_id="req-add-text",
        kb_id=kb_id,
        filename="native-e2e.txt",
        content=TEXT,
    )
    added = await call("kbm_add_text", admin, **add_bag)
    assert envelope_ok(added), added
    doc_id = added["data"]["document"]["doc_id"]
    step("text_added", doc_id=doc_id)

    docs = await call("kbm_list_documents", admin, kb_id=kb_id)
    assert envelope_ok(docs) and docs["data"]["total"] >= 1, docs

    read = await call("kbm_read_document", admin, kb_id=kb_id, doc_id=doc_id, limit=50)
    assert envelope_ok(read), read
    chunks = read["data"]["chunks"]
    assert chunks, "no chunks read back"
    first_chunk = chunks[0]["chunk_id"]
    step("document_read", chunks=len(chunks))

    # --- chunk add / update / delete -------------------------------------
    added_chunk = await call(
        "kbm_add_chunk",
        admin,
        kb_id=kb_id,
        doc_id=doc_id,
        content="extra NATIVEMARKERTWO chunk",
        request_id="req-add-chunk",
    )
    assert envelope_ok(added_chunk), added_chunk
    new_chunk = added_chunk["data"]["chunk"]["chunk_id"]
    updated = await call(
        "kbm_update_chunk",
        admin,
        kb_id=kb_id,
        doc_id=doc_id,
        chunk_id=new_chunk,
        content="updated NATIVEMARKERTHREE chunk",
        request_id="req-upd-chunk",
    )
    assert envelope_ok(updated), updated
    assert updated["data"]["replaced_chunk_id"] == new_chunk
    final_chunk = updated["data"]["chunk"]["chunk_id"]
    deleted = await call(
        "kbm_delete_chunk",
        admin,
        kb_id=kb_id,
        doc_id=doc_id,
        chunk_id=final_chunk,
        request_id="req-del-chunk",
    )
    assert envelope_ok(deleted), deleted
    step("chunks_mutated")

    # --- search -----------------------------------------------------------
    found = await call("kbm_search", admin, query=MARKER, kb_ids=[kb_id], top_k=5)
    assert envelope_ok(found), found
    assert any(MARKER in hit["content"] for hit in found["data"]["results"]), found
    step("search_found", hits=len(found["data"]["results"]))

    # --- whole-document replace ------------------------------------------
    replaced = await call(
        "kbm_replace_document",
        admin,
        kb_id=kb_id,
        doc_id=doc_id,
        filename="native-e2e-2.txt",
        content="replacement NATIVEMARKERFOUR content",
        request_id="req-replace",
    )
    assert envelope_ok(replaced), replaced
    new_doc = replaced["data"]["document"]["doc_id"]
    assert new_doc != doc_id, "replace must mint a new doc_id"
    assert replaced["data"].get("old_doc_id") == doc_id
    refound = await call("kbm_search", admin, query="NATIVEMARKERFOUR", kb_ids=[kb_id])
    assert envelope_ok(refound) and refound["data"]["results"], refound
    step("document_replaced", new_doc=new_doc)

    # --- update_kb via changes (includes chunk_overlap=0) ------------------
    updated_kb = await call(
        "kbm_update_kb",
        admin,
        request_id="req-update-kb",
        kb_id=kb_id,
        changes={"description": "updated", "chunk_size": 64, "chunk_overlap": 0},
    )
    assert updated_kb["status"] == "succeeded", updated_kb
    listed_after_update = await call("kbm_list_kbs", admin)
    assert envelope_ok(listed_after_update), listed_after_update
    kb_state = next(
        item for item in listed_after_update["data"]["kbs"] if item["kb_id"] == kb_id
    )
    assert kb_state["description"] == "updated", kb_state
    assert kb_state["chunk_size"] == 64, kb_state
    assert kb_state["chunk_overlap"] == 0, kb_state
    step("kb_updated", chunk_overlap=kb_state["chunk_overlap"])

    # --- attachments: real docx + pdf through the registered listener -----
    from docx import Document  # noqa: PLC0415
    from reportlab.pdfgen import canvas  # noqa: PLC0415

    src_dir = Path(TMP_ROOT) / "e2e_src"
    src_dir.mkdir(parents=True, exist_ok=True)

    docx_path = src_dir / "e2e.docx"
    document = Document()
    document.add_paragraph("DOCXMARKER alpha beta")
    document.save(str(docx_path))

    pdf_path = src_dir / "e2e.pdf"
    pdf_canvas = canvas.Canvas(str(pdf_path))
    pdf_canvas.drawString(100, 750, "PDFMARKER gamma delta")
    pdf_canvas.save()

    # Real File message segments on the trusted event; the registered
    # kbm_on_message listener records them (no fake component bypass).
    admin.set_messages(
        [
            File("e2e.docx", file=str(docx_path)),
            File("e2e.pdf", file=str(pdf_path)),
        ]
    )
    message_handler = next(
        handler
        for handler in star_handlers_registry.get_handlers_by_module_name(MODULE)
        if handler.handler_name == "kbm_on_message"
    )
    assert message_handler.handler is not None
    await message_handler.handler(admin)
    attachments = {
        item["filename"]: item["attachment_id"]
        for item in source_manager.list_attachments(scope)
    }
    assert "e2e.docx" in attachments and "e2e.pdf" in attachments, attachments

    listed_att = await call("kbm_list_attachments", admin)
    assert envelope_ok(listed_att), listed_att

    docx_import = await call(
        "kbm_import_attachment",
        admin,
        request_id="req-att-docx",
        kb_id=kb_id,
        attachment_id=attachments["e2e.docx"],
    )
    assert envelope_ok(docx_import), docx_import
    pdf_import = await call(
        "kbm_import_attachment",
        admin,
        request_id="req-att-pdf",
        kb_id=kb_id,
        attachment_id=attachments["e2e.pdf"],
    )
    assert envelope_ok(pdf_import), pdf_import

    docx_hit = await call("kbm_search", admin, query="DOCXMARKER", kb_ids=[kb_id])
    pdf_hit = await call("kbm_search", admin, query="PDFMARKER", kb_ids=[kb_id])
    assert docx_hit["data"]["results"], docx_hit
    assert pdf_hit["data"]["results"], pdf_hit
    step("attachments_imported")

    # --- URL via deterministic SourceDocument double ----------------------
    async def fake_load_url(url):
        return SourceDocument(
            filename="webpage.txt",
            content=b"WEBPAGEMARKER epsilon zeta",
            source=SOURCE_URL,
        )

    source_manager.load_url = fake_load_url
    url_import = await call(
        "kbm_import_url",
        admin,
        kb_id=kb_id,
        url="https://example.invalid/webpage",
        request_id="req-import-url",
    )
    assert envelope_ok(url_import), url_import
    step("url_imported_stub")

    # --- non-admin denied and storage unchanged --------------------------
    before_total = (await call("kbm_list_documents", admin, kb_id=kb_id))["data"]["total"]
    dummy_values = {
        "request_id": "req-denied",
        "kb_id": kb_id,
        "doc_id": new_doc,
        "chunk_id": "denied-chunk",
        "name": "denied",
        "description": "denied",
        "embedding_provider_id": "e2e-embedding",
        "chunk_size": 64,
        "chunk_overlap": 0,
        "changes": {"description": "denied"},
        "filename": "denied.txt",
        "content": "denied",
        "query": "denied",
        "kb_ids": [kb_id],
        "top_k": 3,
        "offset": 0,
        "limit": 20,
        "search": "",
        "attachment_id": attachments["e2e.docx"],
        "url": "https://example.invalid/denied",
        "job_id": "denied-job",
    }
    denied = []
    for name in TOOL_NAMES:
        allowed, required = tool_signature(tools[name])
        bag = {key: dummy_values[key] for key in allowed}
        assert required <= set(bag), f"{name}: dummy bag missing {sorted(required)}"
        env = await call(name, non_admin, **bag)
        if envelope_ok(env):
            denied.append(name)
        else:
            assert env["error"] is not None, f"{name}: denial has no error"
            assert env["error"]["code"] == "permission_denied", (
                f"{name}: expected permission_denied, got {env}"
            )
    after_total = (await call("kbm_list_documents", admin, kb_id=kb_id))["data"]["total"]
    assert not denied, f"non-admin tools succeeded: {denied}"
    assert after_total == before_total, "non-admin call changed storage"
    step("non_admin_denied")

    # --- reload ----------------------------------------------------------
    # Create genuinely idle HTTP sessions to prove reload closes them.
    idle_public = await source_manager._ensure_session(internal=False)
    idle_internal = await source_manager._ensure_session(internal=True)
    assert idle_public is not None and not idle_public.closed
    assert idle_internal is not None and not idle_internal.closed
    old_plugin, old_jobs, old_source = plugin, job_manager, source_manager
    reloaded, reload_error = await plugin_manager.reload("astrbot_plugin_kb_manager")
    if not reloaded or star_map.get(MODULE) is None:
        raise AssertionError(f"plugin reload failed: {reload_error!r}")
    plugin = star_map[MODULE].star_cls
    _OPEN["plugin"] = plugin
    assert plugin is not old_plugin, "reload reused the old instance"
    tools, count_second = collect_tools()
    assert count_second == len(TOOL_NAMES), count_second
    assert old_jobs._closed is True, "old JobManager not closed on reload"
    assert old_jobs._tasks == {}, f"old jobs still running: {list(old_jobs._tasks)}"
    assert old_jobs._db is None, "old JobManager database not released"
    assert old_source._closed is True, "old SourceManager not closed on reload"
    assert old_source._cleanup_task is None, "old cleanup task not cancelled"
    assert old_source._public_session is None, "old public HTTP session not released"
    assert old_source._internal_session is None, "old internal HTTP session released"
    assert old_source._public_resolver is None, "old DNS resolver not released"
    assert idle_public.closed is True, "public HTTP session was not closed"
    assert idle_internal.closed is True, "internal HTTP session was closed"
    rebound = tools["kbm_add_text"].handler
    assert (
        isinstance(rebound, functools.partial) and rebound.args[0] is plugin
    ), "reloaded tools are not rebound to the new plugin instance"
    backend, source_manager, job_manager = plugin_managers(plugin)
    assert job_manager._closed is False
    assert source_manager._closed is False
    step("reloaded", tools=count_second)

    # --- idempotent request_id reuse after reload ------------------------
    reuse = await call("kbm_add_text", admin, **add_bag)
    assert envelope_ok(reuse), reuse
    assert reuse["job_id"] == added["job_id"], "request_id did not reuse the job"
    total_after_reuse = (await call("kbm_list_documents", admin, kb_id=kb_id))["data"]["total"]
    assert total_after_reuse == after_total, "reused request imported a second time"
    step("request_id_reused")

    # --- attachment idempotency without cached source --------------------
    shutil.rmtree(Path(TMP_ROOT) / "e2e_src", ignore_errors=True)
    for cache in (Path(TMP_ROOT) / "data" / "plugin_data").rglob("attachments"):
        shutil.rmtree(cache, ignore_errors=True)
    att_reuse = await call(
        "kbm_import_attachment",
        admin,
        request_id="req-att-docx",
        kb_id=kb_id,
        attachment_id=attachments["e2e.docx"],
    )
    assert envelope_ok(att_reuse), att_reuse
    assert att_reuse["job_id"] == docx_import["job_id"], "attachment request_id was replayed"
    total_after_att = (await call("kbm_list_documents", admin, kb_id=kb_id))["data"]["total"]
    assert total_after_att == after_total, "attachment import re-ran after reload"
    step("attachment_idempotent")

    # --- delete document then whole library ------------------------------
    docs_before_delete = await call("kbm_list_documents", admin, kb_id=kb_id, limit=100)
    assert envelope_ok(docs_before_delete), docs_before_delete
    before_docs_total = docs_before_delete["data"]["total"]
    deleted_doc = await call(
        "kbm_delete_document",
        admin,
        request_id="req-del-doc",
        kb_id=kb_id,
        doc_id=new_doc,
    )
    assert deleted_doc["status"] == "succeeded", deleted_doc
    assert deleted_doc["data"]["deleted"] is True, deleted_doc
    assert deleted_doc["data"]["doc_id"] == new_doc, deleted_doc
    docs_after_delete = await call("kbm_list_documents", admin, kb_id=kb_id, limit=100)
    assert envelope_ok(docs_after_delete), docs_after_delete
    assert docs_after_delete["data"]["total"] == before_docs_total - 1, docs_after_delete
    assert all(
        item["doc_id"] != new_doc for item in docs_after_delete["data"]["documents"]
    ), docs_after_delete
    step("document_deleted")

    deleted_kb = await call(
        "kbm_delete_kb",
        admin,
        request_id="req-del-kb",
        kb_id=kb_id,
    )
    assert deleted_kb["status"] == "succeeded", deleted_kb
    assert deleted_kb["data"]["deleted"] is True, deleted_kb
    remaining = await call("kbm_list_kbs", admin)
    assert all(item["kb_id"] != kb_id for item in remaining["data"]["kbs"])
    storage = Path(TMP_ROOT) / "data" / "knowledge_base" / kb_id
    assert not storage.exists(), f"KB storage leaked: {storage}"
    step("deleted")

    await plugin.terminate()
    assert job_manager._closed is True, "plugin terminate did not close jobs"
    assert job_manager._tasks == {}, "plugin terminate left running jobs"
    assert job_manager._db is None, "plugin terminate left the jobs database open"
    assert source_manager._closed is True, "plugin terminate did not close sources"
    assert source_manager._cleanup_task is None, "cleanup task still alive"
    assert source_manager._public_session is None, "public HTTP session still open"
    assert source_manager._internal_session is None, "internal HTTP session still open"
    assert source_manager._public_resolver is None, "DNS resolver still open"
    step("cleanup_done")


def _emit():
    print("KB_E2E_REPORT=" + json.dumps(REPORT), flush=True)


try:
    main()
except BaseException as exc:  # noqa: BLE001
    REPORT["error"] = f"{type(exc).__name__}: {exc}"
    REPORT["traceback"] = traceback.format_exc()
finally:
    leftover = [
        thread.name
        for thread in threading.enumerate()
        if thread is not threading.main_thread() and not thread.daemon
    ]
    REPORT["leftover_threads"] = leftover
    if leftover:
        REPORT["ok"] = False
        REPORT["exit_note"] = (
            "forced exit after recording non-daemon threads still alive"
        )
    _emit()
    if leftover:
        # Only when the interpreter would otherwise hang: record first, then
        # hard-exit so pytest is not blocked. Plugin resources are asserted
        # closed above, so this never hides a plugin leak.
        os._exit(0)
'''


def _build_tmp_root(tmp_path: Path) -> Path:
    root = tmp_path / "astrbot_root"
    plugin_dir = root / "data" / "plugins" / "astrbot_plugin_kb_manager"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    for name in _COPY_FILES:
        source = PLUGIN_SRC / name
        if source.exists():
            shutil.copy2(source, plugin_dir / name)
    (root / "data" / "config").mkdir(parents=True, exist_ok=True)
    return root


def _run_child(tmp_path: Path) -> dict:
    root = _build_tmp_root(tmp_path)
    main_file = root / "data" / "plugins" / "astrbot_plugin_kb_manager" / "main.py"
    if not main_file.exists():
        pytest.fail(
            "fixture ready, but main.py is not present yet (owned by task C); "
            "nothing was faked. Re-run once main.py lands.",
        )
    child = tmp_path / "_child_runner.py"
    child.write_text(_CHILD_SOURCE, encoding="utf-8")
    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    completed = subprocess.run(
        [sys.executable, str(child), str(root), str(UPSTREAM_ROOT), str(PLUGIN_SRC)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        env=env,
        cwd=str(tmp_path),
    )
    output = (completed.stdout or "") + "\n" + (completed.stderr or "")
    report = None
    for line in output.splitlines():
        if line.startswith("KB_E2E_REPORT="):
            report = json.loads(line[len("KB_E2E_REPORT=") :])
    if report is None:
        pytest.fail(
            f"child produced no report (rc={completed.returncode}).\n{output[-4000:]}",
        )
    if not report.get("ok"):
        pytest.fail(
            f"native e2e failed: {report.get('error')}\n"
            f"{report.get('traceback')}\n{output[-2000:]}",
        )
    return report


def test_native_plugin_load_and_end_to_end(tmp_path):
    report = _run_child(tmp_path)
    assert report["ok"] is True
    assert report["leftover_threads"] == [], report["leftover_threads"]
    assert report["exit_note"] is None, report["exit_note"]
    names = [step["name"] for step in report["steps"]]
    for expected in (
        "context_built",
        "plugin_loaded",
        "managers_bound",
        "tools_registered",
        "schemas_checked",
        "kb_created",
        "text_added",
        "document_read",
        "chunks_mutated",
        "search_found",
        "document_replaced",
        "kb_updated",
        "attachments_imported",
        "url_imported_stub",
        "non_admin_denied",
        "reloaded",
        "request_id_reused",
        "attachment_idempotent",
        "document_deleted",
        "deleted",
        "cleanup_done",
    ):
        if expected not in names:
            raise AssertionError(f"missing e2e step {expected!r}: {names}")
