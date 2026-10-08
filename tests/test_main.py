"""Entry-point tests for ``main.KBManagerPlugin``.

The tests use the real AstrBot decorators, Star class and tool registry, with
lightweight service doubles whose state is recorded. ``ASTRBOT_ROOT`` is
pointed at a throwaway directory before the first ``astrbot`` import so the
workspace's real data directory is never touched.
"""

from __future__ import annotations

import functools
import inspect
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

# Import the plugin as a package, mirroring how AstrBot loads it: only the
# parent of ``data/plugins`` and the upstream source go on sys.path.
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
UPSTREAM_ROOT = WORKSPACE_ROOT / "upstream" / "AstrBot"
DATA_DIR = WORKSPACE_ROOT / "data"
for _path in (str(UPSTREAM_ROOT), str(DATA_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# AstrBot resolves runtime paths from ASTRBOT_ROOT at call time; point it at a
# throwaway directory before the first astrbot import.
os.environ.setdefault("ASTRBOT_ROOT", tempfile.mkdtemp(prefix="kb_main_root_"))

import astrbot.api  # noqa: E402,F401
from astrbot.core.agent.message import TextPart  # noqa: E402
from astrbot.core.agent.tool import FunctionTool, ToolSet  # noqa: E402
from astrbot.core.message.components import File, Image, Plain  # noqa: E402
from astrbot.core.provider.entities import ProviderRequest  # noqa: E402
from astrbot.core.provider.register import llm_tools  # noqa: E402
from plugins.astrbot_plugin_kb_manager import common  # noqa: E402
from plugins.astrbot_plugin_kb_manager import main as main_module  # noqa: E402
from plugins.astrbot_plugin_kb_manager.autonomy import (  # noqa: E402
    AUTONOMY_BLOCK_START,
    DEFAULT_AUTONOMY_PROMPT,
)
from plugins.astrbot_plugin_kb_manager.main import KBManagerPlugin  # noqa: E402

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def enabled_session(monkeypatch):
    """Default unit tests to a session where this plugin is enabled.

    ``SessionPluginManager`` is mocked so tests never depend on shared
    preferences; individual tests re-patch it to exercise the disabled or the
    failing-lookup path.
    """

    async def _enabled(umo: str, plugin_name: str) -> bool:
        assert plugin_name == "astrbot_plugin_kb_manager"
        return True

    monkeypatch.setattr(
        main_module.SessionPluginManager,
        "is_plugin_enabled_for_session",
        _enabled,
    )


TOOL_NAMES = (
    "kbm_list_kbs",
    "kbm_create_kb",
    "kbm_update_kb",
    "kbm_delete_kb",
    "kbm_list_documents",
    "kbm_read_document",
    "kbm_search",
    "kbm_add_text",
    "kbm_list_attachments",
    "kbm_import_attachment",
    "kbm_import_url",
    "kbm_replace_document",
    "kbm_delete_document",
    "kbm_add_chunk",
    "kbm_update_chunk",
    "kbm_delete_chunk",
    "kbm_job_status",
)

WRITE_TOOLS = (
    "kbm_create_kb",
    "kbm_update_kb",
    "kbm_delete_kb",
    "kbm_add_text",
    "kbm_import_attachment",
    "kbm_import_url",
    "kbm_replace_document",
    "kbm_delete_document",
    "kbm_add_chunk",
    "kbm_update_chunk",
    "kbm_delete_chunk",
)

EXPECTED_REQUIRED: dict[str, list[str]] = {
    "kbm_list_kbs": [],
    "kbm_create_kb": ["request_id", "name"],
    "kbm_update_kb": ["request_id", "kb_id", "changes"],
    "kbm_delete_kb": ["request_id", "kb_id"],
    "kbm_list_documents": ["kb_id"],
    "kbm_read_document": ["kb_id", "doc_id"],
    "kbm_search": ["query", "kb_ids"],
    "kbm_add_text": ["request_id", "kb_id", "filename", "content"],
    "kbm_list_attachments": [],
    "kbm_import_attachment": ["request_id", "kb_id", "attachment_id"],
    "kbm_import_url": ["request_id", "kb_id", "url"],
    "kbm_replace_document": [
        "request_id",
        "kb_id",
        "doc_id",
        "filename",
        "content",
    ],
    "kbm_delete_document": ["request_id", "kb_id", "doc_id"],
    "kbm_add_chunk": ["request_id", "kb_id", "doc_id", "content"],
    "kbm_update_chunk": ["request_id", "kb_id", "doc_id", "chunk_id", "content"],
    "kbm_delete_chunk": ["request_id", "kb_id", "doc_id", "chunk_id"],
    "kbm_job_status": ["job_id"],
}

EXPECTED_TYPES: dict[str, dict[str, str]] = {
    "kbm_list_kbs": {},
    "kbm_create_kb": {
        "request_id": "string",
        "name": "string",
        "description": "string",
        "embedding_provider_id": "string",
        "chunk_size": "number",
        "chunk_overlap": "number",
    },
    "kbm_update_kb": {
        "request_id": "string",
        "kb_id": "string",
        "changes": "object",
    },
    "kbm_delete_kb": {"request_id": "string", "kb_id": "string"},
    "kbm_list_documents": {
        "kb_id": "string",
        "offset": "number",
        "limit": "number",
        "search": "string",
    },
    "kbm_read_document": {
        "kb_id": "string",
        "doc_id": "string",
        "offset": "number",
        "limit": "number",
    },
    "kbm_search": {
        "query": "string",
        "kb_ids": "array",
        "top_k": "number",
    },
    "kbm_add_text": {
        "request_id": "string",
        "kb_id": "string",
        "filename": "string",
        "content": "string",
    },
    "kbm_list_attachments": {},
    "kbm_import_attachment": {
        "request_id": "string",
        "kb_id": "string",
        "attachment_id": "string",
    },
    "kbm_import_url": {
        "request_id": "string",
        "kb_id": "string",
        "url": "string",
    },
    "kbm_replace_document": {
        "request_id": "string",
        "kb_id": "string",
        "doc_id": "string",
        "filename": "string",
        "content": "string",
    },
    "kbm_delete_document": {
        "request_id": "string",
        "kb_id": "string",
        "doc_id": "string",
    },
    "kbm_add_chunk": {
        "request_id": "string",
        "kb_id": "string",
        "doc_id": "string",
        "content": "string",
    },
    "kbm_update_chunk": {
        "request_id": "string",
        "kb_id": "string",
        "doc_id": "string",
        "chunk_id": "string",
        "content": "string",
    },
    "kbm_delete_chunk": {
        "request_id": "string",
        "kb_id": "string",
        "doc_id": "string",
        "chunk_id": "string",
    },
    "kbm_job_status": {"job_id": "string"},
}

CALL_ARGS: dict[str, dict[str, Any]] = {
    "kbm_list_kbs": {},
    "kbm_create_kb": {"request_id": "req-1", "name": "kb"},
    "kbm_update_kb": {"request_id": "req-1", "kb_id": "kb-1", "changes": {}},
    "kbm_delete_kb": {"request_id": "req-1", "kb_id": "kb-1"},
    "kbm_list_documents": {"kb_id": "kb-1"},
    "kbm_read_document": {"kb_id": "kb-1", "doc_id": "doc-1"},
    "kbm_search": {"query": "q", "kb_ids": ["kb-1"]},
    "kbm_add_text": {
        "request_id": "req-1",
        "kb_id": "kb-1",
        "filename": "a.txt",
        "content": "hello",
    },
    "kbm_list_attachments": {},
    "kbm_import_attachment": {
        "request_id": "req-1",
        "kb_id": "kb-1",
        "attachment_id": "att-1",
    },
    "kbm_import_url": {
        "request_id": "req-1",
        "kb_id": "kb-1",
        "url": "https://example.com/page",
    },
    "kbm_replace_document": {
        "request_id": "req-1",
        "kb_id": "kb-1",
        "doc_id": "doc-1",
        "filename": "a.txt",
        "content": "hello",
    },
    "kbm_delete_document": {
        "request_id": "req-1",
        "kb_id": "kb-1",
        "doc_id": "doc-1",
    },
    "kbm_add_chunk": {
        "request_id": "req-1",
        "kb_id": "kb-1",
        "doc_id": "doc-1",
        "content": "chunk",
    },
    "kbm_update_chunk": {
        "request_id": "req-1",
        "kb_id": "kb-1",
        "doc_id": "doc-1",
        "chunk_id": "chunk-1",
        "content": "chunk",
    },
    "kbm_delete_chunk": {
        "request_id": "req-1",
        "kb_id": "kb-1",
        "doc_id": "doc-1",
        "chunk_id": "chunk-1",
    },
    "kbm_job_status": {"job_id": "job-1"},
}


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class FakeEvent:
    """Minimal AstrMessageEvent surface used by the plugin."""

    def __init__(
        self,
        *,
        admin: bool = True,
        umo: str = "test-platform:GroupMessage:7",
        sender: str = "10001",
        messages: list[Any] | None = None,
        plugins_name: list[str] | None = None,
    ) -> None:
        self.unified_msg_origin = umo
        self._sender = sender
        self._admin = admin
        self._messages = list(messages or [])
        self.plugins_name = plugins_name
        self.stopped = False

    def is_admin(self) -> bool:
        return self._admin

    def get_sender_id(self) -> str:
        return self._sender

    def get_messages(self) -> list[Any]:
        return self._messages

    def stop_event(self) -> None:
        self.stopped = True


class Exploding:
    """Fails the test if any attribute is accessed."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"shared resource must not be touched: {name}")


class FakeBackend:
    def __init__(self, fail_initialize: BaseException | None = None) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self._fail_initialize = fail_initialize

    async def initialize(self) -> dict[str, Any]:
        self.calls.append(("backend.initialize",))
        if self._fail_initialize is not None:
            raise self._fail_initialize
        return {"initialized": True}

    async def list_kbs(self) -> dict[str, Any]:
        self.calls.append(("backend.list_kbs",))
        return {"kbs": [], "embedding_providers": []}

    async def list_documents(self, kb_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("backend.list_documents", kb_id))
        return {"documents": [], "total": 0}

    async def read_document(
        self, kb_id: str, doc_id: str, **kwargs: Any
    ) -> dict[str, Any]:
        self.calls.append(("backend.read_document", kb_id, doc_id))
        return {"document": {}, "chunks": [], "total": 0}

    async def search(
        self, query: str, kb_ids: list[str], **kwargs: Any
    ) -> dict[str, Any]:
        self.calls.append(("backend.search", query, tuple(kb_ids)))
        return {"results": []}

    async def create_kb(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("backend.create_kb", args))
        return {"kb": {"kb_id": "kb-new"}}

    async def update_kb(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("backend.update_kb", args, kwargs))
        return {"kb": {"kb_id": "kb-1"}}

    async def delete_kb(self, kb_id: str) -> dict[str, Any]:
        self.calls.append(("backend.delete_kb", kb_id))
        return {"deleted": True, "kb_id": kb_id}

    async def add_document(
        self, kb_id: str, filename: str, content: bytes, progress: Any = None
    ) -> dict[str, Any]:
        self.calls.append(("backend.add_document", kb_id, filename, content))
        return {"document": {"doc_id": "doc-new"}, "chunk_count": 1}

    async def replace_document(
        self,
        kb_id: str,
        doc_id: str,
        filename: str,
        content: bytes,
        progress: Any = None,
    ) -> dict[str, Any]:
        self.calls.append(("backend.replace_document", kb_id, doc_id, filename))
        return {"old_doc_id": doc_id, "document": {"doc_id": "doc-new"}}

    async def delete_document(self, kb_id: str, doc_id: str) -> dict[str, Any]:
        self.calls.append(("backend.delete_document", kb_id, doc_id))
        return {"deleted": True}

    async def add_chunk(self, kb_id: str, doc_id: str, content: str) -> dict[str, Any]:
        self.calls.append(("backend.add_chunk", kb_id, doc_id, content))
        return {"chunk": {"chunk_id": "chunk-new"}}

    async def update_chunk(
        self, kb_id: str, doc_id: str, chunk_id: str, content: str
    ) -> dict[str, Any]:
        self.calls.append(("backend.update_chunk", kb_id, doc_id, chunk_id))
        return {"chunk": {"chunk_id": chunk_id}}

    async def delete_chunk(
        self, kb_id: str, doc_id: str, chunk_id: str
    ) -> dict[str, Any]:
        self.calls.append(("backend.delete_chunk", kb_id, doc_id, chunk_id))
        return {"deleted": True}


class FakeSources:
    def __init__(
        self,
        fail_initialize: BaseException | None = None,
        timeline: list[tuple[Any, ...]] | None = None,
    ) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.remembered: list[tuple[str, list[Any]]] = []
        self.attachment_infos: list[dict[str, Any]] = []
        self.load_attachment_count = 0
        self.load_url_count = 0
        self.timeline = timeline if timeline is not None else []
        self._fail_initialize = fail_initialize

    async def initialize(self) -> None:
        self.calls.append(("sources.initialize",))
        if self._fail_initialize is not None:
            raise self._fail_initialize

    async def close(self) -> None:
        self.calls.append(("sources.close",))

    async def remember_attachments(
        self, scope: str, components: list[Any]
    ) -> dict[str, Any]:
        self.remembered.append((scope, list(components)))
        return {"attachments": []}

    async def load_attachment(self, scope: str, attachment_id: str) -> Any:
        self.load_attachment_count += 1
        self.timeline.append(("load_attachment", attachment_id))
        return common.SourceDocument(
            "attachment.txt", b"attachment-body", common.SOURCE_ATTACHMENT
        )

    async def load_url(self, url: str) -> Any:
        self.load_url_count += 1
        self.timeline.append(("load_url", url))
        return common.SourceDocument("page.md", b"page-body", common.SOURCE_URL)

    def list_attachments(self, scope: str) -> list[dict[str, Any]]:
        self.calls.append(("sources.list_attachments", scope))
        return list(self.attachment_infos)


class FakeJobs:
    def __init__(
        self,
        *,
        run_work: bool = False,
        fail_initialize: BaseException | None = None,
        timeline: list[tuple[Any, ...]] | None = None,
    ) -> None:
        self.submissions: list[dict[str, Any]] = []
        self.gets: list[tuple[str, str]] = []
        self.progress_records: list[tuple[str, str, int, int]] = []
        self.initialize_calls = 0
        self.close_calls = 0
        self.run_work = run_work
        self.timeline = timeline if timeline is not None else []
        self._fail_initialize = fail_initialize

    async def initialize(self) -> None:
        self.initialize_calls += 1
        if self._fail_initialize is not None:
            raise self._fail_initialize

    async def close(self) -> None:
        self.close_calls += 1

    async def submit(
        self,
        scope: str,
        request_id: str,
        operation: str,
        payload: dict[str, Any],
        lock_key: str,
        work: Any,
        wait_seconds: float = 3,
    ) -> dict[str, Any]:
        self.submissions.append(
            {
                "scope": scope,
                "request_id": request_id,
                "operation": operation,
                "payload": payload,
                "lock_key": lock_key,
                "wait_seconds": wait_seconds,
            }
        )
        self.timeline.append(("jobs.submit", operation))
        if self.run_work:

            async def progress(stage: str, current: int, total: int) -> None:
                self.progress_records.append((operation, stage, current, total))
                self.timeline.append(("progress", operation, stage))

            data = await work(progress)
            return common.ok_result(data, job_id="job-x")
        return common.ok_result({"queued": True}, job_id="job-x")

    async def get(self, scope: str, job_id: str) -> dict[str, Any]:
        self.gets.append((scope, job_id))
        return common.make_result("succeeded", job_id=job_id, data={"job_id": job_id})


class FakeContext:
    """Context stub exposing the public tool-manager accessor."""

    def __init__(self, manager: Any = None) -> None:
        self._manager = manager if manager is not None else llm_tools

    def get_llm_tool_manager(self) -> Any:
        return self._manager


def make_plugin(
    *,
    backend: Any = None,
    sources: Any = None,
    jobs: Any = None,
    config: dict[str, Any] | None = None,
    context: Any = None,
) -> KBManagerPlugin:
    plugin = KBManagerPlugin(context if context is not None else FakeContext(), config)
    if backend is not None:
        plugin._backend = backend
    if sources is not None:
        plugin._sources = sources
    if jobs is not None:
        plugin._jobs = jobs
    return plugin


def decode(result: str) -> dict[str, Any]:
    return json.loads(result)


# ---------------------------------------------------------------------------
# Schema and registration
# ---------------------------------------------------------------------------


async def test_all_tool_schemas_are_registered_and_typed():
    plugin = make_plugin()
    plugin._apply_tool_schemas()

    for name in TOOL_NAMES:
        assert callable(getattr(plugin, name, None)), f"method missing: {name}"
        tool = llm_tools.get_func(name)
        assert tool is not None, f"tool not registered: {name}"
        parameters = tool.parameters
        assert parameters["type"] == "object"
        properties = parameters["properties"]
        assert set(properties) == set(EXPECTED_TYPES[name]), name
        assert "self" not in properties and "event" not in properties, name
        for arg, value in properties.items():
            assert value.get("type") == EXPECTED_TYPES[name][arg], (name, arg)
            assert value.get("description"), (name, arg)
            if value.get("type") == "array":
                assert value.get("items", {}).get("type") == "string", (name, arg)
        assert parameters.get("required") == EXPECTED_REQUIRED[name], name

    changes_schema = llm_tools.get_func("kbm_update_kb").parameters["properties"][
        "changes"
    ]
    assert set(changes_schema["properties"]) == {
        "description",
        "chunk_size",
        "chunk_overlap",
    }
    assert changes_schema["additionalProperties"] is False
    assert changes_schema["properties"]["chunk_size"]["type"] == "number"
    assert changes_schema["properties"]["chunk_overlap"]["type"] == "number"
    assert changes_schema["properties"]["description"]["type"] == "string"


async def test_written_tools_require_request_id():
    plugin = make_plugin()
    plugin._apply_tool_schemas()
    for name in WRITE_TOOLS:
        required = llm_tools.get_func(name).parameters["required"]
        assert "request_id" in required, name


async def test_scope_is_never_a_tool_parameter():
    for name in TOOL_NAMES:
        parameters = inspect.signature(getattr(KBManagerPlugin, name)).parameters
        assert "scope" not in parameters, name


# ---------------------------------------------------------------------------
# Availability for ordinary senders and session/tool gating
# ---------------------------------------------------------------------------


async def test_all_tools_are_available_to_ordinary_senders():
    backend = FakeBackend()
    sources = FakeSources()
    jobs = FakeJobs()
    plugin = make_plugin(backend=backend, sources=sources, jobs=jobs)
    event = FakeEvent(admin=False, umo="umo-user", sender="user-2002")

    for name in TOOL_NAMES:
        result = decode(await getattr(plugin, name)(event, **CALL_ARGS[name]))
        assert result["status"] == "succeeded", (name, result)
        assert result["error"] is None, (name, result)

    assert backend.calls
    expected_scope = common.encode_scope("umo-user", "user-2002")
    assert jobs.gets == [(expected_scope, "job-1")]
    assert ("sources.list_attachments", expected_scope) in sources.calls
    assert event.stopped is False


async def test_disabled_session_refuses_tools_without_service_calls(monkeypatch):
    async def _disabled(umo: str, plugin_name: str) -> bool:
        return False

    monkeypatch.setattr(
        main_module.SessionPluginManager,
        "is_plugin_enabled_for_session",
        _disabled,
    )
    backend = FakeBackend()
    sources = FakeSources()
    jobs = FakeJobs()
    plugin = make_plugin(backend=backend, sources=sources, jobs=jobs)
    event = FakeEvent(admin=False, plugins_name=["astrbot_plugin_kb_manager"])

    for name in TOOL_NAMES:
        result = decode(await getattr(plugin, name)(event, **CALL_ARGS[name]))
        assert result["status"] == "failed", name
        assert result["error"]["code"] == "plugin_disabled", name

    assert backend.calls == []
    assert sources.calls == []
    assert sources.remembered == []
    assert jobs.submissions == []
    assert jobs.gets == []


async def test_event_plugin_whitelist_excluding_this_plugin_refuses_tools():
    plugin = make_plugin()
    event = FakeEvent(plugins_name=["some_other_plugin"])

    result = decode(await plugin.kbm_list_kbs(event))
    assert result["status"] == "failed"
    assert result["error"]["code"] == "plugin_disabled"


async def test_session_lookup_failure_is_fail_closed(monkeypatch):
    async def _boom(umo: str, plugin_name: str) -> bool:
        raise RuntimeError("preferences unavailable")

    monkeypatch.setattr(
        main_module.SessionPluginManager,
        "is_plugin_enabled_for_session",
        _boom,
    )
    plugin = make_plugin()

    result = decode(await plugin.kbm_list_kbs(FakeEvent()))
    assert result["status"] == "failed"
    assert result["error"]["code"] == "plugin_disabled"


async def test_globally_disabled_tool_is_refused():
    plugin = make_plugin()
    disabled = llm_tools.get_func("kbm_delete_kb")
    disabled.active = False
    try:
        result = decode(
            await plugin.kbm_delete_kb(FakeEvent(), request_id="req-1", kb_id="kb-1")
        )
        assert result["status"] == "failed"
        assert result["error"]["code"] == "tool_disabled"
    finally:
        disabled.active = True


# ---------------------------------------------------------------------------
# Read tool wiring
# ---------------------------------------------------------------------------


async def test_read_tools_return_envelopes_and_use_trusted_scope():
    backend = FakeBackend()
    sources = FakeSources()
    sources.attachment_infos = [
        {"attachment_id": "att-1", "filename": "a.txt"},
    ]
    jobs = FakeJobs()
    plugin = make_plugin(backend=backend, sources=sources, jobs=jobs)
    event = FakeEvent(umo="umo-x", sender="sender-y")

    result = decode(await plugin.kbm_list_kbs(event))
    assert result["status"] == "succeeded"
    assert result["data"] == {"kbs": [], "embedding_providers": []}

    result = decode(await plugin.kbm_list_attachments(event))
    assert result["data"] == {
        "attachments": [{"attachment_id": "att-1", "filename": "a.txt"}]
    }

    result = decode(await plugin.kbm_job_status(event, "job-1"))
    assert result["status"] == "succeeded"
    assert result["job_id"] == "job-1"
    expected_scope = common.encode_scope("umo-x", "sender-y")
    assert jobs.gets == [(expected_scope, "job-1")]
    assert ("sources.list_attachments", expected_scope) in sources.calls


# ---------------------------------------------------------------------------
# Write tool wiring through the job manager
# ---------------------------------------------------------------------------


async def test_write_tools_submit_through_jobs_with_stable_payloads():
    backend = FakeBackend()
    sources = FakeSources()
    jobs = FakeJobs()
    plugin = make_plugin(backend=backend, sources=sources, jobs=jobs)
    event = FakeEvent(umo="umo-w", sender="sender-w")

    for name in WRITE_TOOLS:
        result = decode(await getattr(plugin, name)(event, **CALL_ARGS[name]))
        assert result["status"] == "succeeded", name

    assert len(jobs.submissions) == len(WRITE_TOOLS)
    by_operation = {item["operation"]: item for item in jobs.submissions}
    assert set(by_operation) == set(WRITE_TOOLS)

    expected_scope = common.encode_scope("umo-w", "sender-w")
    for name, item in by_operation.items():
        assert item["scope"] == expected_scope, name
        assert item["request_id"] == "req-1", name
    assert by_operation["kbm_create_kb"]["lock_key"] == common.LOCK_GLOBAL
    for name in WRITE_TOOLS:
        if name == "kbm_create_kb":
            continue
        assert by_operation[name]["lock_key"] == common.kb_lock_key("kb-1"), name

    assert by_operation["kbm_add_text"]["payload"] == {
        "kb_id": "kb-1",
        "filename": "a.txt",
        "content": "hello",
    }
    assert by_operation["kbm_import_attachment"]["payload"] == {
        "kb_id": "kb-1",
        "attachment_id": "att-1",
    }
    assert by_operation["kbm_import_url"]["payload"] == {
        "kb_id": "kb-1",
        "url": "https://example.com/page",
    }
    assert by_operation["kbm_create_kb"]["payload"] == {
        "name": "kb",
        "description": "",
        "embedding_provider_id": "",
        "chunk_size": 512,
        "chunk_overlap": 50,
    }
    # Submission alone must not download attachments or URLs.
    assert sources.load_attachment_count == 0
    assert sources.load_url_count == 0
    assert backend.calls == []


async def test_update_kb_rejects_unknown_change_fields():
    jobs = FakeJobs()
    plugin = make_plugin(jobs=jobs)
    event = FakeEvent()

    result = decode(
        await plugin.kbm_update_kb(
            event,
            request_id="req-1",
            kb_id="kb-1",
            changes={"description": "d", "unknown": 1},
        )
    )
    assert result["status"] == "failed"
    assert result["error"]["code"] == "invalid_argument"
    assert jobs.submissions == []

    result = decode(
        await plugin.kbm_update_kb(
            event,
            request_id="req-1",
            kb_id="kb-1",
            changes={"description": "d"},
        )
    )
    assert result["status"] == "succeeded"
    assert jobs.submissions[0]["payload"] == {
        "kb_id": "kb-1",
        "changes": {"description": "d"},
    }


async def test_add_text_rejects_non_text_extensions_and_large_content():
    jobs = FakeJobs()
    plugin = make_plugin(jobs=jobs)
    plugin._max_file_bytes = 4
    event = FakeEvent()

    result = decode(
        await plugin.kbm_add_text(
            event,
            request_id="req-1",
            kb_id="kb-1",
            filename="manual.pdf",
            content="text",
        )
    )
    assert result["status"] == "failed"
    assert result["error"]["code"] == "invalid_argument"
    assert jobs.submissions == []

    result = decode(
        await plugin.kbm_add_text(
            event,
            request_id="req-1",
            kb_id="kb-1",
            filename="manual",
            content="toolong",
        )
    )
    assert result["status"] == "failed"
    assert result["error"]["code"] == "payload_too_large"
    assert jobs.submissions == []


async def test_real_jobs_reuses_request_id_without_rerunning_work():
    plugin = make_plugin(backend=FakeBackend(), sources=FakeSources())
    # keep the real JobManager created by __init__
    await plugin.initialize()
    event = FakeEvent(umo="umo-idem", sender="sender-idem")

    first = decode(
        await plugin.kbm_add_text(
            event,
            request_id="idem-add-1",
            kb_id="kb-1",
            filename="a.txt",
            content="hello",
        )
    )
    second = decode(
        await plugin.kbm_add_text(
            event,
            request_id="idem-add-1",
            kb_id="kb-1",
            filename="a.txt",
            content="hello",
        )
    )

    assert first["status"] == "succeeded"
    assert second["status"] == "succeeded"
    assert first["job_id"] == second["job_id"]
    assert first["data"] == second["data"]
    backend_calls = [
        call for call in plugin._backend.calls if call[0] == "backend.add_document"
    ]
    assert len(backend_calls) == 1
    await plugin.terminate()


async def test_reload_reuses_import_result_without_touching_attachments():
    sources = FakeSources()
    plugin = make_plugin(backend=FakeBackend(), sources=sources)
    await plugin.initialize()
    event = FakeEvent(umo="umo-reload", sender="sender-reload")

    first = decode(
        await plugin.kbm_import_attachment(
            event,
            request_id="reload-import-1",
            kb_id="kb-1",
            attachment_id="att-1",
        )
    )
    assert first["status"] == "succeeded"
    assert sources.load_attachment_count == 1
    await plugin.terminate()

    # Simulate a reload: same data dir, fresh plugin, attachment cache gone.
    class ExplodingSources(FakeSources):
        async def load_attachment(self, scope: str, attachment_id: str) -> Any:
            raise AssertionError("attachment must not be downloaded again")

    reloaded = make_plugin(backend=FakeBackend(), sources=ExplodingSources())
    await reloaded.initialize()
    second = decode(
        await reloaded.kbm_import_attachment(
            event,
            request_id="reload-import-1",
            kb_id="kb-1",
            attachment_id="att-1",
        )
    )
    assert second["status"] == "succeeded"
    assert second["job_id"] == first["job_id"]
    assert second["data"] == first["data"]
    await reloaded.terminate()


async def test_attachment_and_url_download_happen_inside_work():
    sources = FakeSources()
    plugin = make_plugin(backend=FakeBackend(), sources=sources)
    await plugin.initialize()
    event = FakeEvent(umo="umo-download", sender="sender-download")

    # Two submissions of the same request id: the second is pure reuse.
    for _ in range(2):
        result = decode(
            await plugin.kbm_import_attachment(
                event,
                request_id="download-att-1",
                kb_id="kb-1",
                attachment_id="att-1",
            )
        )
        assert result["status"] == "succeeded"
    for _ in range(2):
        result = decode(
            await plugin.kbm_import_url(
                event,
                request_id="download-url-1",
                kb_id="kb-1",
                url="https://example.com/page",
            )
        )
        assert result["status"] == "succeeded"
    assert sources.load_attachment_count == 1
    assert sources.load_url_count == 1
    await plugin.terminate()


# ---------------------------------------------------------------------------
# Message listener and request context
# ---------------------------------------------------------------------------


def tool_names(request: ProviderRequest) -> set[str]:
    if request.func_tool is None:
        return set()
    return {tool.name for tool in request.func_tool.tools}


def _foreign_handler(*args: Any, **kwargs: Any) -> str:
    return "external"


def _foreign_tool(name: str = "foreign_tool") -> FunctionTool:
    return FunctionTool(
        name=name,
        description="external tool",
        parameters={
            "type": "object",
            "properties": {"x": {"type": "string", "description": "x"}},
        },
        handler=functools.partial(_foreign_handler),
        handler_module_path="external.plugin.main",
    )


async def test_message_listener_remembers_real_files_for_enabled_senders():
    sources = FakeSources()
    plugin = make_plugin(sources=sources)
    file_component = File(name="report.txt", file="C:/tmp/report.txt")
    event = FakeEvent(
        admin=False,
        messages=[Plain("hello"), file_component, Image(file="C:/tmp/x.png")],
    )

    await plugin.kbm_on_message(event)
    assert len(sources.remembered) == 1
    scope, components = sources.remembered[0]
    assert scope == common.encode_scope(event.unified_msg_origin, event.get_sender_id())
    assert components == [file_component]
    assert event.stopped is False


async def test_llm_request_hint_is_temporary_for_every_enabled_sender():
    sources = FakeSources()
    sources.attachment_infos = [
        {"attachment_id": "att-1", "filename": "a.txt"},
        {"attachment_id": "att-2", "filename": "b.md"},
    ]
    plugin = make_plugin(sources=sources)

    for admin in (True, False):
        event = FakeEvent(
            admin=admin, umo=f"umo-hint-{admin}", sender=f"sender-{admin}"
        )
        request = ProviderRequest(prompt="hello")
        await plugin.kbm_on_llm_request(event, request)
        assert len(request.extra_user_content_parts) == 1
        part = request.extra_user_content_parts[0]
        assert isinstance(part, TextPart)
        assert part._no_save is True
        assert "att-1" in part.text and "a.txt" in part.text
        assert "att-2" in part.text and "b.md" in part.text
        assert AUTONOMY_BLOCK_START in request.system_prompt


async def test_plain_request_gets_autonomy_prompt_and_tools_without_attachments():
    plugin = make_plugin(sources=FakeSources())
    event = FakeEvent(admin=False)  # ordinary sender, no maintenance command
    request = ProviderRequest(prompt="大家早上好")

    await plugin.kbm_on_llm_request(event, request)

    assert AUTONOMY_BLOCK_START in request.system_prompt
    assert DEFAULT_AUTONOMY_PROMPT in request.system_prompt
    assert request.extra_user_content_parts == []
    assert tool_names(request) == set(TOOL_NAMES)

    empty = ProviderRequest(prompt="hi")
    empty.func_tool = ToolSet()
    await plugin.kbm_on_llm_request(event, empty)
    assert tool_names(empty) == set(TOOL_NAMES)


async def test_llm_request_prompt_is_preserved_and_idempotent():
    plugin = make_plugin()
    event = FakeEvent()
    request = ProviderRequest(prompt="hi", system_prompt="【人格】保持简洁。")

    await plugin.kbm_on_llm_request(event, request)
    first = request.system_prompt
    assert first.startswith("【人格】保持简洁。")
    assert first.count(AUTONOMY_BLOCK_START) == 1

    await plugin.kbm_on_llm_request(event, request)
    assert request.system_prompt == first
    assert tool_names(request) == set(TOOL_NAMES)


async def test_custom_autonomy_prompt_replaces_the_module_default():
    plugin = make_plugin(config={"autonomy_system_prompt": "自定义维护规则"})
    request = ProviderRequest(prompt="hi")

    await plugin.kbm_on_llm_request(FakeEvent(), request)

    assert "自定义维护规则" in request.system_prompt
    assert DEFAULT_AUTONOMY_PROMPT not in request.system_prompt


async def test_blank_autonomy_prompt_falls_back_to_default():
    plugin = make_plugin(config={"autonomy_system_prompt": "   \n"})
    request = ProviderRequest(prompt="hi")

    await plugin.kbm_on_llm_request(FakeEvent(), request)

    assert DEFAULT_AUTONOMY_PROMPT in request.system_prompt


async def test_request_tool_set_is_merged_without_mutating_shared_state():
    plugin = make_plugin()
    bare = llm_tools.get_func("kbm_list_kbs")
    assert bare is not None and bare.handler is not None
    foreign = _foreign_tool()
    shared = ToolSet()
    shared.add_tool(bare)
    shared.add_tool(foreign)
    request = ProviderRequest(prompt="hi", func_tool=shared)

    await plugin.kbm_on_llm_request(FakeEvent(), request)

    assert request.func_tool is not shared
    assert shared.tools == [bare, foreign]
    merged = {tool.name: tool for tool in request.func_tool.tools}
    assert set(merged) == set(TOOL_NAMES) | {"foreign_tool"}
    assert merged["foreign_tool"] is foreign
    wrapped = merged["kbm_list_kbs"]
    assert wrapped is not bare
    assert wrapped.handler is None
    assert type(wrapped).__name__ == "_PermissionGuardedTool"
    assert wrapped._wrapped is bare


async def test_existing_external_same_name_tool_is_not_overridden():
    plugin = make_plugin()
    external = _foreign_tool("kbm_search")
    request = ProviderRequest(prompt="hi")
    request.func_tool = ToolSet()
    request.func_tool.add_tool(external)

    await plugin.kbm_on_llm_request(FakeEvent(), request)

    merged = {tool.name: tool for tool in request.func_tool.tools}
    assert merged["kbm_search"] is external
    assert set(merged) == set(TOOL_NAMES)


async def test_globally_disabled_tool_is_excluded_from_the_request():
    plugin = make_plugin()
    disabled = llm_tools.get_func("kbm_delete_kb")
    disabled.active = False
    try:
        shared = ToolSet()
        shared.add_tool(disabled)
        request = ProviderRequest(prompt="hi", func_tool=shared)

        await plugin.kbm_on_llm_request(FakeEvent(), request)

        assert tool_names(request) == set(TOOL_NAMES) - {"kbm_delete_kb"}
        assert shared.tools == [disabled]
    finally:
        disabled.active = True


async def test_disabled_session_skips_injection_and_removes_own_tools(monkeypatch):
    async def _disabled(umo: str, plugin_name: str) -> bool:
        return False

    monkeypatch.setattr(
        main_module.SessionPluginManager,
        "is_plugin_enabled_for_session",
        _disabled,
    )
    sources = FakeSources()
    plugin = make_plugin(sources=sources)
    bare = llm_tools.get_func("kbm_list_kbs")
    foreign = _foreign_tool()
    shared = ToolSet()
    shared.add_tool(bare)
    shared.add_tool(foreign)
    request = ProviderRequest(prompt="hi", system_prompt="【人格】", func_tool=shared)
    event = FakeEvent(admin=False)

    await plugin.kbm_on_llm_request(event, request)

    assert request.system_prompt == "【人格】"
    assert shared.tools == [bare, foreign]
    assert [tool.name for tool in request.func_tool.tools] == ["foreign_tool"]

    empty_request = ProviderRequest(prompt="hi")
    await plugin.kbm_on_llm_request(event, empty_request)
    assert empty_request.system_prompt == ""
    assert empty_request.func_tool is None

    # The attachment listener is gated by the same session state.
    file_component = File(name="a.txt", file="C:/tmp/a.txt")
    await plugin.kbm_on_message(FakeEvent(messages=[file_component]))
    assert sources.remembered == []


async def test_plugin_whitelist_exclusion_disables_injection_and_tools():
    plugin = make_plugin()
    event = FakeEvent(plugins_name=["other_plugin"])
    request = ProviderRequest(prompt="hi")

    await plugin.kbm_on_llm_request(event, request)

    assert request.system_prompt == ""
    assert request.func_tool is None


async def test_session_lookup_failure_blocks_injection_and_tools(monkeypatch):
    async def _boom(umo: str, plugin_name: str) -> bool:
        raise RuntimeError("preferences unavailable")

    monkeypatch.setattr(
        main_module.SessionPluginManager,
        "is_plugin_enabled_for_session",
        _boom,
    )
    plugin = make_plugin()
    request = ProviderRequest(prompt="hi")

    await plugin.kbm_on_llm_request(FakeEvent(), request)

    assert request.system_prompt == ""
    assert request.func_tool is None


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def test_initialize_starts_services_in_order():
    backend = FakeBackend()
    sources = FakeSources()
    jobs = FakeJobs()
    plugin = make_plugin(backend=backend, sources=sources, jobs=jobs)

    await plugin.initialize()
    assert backend.calls == [("backend.initialize",)]
    assert sources.calls == [("sources.initialize",)]
    assert jobs.initialize_calls == 1
    assert sources.calls.count(("sources.close",)) == 0


async def test_initialize_failure_closes_opened_sources():
    backend = FakeBackend()
    sources = FakeSources()
    jobs = FakeJobs(fail_initialize=RuntimeError("jobs down"))
    plugin = make_plugin(backend=backend, sources=sources, jobs=jobs)

    with pytest.raises(RuntimeError):
        await plugin.initialize()
    assert sources.calls == [("sources.initialize",), ("sources.close",)]


async def test_initialize_failure_before_sources_opens_no_cleanup():
    backend = FakeBackend()
    sources = FakeSources(fail_initialize=RuntimeError("sources down"))
    jobs = FakeJobs()
    plugin = make_plugin(backend=backend, sources=sources, jobs=jobs)

    with pytest.raises(RuntimeError):
        await plugin.initialize()
    assert sources.calls == [("sources.initialize",)]
    assert jobs.initialize_calls == 0


async def test_backend_initialize_failure_stops_the_chain():
    backend = FakeBackend(fail_initialize=RuntimeError("backend down"))
    sources = FakeSources()
    jobs = FakeJobs()
    plugin = make_plugin(backend=backend, sources=sources, jobs=jobs)

    with pytest.raises(RuntimeError):
        await plugin.initialize()
    assert sources.calls == []
    assert jobs.initialize_calls == 0


async def test_terminate_closes_jobs_then_sources_without_shared_resources():
    order: list[str] = []

    class OrderedJobs(FakeJobs):
        async def close(self) -> None:
            order.append("jobs.close")

    class OrderedSources(FakeSources):
        async def close(self) -> None:
            order.append("sources.close")

    context = SimpleNamespace(kb_manager=Exploding())
    plugin = make_plugin(
        backend=FakeBackend(),
        sources=OrderedSources(),
        jobs=OrderedJobs(),
        context=context,
    )
    await plugin.terminate()
    assert order == ["jobs.close", "sources.close"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


async def test_config_units_are_converted_and_invalid_values_fall_back():
    config = {
        "default_embedding_provider_id": "emb-1",
        "max_file_mb": 2,
        "max_url_mb": 3,
        "attachment_ttl_minutes": 4,
        "http_timeout_seconds": 5,
        "max_concurrent_jobs": 6,
    }
    plugin = KBManagerPlugin(FakeContext(), config)
    assert plugin._backend._default_embedding_provider_id == "emb-1"
    assert plugin._sources.max_file_bytes == 2 * 1024 * 1024
    assert plugin._sources.max_url_bytes == 3 * 1024 * 1024
    assert plugin._sources.attachment_ttl == 4 * 60
    assert plugin._sources.http_timeout == 5
    assert plugin._jobs._max_concurrent == 6

    invalid = {
        "max_file_mb": -1,
        "max_url_mb": "five",
        "attachment_ttl_minutes": True,
        "http_timeout_seconds": 0,
        "max_concurrent_jobs": None,
    }
    fallback = KBManagerPlugin(FakeContext(), invalid)
    assert fallback._sources.max_file_bytes == 20 * 1024 * 1024
    assert fallback._sources.max_url_bytes == 5 * 1024 * 1024
    assert fallback._sources.attachment_ttl == 30 * 60
    assert fallback._sources.http_timeout == 30
    assert fallback._jobs._max_concurrent == 3


# ---------------------------------------------------------------------------
# chunk_overlap = 0 and schema ownership
# ---------------------------------------------------------------------------


async def test_update_kb_accepts_zero_overlap_and_keeps_other_fields():
    backend = FakeBackend()
    jobs = FakeJobs(run_work=True)
    plugin = make_plugin(backend=backend, jobs=jobs)
    event = FakeEvent()

    result = decode(
        await plugin.kbm_update_kb(
            event,
            request_id="req-zero",
            kb_id="kb-1",
            changes={"chunk_overlap": 0},
        )
    )
    assert result["status"] == "succeeded"
    assert jobs.submissions[0]["payload"] == {
        "kb_id": "kb-1",
        "changes": {"chunk_overlap": 0},
    }
    update_calls = [call for call in backend.calls if call[0] == "backend.update_kb"]
    assert update_calls[0][1] == ("kb-1",)
    assert update_calls[0][2] == {
        "description": None,
        "chunk_size": None,
        "chunk_overlap": 0,
    }

    # Providing one field must leave the others untouched (None passthrough).
    backend.calls.clear()
    jobs.submissions.clear()
    result = decode(
        await plugin.kbm_update_kb(
            event,
            request_id="req-zero-2",
            kb_id="kb-1",
            changes={"description": "d", "chunk_overlap": 0},
        )
    )
    assert result["status"] == "succeeded"
    update_calls = [call for call in backend.calls if call[0] == "backend.update_kb"]
    assert update_calls[0][2] == {
        "description": "d",
        "chunk_size": None,
        "chunk_overlap": 0,
    }


async def test_update_kb_rejects_zero_chunk_size_and_negative_overlap():
    jobs = FakeJobs()
    plugin = make_plugin(jobs=jobs)
    event = FakeEvent()

    for changes in ({"chunk_size": 0}, {"chunk_overlap": -1}):
        result = decode(
            await plugin.kbm_update_kb(
                event,
                request_id="req-bad",
                kb_id="kb-1",
                changes=changes,
            )
        )
        assert result["status"] == "failed"
        assert result["error"]["code"] == "invalid_argument"
    assert jobs.submissions == []


def _external_handler(*args: Any, **kwargs: Any) -> str:
    return "external"


async def test_external_tool_with_same_name_is_not_modified():
    external = FunctionTool(
        name="kbm_list_kbs",
        description="external tool that happens to share a name",
        parameters={
            "type": "object",
            "properties": {"x": {"type": "string", "description": "x"}},
        },
        handler=functools.partial(_external_handler),
        handler_module_path="external.plugin.main",
    )
    snapshot = json.loads(json.dumps(external.parameters))
    llm_tools.func_list.append(external)
    try:
        make_plugin()  # construction applies the schema enrichment
        own_tools = [
            tool
            for tool in llm_tools.func_list
            if tool is not external and tool.name == "kbm_list_kbs"
        ]
        assert own_tools, "plugin tool missing from the registry"
        own = own_tools[-1]
        assert own.parameters.get("required") == []
        assert external.parameters == snapshot
        assert "required" not in external.parameters
    finally:
        llm_tools.func_list.remove(external)


# ---------------------------------------------------------------------------
# Attachment hint ordering and download progress
# ---------------------------------------------------------------------------


async def test_llm_request_hint_keeps_the_latest_ten_attachments():
    sources = FakeSources()
    sources.attachment_infos = [
        {
            "attachment_id": f"att-{index}",
            "filename": f"file-{index}.txt",
            "expires_at": 1000 + index,
        }
        for index in range(1, 12)
    ]
    plugin = make_plugin(sources=sources)
    request = ProviderRequest(prompt="hello")

    await plugin.kbm_on_llm_request(FakeEvent(), request)
    part = request.extra_user_content_parts[0]
    lines = [line for line in part.text.splitlines() if line.startswith("- ")]
    assert len(lines) == 10
    assert "- att-11: file-11.txt" in part.text
    assert "- att-2: file-2.txt" in part.text
    assert "- att-1: file-1.txt" not in part.text
    assert "kbm_list_attachments" in part.text


async def test_import_progress_stages_run_before_downloads():
    timeline: list[tuple[Any, ...]] = []
    sources = FakeSources(timeline=timeline)
    jobs = FakeJobs(run_work=True, timeline=timeline)
    plugin = make_plugin(backend=FakeBackend(), sources=sources, jobs=jobs)
    event = FakeEvent()

    await plugin.kbm_import_attachment(
        event,
        request_id="prog-att",
        kb_id="kb-1",
        attachment_id="att-1",
    )
    await plugin.kbm_import_url(
        event,
        request_id="prog-url",
        kb_id="kb-1",
        url="https://example.com/page",
    )

    assert ("kbm_import_attachment", "loading", 0, 0) in jobs.progress_records
    assert ("kbm_import_url", "fetching", 0, 0) in jobs.progress_records
    assert timeline.index(
        ("progress", "kbm_import_attachment", "loading")
    ) < timeline.index(("load_attachment", "att-1"))
    assert timeline.index(("progress", "kbm_import_url", "fetching")) < timeline.index(
        ("load_url", "https://example.com/page")
    )
