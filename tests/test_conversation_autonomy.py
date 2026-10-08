"""Real-framework acceptance test for autonomous KB maintenance in normal chat.

The plugin target (locked by the main agent) is:

* ``on_llm_request`` appends a configurable SYSTEM autonomy block for **every**
  ordinary LLM conversation, with no administrator gate and no user command;
* the same hook adds this plugin's *enabled* tools to ``request.func_tool``
  through ``FunctionToolManager.get_full_tool_set()`` (permission-wrapped),
  preserving the tools other producers already put there;
* a session-level plugin disable suppresses both the SYSTEM block and the
  tools; a globally inactive tool is not injected; a per-tool ``admin``
  permission still rejects an ordinary caller at call time;
* the SYSTEM rules are configurable: a plugin-config policy replaces the
  default block end to end.

Everything runs in an isolated child process with a throwaway
``ASTRBOT_ROOT`` under ``tmp_path``: the real ``PluginManager``, the real
registered ``OnLLMRequestEvent`` handler, the real ``FunctionToolExecutor``,
the real ``ToolLoopAgentRunner`` and the real SQLite/FAISS storage are used.
The chat model is a **scripted stand-in** that only represents the decision
"recognise a valuable fact -> search -> add"; it does not measure, and this
test does not claim, real model judgement quality.

Run with the plugin development interpreter::

    <venv>/Scripts/python.exe -m pytest tests/test_conversation_autonomy.py -q
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

_COPY_FILES = (
    "main.py",
    "backend.py",
    "sources.py",
    "jobs.py",
    "common.py",
    "autonomy.py",
    "metadata.yaml",
    "_conf_schema.json",
    "requirements.txt",
)

_CHILD_SOURCE = r'''
import asyncio
import hashlib
import json
import os
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
import mcp.types  # noqa: E402
from astrbot.core import astrbot_config, db_helper, sp  # noqa: E402
from astrbot.core.agent.hooks import BaseAgentRunHooks  # noqa: E402
from astrbot.core.agent.run_context import ContextWrapper  # noqa: E402
from astrbot.core.agent.runners.tool_loop_agent_runner import (  # noqa: E402
    ToolLoopAgentRunner,
)
from astrbot.core.agent.tool import FunctionTool, ToolSet  # noqa: E402
from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor  # noqa: E402
from astrbot.core.knowledge_base.kb_mgr import KnowledgeBaseManager  # noqa: E402
from astrbot.core.platform.message_type import MessageType  # noqa: E402
from astrbot.core.provider.entities import (  # noqa: E402
    LLMResponse,
    ProviderRequest,
    TokenUsage,
)
from astrbot.core.provider.provider import Provider  # noqa: E402
from astrbot.core.provider.register import llm_tools  # noqa: E402
from astrbot.core.star.context import Context  # noqa: E402
from astrbot.core.star.session_plugin_manager import SessionPluginManager  # noqa: E402
from astrbot.core.star.star import star_map  # noqa: E402
from astrbot.core.star.star_handler import (  # noqa: E402
    EventType,
    star_handlers_registry,
)
from astrbot.core.star.star_manager import PluginManager  # noqa: E402

from data.plugins.astrbot_plugin_kb_manager.autonomy import (  # noqa: E402
    AUTONOMY_BLOCK_END,
    AUTONOMY_BLOCK_START,
    DEFAULT_AUTONOMY_PROMPT,
)

MODULE = "data.plugins.astrbot_plugin_kb_manager.main"
PLUGIN_NAME = "astrbot_plugin_kb_manager"
NON_ADMIN_UMO = "e2e-platform:GroupMessage:group-e2e!user-2002"
NON_ADMIN_SENDER = "user-2002"
PERSONA = "你是群里的专业助手，回复保持简洁。"
KNOWLEDGE_TEXT = (
    "补充一条排障经验：服务端突发 502 时，先看反向代理的 upstream 超时，"
    "再把 keepalive 从 60 秒调到 15 秒，最后滚动重启后端；"
    "这套顺序在我们环境里稳定生效。"
)
CHITCHAT_TEXT = "哈哈，今天天气不错，随便聊聊。"
FACT_MARKER = "AUTONOMYMARKERFACT"
TERMINAL = {"succeeded", "failed", "partial", "interrupted"}
NON_TERMINAL = {"queued", "running"}
ENVELOPE_KEYS = {"status", "job_id", "data", "error"}

REPORT = {
    "ok": False,
    "steps": [],
    "error": None,
    "traceback": None,
    "leftover_threads": [],
    "exit_note": None,
}
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
    """Minimal non-admin group message event with a trusted UMO + sender."""

    def __init__(self, umo, sender_id, admin=False, text=""):
        self.unified_msg_origin = umo
        self._sender_id = sender_id
        self._admin = admin
        self._extras = {}
        self._messages = []
        self._stopped = False
        self.message_str = text
        self.message_obj = SimpleNamespace(
            message=[],
            message_str=text,
            type=MessageType.GROUP_MESSAGE,
        )

    def get_sender_id(self):
        return self._sender_id

    def is_admin(self):
        return self._admin

    def get_messages(self):
        return list(self._messages)

    def set_messages(self, messages):
        self._messages = list(messages)
        self.message_obj = SimpleNamespace(
            message=self._messages,
            message_str=self.message_str,
            type=MessageType.GROUP_MESSAGE,
        )

    def get_group_id(self):
        return "group-e2e"

    def get_platform_name(self):
        return "e2e-platform"

    def get_platform_id(self):
        return "e2e-platform"

    def get_self_id(self):
        return "bot-1"

    def get_message_type(self):
        return MessageType.GROUP_MESSAGE

    def get_extra(self, key=None, default=None):
        return self._extras.get(key, default)

    def set_extra(self, key, value):
        self._extras[key] = value

    def stop_event(self):
        self._stopped = True

    def is_stopped(self):
        return self._stopped

    def get_result(self):
        return None


class ScriptedConversationProvider(Provider):
    """Scripted stand-in for the chat model's maintenance decision.

    The script is a fixed list of ``(tool_name, args)`` steps followed by a
    final assistant reply. It represents "the model recognised a valuable
    fact, searched, then added it" - nothing more.
    """

    def __init__(self, steps):
        super().__init__({"id": "scripted-chat", "type": "scripted"}, {})
        self._steps = list(steps)
        self.calls = []
        self.issued_tool_calls = []

    def get_current_key(self):
        return "scripted"

    def set_key(self, key):
        pass

    async def get_models(self):
        return ["scripted-model"]

    async def text_chat(self, **kwargs):
        func_tool = kwargs.get("func_tool")
        names = func_tool.names() if func_tool else []
        self.calls.append(names)
        if self._steps:
            name, args = self._steps.pop(0)
            self.issued_tool_calls.append(name)
            return LLMResponse(
                role="assistant",
                completion_text="",
                tools_call_name=[name],
                tools_call_args=[dict(args)],
                tools_call_ids=["call-%d" % len(self.calls)],
                usage=TokenUsage(input_other=5, output=2),
            )
        return LLMResponse(
            role="assistant",
            completion_text="好的，这条经验我确认过了。",
            usage=TokenUsage(input_other=5, output=2),
        )


def parse_env(result):
    if not isinstance(result, str):
        raise AssertionError("tool did not return a str: %r" % (result,))
    try:
        env = json.loads(result)
    except ValueError as exc:
        raise AssertionError("tool result is not JSON: %r" % result[:200]) from exc
    assert isinstance(env, dict), env
    assert set(env) == ENVELOPE_KEYS, env
    return env


def envelope_ok(env):
    return env.get("status") == "succeeded"


def event_proxy(event, context):
    """The wrapper only touches ``context.context.event``."""

    return SimpleNamespace(context=SimpleNamespace(event=event, context=context))


async def call_raw(tool_name, event, **bag):
    """Call the registry tool handler directly (setup / verification only)."""

    tool = llm_tools.get_func(tool_name)
    assert tool is not None, "missing registry tool %s" % tool_name
    result = tool.handler(event, **bag)
    if hasattr(result, "__aiter__"):
        last = None
        async for item in result:
            last = item
        result = last
    else:
        result = await result
    env = parse_env(result)
    return await wait_terminal(tool_name, event, env)


async def wait_terminal(tool_name, event, env):
    if (
        tool_name != "kbm_job_status"
        and env["status"] in NON_TERMINAL
        and env.get("job_id")
    ):
        for _ in range(400):
            await asyncio.sleep(0.05)
            env = await call_raw(
                "kbm_job_status", event, job_id=env["job_id"]
            )
            if env["status"] in TERMINAL:
                break
    return env


async def exec_wrapped_text(tool, event, context, **bag):
    """Execute an injected (permission-wrapped) tool through the framework.

    Returns the raw joined text so a permission refusal (which the framework
    returns as a plain string, not an envelope) can be asserted verbatim.
    """

    texts = []
    run_context = ContextWrapper(
        context=SimpleNamespace(event=event, context=context)
    )
    async for res in FunctionToolExecutor.execute(
        tool=tool, run_context=run_context, **bag
    ):
        if res is None:
            continue
        for item in getattr(res, "content", []) or []:
            text = getattr(item, "text", None)
            if isinstance(text, str):
                texts.append(text)
    assert texts, "wrapped tool %s produced no text result" % tool.name
    return "\n".join(texts)


async def exec_wrapped(tool, event, context, **bag):
    """Execute the wrapped tool and return its JSON result envelope."""

    env = parse_env(await exec_wrapped_text(tool, event, context, **bag))
    return await wait_terminal(tool.name, event, env)


async def _safe_cleanup():
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


def find_llm_request_handler():
    for handler in star_handlers_registry.get_handlers_by_module_name(MODULE):
        if (
            handler.event_type == EventType.OnLLMRequestEvent
            and handler.handler_name == "kbm_on_llm_request"
        ):
            return handler
    raise AssertionError("kbm_on_llm_request is not registered")


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
        raise AssertionError("plugin load failed: %r" % (load_error,))
    metadata = star_map[MODULE]
    plugin = metadata.star_cls
    _OPEN["plugin"] = plugin
    if metadata.version != "0.2.0":
        raise AssertionError(
            "plugin version %r != '0.2.0'" % metadata.version
        )
    step("plugin_loaded", version=metadata.version)

    hook = find_llm_request_handler()
    non_admin = FakeEvent(NON_ADMIN_UMO, NON_ADMIN_SENDER, admin=False)

    async def run_hook(event, request):
        await hook.handler(event, request)

    # --- A. hook injects the SYSTEM block and the wrapped tools -----------
    request_a = ProviderRequest(
        prompt=KNOWLEDGE_TEXT,
        system_prompt=PERSONA,
        func_tool=None,
        session_id=NON_ADMIN_UMO,
    )
    await run_hook(non_admin, request_a)
    system_prompt = request_a.system_prompt
    assert PERSONA in system_prompt, system_prompt
    assert system_prompt.count(AUTONOMY_BLOCK_START) == 1, system_prompt
    assert system_prompt.count(AUTONOMY_BLOCK_END) == 1, system_prompt
    assert DEFAULT_AUTONOMY_PROMPT in system_prompt, system_prompt
    assert request_a.func_tool is not None and not request_a.func_tool.empty(), (
        "hook did not inject any tools for an ordinary conversation"
    )
    names_a = request_a.func_tool.names()
    for required_tool in ("kbm_list_kbs", "kbm_search", "kbm_add_text", "kbm_create_kb"):
        assert required_tool in names_a, (required_tool, names_a)
    for tool in request_a.func_tool:
        if tool.name.startswith("kbm_"):
            assert getattr(tool, "handler", None) is None, (
                "%s must be the permission proxy, not a raw handler" % tool.name
            )
    # Re-running the hook must not duplicate the block nor the tools.
    await run_hook(non_admin, request_a)
    assert request_a.system_prompt.count(AUTONOMY_BLOCK_START) == 1, (
        request_a.system_prompt
    )
    assert len(request_a.func_tool.names()) == len(names_a), (
        request_a.func_tool.names()
    )
    step("hook_injects_system_and_tools", tools=len(names_a))

    # Existing tools must survive the injection.
    external = FunctionTool(
        name="external_probe_tool",
        description="probe",
        parameters={"type": "object", "properties": {}},
        handler=None,
        handler_module_path="external_module.main",
    )
    request_ext = ProviderRequest(
        prompt=KNOWLEDGE_TEXT,
        system_prompt=PERSONA,
        func_tool=ToolSet(tools=[external]),
        session_id=NON_ADMIN_UMO,
    )
    await run_hook(non_admin, request_ext)
    names_ext = request_ext.func_tool.names()
    assert "external_probe_tool" in names_ext, names_ext
    assert "kbm_add_text" in names_ext, names_ext
    step("hook_preserves_external_tools")

    # --- B. scripted conversation: search -> add -> report ----------------
    created = await call_raw(
        "kbm_create_kb",
        non_admin,
        request_id="req-autonomy-kb",
        name="autonomy-e2e",
        description="autonomy acceptance",
        embedding_provider_id="e2e-embedding",
        chunk_size=64,
        chunk_overlap=0,
    )
    assert envelope_ok(created), created
    listed = await call_raw("kbm_list_kbs", non_admin)
    assert envelope_ok(listed), listed
    kb_id = None
    for item in listed["data"]["kbs"]:
        if item.get("name") == "autonomy-e2e":
            kb_id = item["kb_id"]
    assert kb_id, listed["data"]
    before = (await call_raw("kbm_list_documents", non_admin, kb_id=kb_id))["data"]["total"]

    request_b = ProviderRequest(
        prompt=KNOWLEDGE_TEXT,
        system_prompt=PERSONA,
        func_tool=None,
        session_id=NON_ADMIN_UMO,
    )
    await run_hook(non_admin, request_b)
    assert "kbm_search" in request_b.func_tool.names(), request_b.func_tool.names()

    provider = ScriptedConversationProvider(
        steps=[
            ("kbm_search", {"query": FACT_MARKER, "kb_ids": [kb_id]}),
            (
                "kbm_add_text",
                {
                    "request_id": "req-autonomy-add",
                    "kb_id": kb_id,
                    "filename": "autonomy-fact.txt",
                    "content": FACT_MARKER + " " + KNOWLEDGE_TEXT,
                },
            ),
        ]
    )
    runner = ToolLoopAgentRunner()
    await runner.reset(
        provider=provider,
        request=request_b,
        run_context=ContextWrapper(
            context=SimpleNamespace(event=non_admin, context=context)
        ),
        tool_executor=FunctionToolExecutor(),
        agent_hooks=BaseAgentRunHooks(),
        streaming=False,
    )
    async for _ in runner.step_until_done(6):
        pass
    assert runner.done(), "scripted ToolLoopAgentRunner did not finish"
    assert provider.issued_tool_calls == ["kbm_search", "kbm_add_text"], (
        provider.issued_tool_calls
    )
    assert "kbm_add_text" in provider.calls[0], provider.calls
    final = runner.get_final_llm_resp()
    assert final is not None and final.completion_text, final
    step("scripted_runner_ran", calls=len(provider.calls))

    after = (await call_raw("kbm_list_documents", non_admin, kb_id=kb_id))["data"]["total"]
    assert after == before + 1, (
        "scripted conversation did not insert a document: %d -> %d" % (before, after)
    )
    hit = await exec_wrapped(
        request_b.func_tool.get_tool("kbm_search"),
        non_admin,
        context,
        query=FACT_MARKER,
        kb_ids=[kb_id],
    )
    assert envelope_ok(hit), hit
    assert any(FACT_MARKER in item["content"] for item in hit["data"]["results"]), hit
    step("scripted_conversation_inserted", total=after)

    # --- C. chit-chat: the scripted model writes nothing ------------------
    before_chat = (
        await call_raw("kbm_list_documents", non_admin, kb_id=kb_id)
    )["data"]["total"]
    request_c = ProviderRequest(
        prompt=CHITCHAT_TEXT,
        system_prompt=PERSONA,
        func_tool=None,
        session_id=NON_ADMIN_UMO,
    )
    await run_hook(non_admin, request_c)
    provider_c = ScriptedConversationProvider(steps=[])
    runner_c = ToolLoopAgentRunner()
    await runner_c.reset(
        provider=provider_c,
        request=request_c,
        run_context=ContextWrapper(
            context=SimpleNamespace(event=non_admin, context=context)
        ),
        tool_executor=FunctionToolExecutor(),
        agent_hooks=BaseAgentRunHooks(),
        streaming=False,
    )
    async for _ in runner_c.step_until_done(4):
        pass
    assert provider_c.issued_tool_calls == [], provider_c.issued_tool_calls
    after_chat = (
        await call_raw("kbm_list_documents", non_admin, kb_id=kb_id)
    )["data"]["total"]
    assert after_chat == before_chat, "chit-chat changed storage"
    step("chitchat_no_write")

    # --- D. session-level plugin disable suppresses injection -------------
    session_config = {NON_ADMIN_UMO: {"disabled_plugins": [PLUGIN_NAME]}}
    await sp.put_async(
        "umo", NON_ADMIN_UMO, "session_plugin_config", session_config
    )
    assert (
        await SessionPluginManager.is_plugin_enabled_for_session(
            NON_ADMIN_UMO, PLUGIN_NAME
        )
        is False
    )
    request_d = ProviderRequest(
        prompt=KNOWLEDGE_TEXT,
        system_prompt=PERSONA,
        func_tool=None,
        session_id=NON_ADMIN_UMO,
    )
    await run_hook(non_admin, request_d)
    assert AUTONOMY_BLOCK_START not in request_d.system_prompt, (
        request_d.system_prompt
    )
    assert DEFAULT_AUTONOMY_PROMPT not in request_d.system_prompt
    assert (
        request_d.func_tool is None
        or request_d.func_tool.get_tool("kbm_add_text") is None
    ), request_d.func_tool
    step("session_disabled_no_injection")

    await sp.remove_async("umo", NON_ADMIN_UMO, "session_plugin_config")
    assert (
        await SessionPluginManager.is_plugin_enabled_for_session(
            NON_ADMIN_UMO, PLUGIN_NAME
        )
        is True
    )
    request_d2 = ProviderRequest(
        prompt=KNOWLEDGE_TEXT,
        system_prompt=PERSONA,
        func_tool=None,
        session_id=NON_ADMIN_UMO,
    )
    await run_hook(non_admin, request_d2)
    assert AUTONOMY_BLOCK_START in request_d2.system_prompt
    assert request_d2.func_tool is not None
    assert request_d2.func_tool.get_tool("kbm_add_text") is not None
    reenabled = await exec_wrapped(
        request_d2.func_tool.get_tool("kbm_add_text"),
        non_admin,
        context,
        request_id="req-after-enable",
        kb_id=kb_id,
        filename="after-enable.txt",
        content="AFTERENABLEMARKER session re-enabled",
    )
    assert envelope_ok(reenabled), reenabled
    step("session_reenabled_injection")

    # --- E. globally inactive tool is skipped, external tools survive -----
    await llm_tools.deactivate_llm_tool_async("kbm_add_text")
    try:
        request_e = ProviderRequest(
            prompt=KNOWLEDGE_TEXT,
            system_prompt=PERSONA,
            func_tool=ToolSet(tools=[external]),
            session_id=NON_ADMIN_UMO,
        )
        await run_hook(non_admin, request_e)
        names_e = request_e.func_tool.names()
        assert "kbm_add_text" not in names_e, names_e
        assert "kbm_search" in names_e, names_e
        assert "external_probe_tool" in names_e, names_e
        assert AUTONOMY_BLOCK_START in request_e.system_prompt
    finally:
        await llm_tools.activate_llm_tool_async("kbm_add_text", star_map=star_map)
    step("global_active_filtered")

    # --- F. per-tool admin permission is enforced by the proxy ------------
    await sp.global_put(
        "tool_permissions", {"_default": {"kbm_add_text": "admin"}}
    )
    try:
        request_f = ProviderRequest(
            prompt=KNOWLEDGE_TEXT,
            system_prompt=PERSONA,
            func_tool=None,
            session_id=NON_ADMIN_UMO,
        )
        await run_hook(non_admin, request_f)
        wrapped = request_f.func_tool.get_tool("kbm_add_text")
        assert wrapped is not None, request_f.func_tool.names()
        before_f = (
            await call_raw("kbm_list_documents", non_admin, kb_id=kb_id)
        )["data"]["total"]
        denied_text = await exec_wrapped_text(
            wrapped,
            non_admin,
            context,
            request_id="req-perm-denied",
            kb_id=kb_id,
            filename="denied.txt",
            content="PERMDENYMARKER must never be written",
        )
        assert "Permission denied" in denied_text, denied_text
        after_f = (
            await call_raw("kbm_list_documents", non_admin, kb_id=kb_id)
        )["data"]["total"]
        assert after_f == before_f, "permission-denied call changed storage"
    finally:
        await sp.global_put("tool_permissions", {})

    # The plugin-wide admin gate was not restored: after clearing the
    # per-tool entry the ordinary caller can write again through the proxy.
    request_g = ProviderRequest(
        prompt=KNOWLEDGE_TEXT,
        system_prompt=PERSONA,
        func_tool=None,
        session_id=NON_ADMIN_UMO,
    )
    await run_hook(non_admin, request_g)
    before_g = (await call_raw("kbm_list_documents", non_admin, kb_id=kb_id))["data"][
        "total"
    ]
    allowed = await exec_wrapped(
        request_g.func_tool.get_tool("kbm_add_text"),
        non_admin,
        context,
        request_id="req-after-perm",
        kb_id=kb_id,
        filename="after-perm.txt",
        content="AFTERPERMMARKER ordinary sender allowed",
    )
    assert envelope_ok(allowed), allowed
    after_g = (await call_raw("kbm_list_documents", non_admin, kb_id=kb_id))["data"][
        "total"
    ]
    assert after_g == before_g + 1, (before_g, after_g)
    step("per_tool_permission_enforced")

    # --- G. the configured SYSTEM policy replaces the default block -------
    custom_policy = "只记录经过验证的排障结论；闲聊与猜测一律不入库。"
    config_path = (
        Path(TMP_ROOT) / "data" / "config" / (PLUGIN_NAME + "_config.json")
    )
    config_path.write_text(
        json.dumps({"autonomy_system_prompt": custom_policy}), encoding="utf-8"
    )
    reloaded, reload_error = await plugin_manager.reload(PLUGIN_NAME)
    if not reloaded or star_map.get(MODULE) is None:
        raise AssertionError("plugin reload failed: %r" % (reload_error,))
    plugin = star_map[MODULE].star_cls
    _OPEN["plugin"] = plugin
    hook = find_llm_request_handler()
    request_h = ProviderRequest(
        prompt=KNOWLEDGE_TEXT,
        system_prompt=PERSONA,
        func_tool=None,
        session_id=NON_ADMIN_UMO,
    )
    await run_hook(non_admin, request_h)
    assert custom_policy in request_h.system_prompt, request_h.system_prompt
    assert DEFAULT_AUTONOMY_PROMPT not in request_h.system_prompt
    assert PERSONA in request_h.system_prompt
    assert request_h.func_tool is not None
    assert request_h.func_tool.get_tool("kbm_add_text") is not None
    step("configured_policy_applied")

    await plugin.terminate()
    _OPEN["plugin"] = None
    step("cleanup_done")


def _emit():
    print("KB_AUTONOMY_REPORT=" + json.dumps(REPORT), flush=True)


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
            "fixture ready, but main.py is not present; nothing was faked. "
            "Re-run once main.py lands.",
        )
    child = tmp_path / "_child_autonomy.py"
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
        if line.startswith("KB_AUTONOMY_REPORT="):
            report = json.loads(line[len("KB_AUTONOMY_REPORT=") :])
    if report is None:
        pytest.fail(
            f"child produced no report (rc={completed.returncode}).\n{output[-4000:]}",
        )
    if not report.get("ok"):
        pytest.fail(
            f"conversation autonomy failed: {report.get('error')}\n"
            f"{report.get('traceback')}\n{output[-2000:]}",
        )
    return report


def test_conversation_autonomy_end_to_end(tmp_path):
    report = _run_child(tmp_path)
    assert report["ok"] is True
    assert report["leftover_threads"] == [], report["leftover_threads"]
    assert report["exit_note"] is None, report["exit_note"]
    names = [step["name"] for step in report["steps"]]
    for expected in (
        "context_built",
        "plugin_loaded",
        "hook_injects_system_and_tools",
        "hook_preserves_external_tools",
        "scripted_runner_ran",
        "scripted_conversation_inserted",
        "chitchat_no_write",
        "session_disabled_no_injection",
        "session_reenabled_injection",
        "global_active_filtered",
        "per_tool_permission_enforced",
        "configured_policy_applied",
        "cleanup_done",
    ):
        if expected not in names:
            raise AssertionError(f"missing autonomy step {expected!r}: {names}")
