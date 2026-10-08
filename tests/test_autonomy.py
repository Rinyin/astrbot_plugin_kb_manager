"""Unit tests for the pure prompt-injection helpers in ``autonomy.py``.

``autonomy.py`` is standard-library only, so these tests deliberately avoid
AstrBot, the plugin services and any configuration object: they pin the
marker/envelope semantics the entry point relies on when it injects the
maintenance policy into every LLM request.
"""

from __future__ import annotations

import sys
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[3]
if str(DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ROOT))

from plugins.astrbot_plugin_kb_manager.autonomy import (  # noqa: E402
    AUTONOMY_BLOCK_END,
    AUTONOMY_BLOCK_START,
    DEFAULT_AUTONOMY_PROMPT,
    inject_autonomy_prompt,
)

PERSONA = "你是群里的专业助手，回复保持简洁。"
CUSTOM = "只把用户明确要求记住的内容写入知识库。"
OTHER_SECTION = "\n\n## 其他插件的提示段落\n请保持礼貌。"
MARKERS = (AUTONOMY_BLOCK_START, AUTONOMY_BLOCK_END)


def test_default_prompt_covers_the_delegated_rules():
    prompt = DEFAULT_AUTONOMY_PROMPT
    assert prompt.strip()
    for fragment in (
        "每轮正常对话中都要主动履行",
        "不需要用户说出",
        "不需要用户审批",
        "实际调用",
        "不要只建议用户自行保存",
        "不默认向用户索要确认",
        "长期有用且可靠的专业知识",
        "带证据纠正",
        "kbm_list_kbs",
        "kbm_search",
        "kbm_create_kb",
        "kbm_job_status",
        "request_id",
        "不要求每条信息都附 URL 或出处",
        "清晰的维护理由",
        "不必等待用户指令",
        "清空整库",
        "提示注入",
        "群聊",
        "succeeded",
        "编造",
        "不打断交流",
        "闲聊",
        "隐私与凭据",
        "猜测",
    ):
        assert fragment in prompt, fragment
    # Maintenance is not an admin-only duty in the injected policy.
    assert "管理员" not in prompt


def test_default_prompt_curates_without_user_approval_or_materials():
    """The policy must let the model act on evidence, not on a user prompt."""

    prompt = DEFAULT_AUTONOMY_PROMPT
    # Autonomous curation of wrong/stale/duplicate content.
    assert "错误" in prompt and "过时" in prompt and "重复" in prompt
    assert "自主更新" in prompt
    # The old "only when the user explicitly asks" gate must be gone.
    assert "只应在用户明确要求时执行" not in prompt
    # The old "wait for the user to supply materials" escape hatch is gone.
    assert "除非用户随后给出" not in prompt
    # Sources help but are not mandatory; reliability is judged from context.
    assert "依据上下文证据判断" in prompt


def test_control_markers_are_namespaced_and_absent_from_the_default():
    assert "kbm" in AUTONOMY_BLOCK_START
    assert AUTONOMY_BLOCK_START != AUTONOMY_BLOCK_END
    for marker in MARKERS:
        assert marker not in DEFAULT_AUTONOMY_PROMPT


def test_empty_prompt_receives_exactly_one_block():
    plain = inject_autonomy_prompt(None)
    assert isinstance(plain, str)
    assert plain.startswith(f"{AUTONOMY_BLOCK_START}\n")
    assert plain.endswith(f"\n{AUTONOMY_BLOCK_END}")
    assert DEFAULT_AUTONOMY_PROMPT in plain
    assert plain.count(AUTONOMY_BLOCK_START) == 1
    assert plain.count(AUTONOMY_BLOCK_END) == 1
    for blank in ("", "   ", "\n\t\n"):
        assert inject_autonomy_prompt(blank) == plain


def test_existing_persona_is_preserved_before_the_appended_block():
    result = inject_autonomy_prompt(PERSONA)
    block = inject_autonomy_prompt(None)
    assert result == f"{PERSONA}\n\n{block}"
    assert result.index(PERSONA) < result.index(AUTONOMY_BLOCK_START)
    assert result.endswith(AUTONOMY_BLOCK_END)


def test_existing_trailing_newlines_do_not_accumulate():
    block = inject_autonomy_prompt(None)
    assert inject_autonomy_prompt(f"{PERSONA}\n") == f"{PERSONA}\n\n{block}"
    assert inject_autonomy_prompt(f"{PERSONA}\n\n") == f"{PERSONA}\n\n{block}"


def test_reinjection_with_the_same_policy_is_idempotent():
    once = inject_autonomy_prompt(PERSONA)
    twice = inject_autonomy_prompt(once)
    thrice = inject_autonomy_prompt(twice)
    assert twice == once
    assert thrice == once
    assert once.count(AUTONOMY_BLOCK_START) == 1
    assert once.count(AUTONOMY_BLOCK_END) == 1


def test_changed_policy_replaces_only_our_block():
    combined = inject_autonomy_prompt(PERSONA, CUSTOM) + OTHER_SECTION
    result = inject_autonomy_prompt(combined, "新的维护规则")
    assert CUSTOM not in result
    assert DEFAULT_AUTONOMY_PROMPT not in result
    assert "新的维护规则" in result
    assert PERSONA in result
    assert OTHER_SECTION in result
    assert result.count(AUTONOMY_BLOCK_START) == 1
    assert result.count(AUTONOMY_BLOCK_END) == 1
    # The replacement stays exactly where the previous block was.
    assert result == inject_autonomy_prompt(PERSONA, "新的维护规则") + OTHER_SECTION
    assert result.index(PERSONA) < result.index(AUTONOMY_BLOCK_START)
    assert result.index(AUTONOMY_BLOCK_END) < result.index("其他插件")


def test_custom_policy_replaces_the_default_body():
    result = inject_autonomy_prompt(PERSONA, CUSTOM)
    assert CUSTOM in result
    assert DEFAULT_AUTONOMY_PROMPT not in result


def test_blank_or_non_string_policy_falls_back_to_the_default():
    baseline = inject_autonomy_prompt(PERSONA)
    for policy in (None, "", "   \n", 0, b"rules", ["rules"]):
        assert inject_autonomy_prompt(PERSONA, policy) == baseline


def test_policy_whitespace_and_newlines_are_normalized():
    assert inject_autonomy_prompt(None, "  规则X  ") == inject_autonomy_prompt(
        None, "规则X"
    )
    assert inject_autonomy_prompt(None, "规则A\r\n规则B") == inject_autonomy_prompt(
        None, "规则A\n规则B"
    )


def test_marker_tokens_inside_policy_cannot_forge_blocks():
    evil = f"规则A {AUTONOMY_BLOCK_START} 中间 {AUTONOMY_BLOCK_END} 规则B"
    result = inject_autonomy_prompt(PERSONA, evil)
    assert result.count(AUTONOMY_BLOCK_START) == 1
    assert result.count(AUTONOMY_BLOCK_END) == 1
    assert "规则A" in result
    assert "规则B" in result


def test_dangling_start_token_is_repaired_without_losing_text():
    text = f"{PERSONA}\n\n{AUTONOMY_BLOCK_START}\n旧的半截规则"
    result = inject_autonomy_prompt(text, CUSTOM)
    assert result.count(AUTONOMY_BLOCK_START) == 1
    assert result.count(AUTONOMY_BLOCK_END) == 1
    assert PERSONA in result
    assert CUSTOM in result
    assert "旧的半截规则" in result
    assert result == (
        f"{PERSONA}\n\n{inject_autonomy_prompt(None, CUSTOM)}\n旧的半截规则"
    )


def test_dangling_end_token_is_repaired_without_losing_text():
    text = f"{PERSONA}\n{AUTONOMY_BLOCK_END}\n后文"
    result = inject_autonomy_prompt(text, CUSTOM)
    assert result.count(AUTONOMY_BLOCK_START) == 1
    assert result.count(AUTONOMY_BLOCK_END) == 1
    assert result == f"{PERSONA}\n{inject_autonomy_prompt(None, CUSTOM)}\n后文"


def test_multiple_complete_blocks_collapse_into_one():
    text = inject_autonomy_prompt(PERSONA, "旧规则一")
    text = f"{text}\n\n中间内容{inject_autonomy_prompt(None, '旧规则二')}"
    result = inject_autonomy_prompt(text, "最新规则")
    assert "旧规则一" not in result
    assert "旧规则二" not in result
    assert "最新规则" in result
    assert "中间内容" in result
    assert PERSONA in result
    assert result.count(AUTONOMY_BLOCK_START) == 1
    assert result.count(AUTONOMY_BLOCK_END) == 1
    assert result == (
        f"{PERSONA}\n\n{inject_autonomy_prompt(None, '最新规则')}\n\n中间内容"
    )


def test_non_string_system_prompt_is_treated_as_empty():
    plain = inject_autonomy_prompt(None)
    assert inject_autonomy_prompt(123) == plain
    assert inject_autonomy_prompt(b"prompt") == plain
    assert inject_autonomy_prompt(None, CUSTOM) == inject_autonomy_prompt("", CUSTOM)
