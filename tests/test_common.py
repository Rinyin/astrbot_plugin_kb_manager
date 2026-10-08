"""Contract tests for the public primitives in ``common.py``."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Import the plugin as a package (``data/plugins``'s parent on sys.path); the
# plugin directory itself must never be on sys.path, and the plugin modules
# only use relative imports internally.
DATA_ROOT = Path(__file__).resolve().parents[3]
if str(DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ROOT))

from plugins.astrbot_plugin_kb_manager.common import (  # noqa: E402
    CODE_INVALID_ARGUMENT,
    JOB_STATES,
    STATUS_FAILED,
    STATUS_PARTIAL,
    STATUS_SUCCEEDED,
    TERMINAL_STATES,
    KBError,
    Scope,
    SourceDocument,
    as_kb_error,
    decode_scope,
    encode_scope,
    ensure_jsonable,
    error_result,
    fingerprint,
    is_terminal,
    json_dumps,
    kb_lock_key,
    make_result,
    ok_result,
    request_key,
)


def test_scope_roundtrip():
    encoded = encode_scope("aiocqhttp:GroupMessage:42", "10001")
    assert encoded == '["aiocqhttp:GroupMessage:42","10001"]'
    assert decode_scope(encoded) == Scope("aiocqhttp:GroupMessage:42", "10001")
    assert Scope("umo", "sender").encode() == encode_scope("umo", "sender")


def test_scope_rejects_malformed_input():
    for raw in ("", "not-json", "{}", '["only-one"]', '["a","b","c"]', "[1,2]"):
        with pytest.raises(KBError) as excinfo:
            decode_scope(raw)
        assert excinfo.value.code == CODE_INVALID_ARGUMENT
    with pytest.raises(KBError):
        encode_scope("", "sender")
    with pytest.raises(KBError):
        encode_scope("umo", "")


def test_kb_error_roundtrip():
    error = KBError("kb_not_found", "missing", details={"kb_id": "k1"}, partial=True)
    payload = error.to_dict()
    assert payload == {
        "code": "kb_not_found",
        "message": "missing",
        "details": {"kb_id": "k1"},
        "partial": True,
    }
    restored = KBError.from_dict(payload)
    assert restored.code == "kb_not_found"
    assert restored.partial is True
    assert restored.details == {"kb_id": "k1"}
    assert str(error) == "kb_not_found: missing"


def test_as_kb_error_passthrough_and_wrap():
    original = KBError("name_conflict", "dup")
    assert as_kb_error(original) is original
    wrapped = as_kb_error(ValueError("boom"))
    assert wrapped.code == "internal"
    assert wrapped.details == {"type": "ValueError"}


def test_result_envelope_shapes():
    ok = ok_result({"kbs": []}, job_id="job-1")
    assert ok == {
        "status": STATUS_SUCCEEDED,
        "job_id": "job-1",
        "data": {"kbs": []},
        "error": None,
    }
    failed = error_result(KBError("kb_not_found", "nope"))
    assert failed["status"] == STATUS_FAILED
    assert failed["data"] is None
    assert failed["error"]["code"] == "kb_not_found"

    partial = error_result(KBError("embedding_failed", "half", partial=True))
    assert partial["status"] == STATUS_PARTIAL
    assert set(partial) == {"status", "job_id", "data", "error"}

    with pytest.raises(KBError):
        make_result("not-a-status")


def test_job_state_helpers():
    assert {
        "queued",
        "running",
        "succeeded",
        "failed",
        "partial",
        "interrupted",
    } == JOB_STATES
    assert TERMINAL_STATES == {"succeeded", "failed", "partial", "interrupted"}
    assert is_terminal(STATUS_SUCCEEDED)
    assert not is_terminal("running")


def test_json_helpers_reject_unsafe_values():
    assert json_dumps({"b": 1, "a": 2}, sort_keys=True) == '{"a":2,"b":1}'
    assert ensure_jsonable({"nested": [1, 2, {"x": None}]}) == {
        "nested": [1, 2, {"x": None}]
    }
    for bad in (b"bytes", float("nan"), float("inf"), {1: "non-string-key"}):
        with pytest.raises(KBError):
            ensure_jsonable(bad)


def test_fingerprint_is_stable():
    first = fingerprint({"operation": "op", "payload": {"b": 1, "a": 2}})
    second = fingerprint({"payload": {"a": 2, "b": 1}, "operation": "op"})
    assert first == second


def test_key_helpers():
    assert kb_lock_key("kb1") == "kb:kb1"
    key = request_key('["umo","sender"]', "req-1")
    assert key == '["[\\"umo\\",\\"sender\\"]","req-1"]'
    with pytest.raises(KBError):
        kb_lock_key("")


def test_source_document_validation():
    doc = SourceDocument("a.txt", b"hello", "file")
    assert doc.size == 5
    assert doc.to_meta() == {"filename": "a.txt", "size": 5, "source": "file"}
    assert isinstance(doc.content, bytes)

    with pytest.raises(KBError):
        SourceDocument("", b"x", "file")
    with pytest.raises(KBError):
        SourceDocument("a.txt", "not-bytes", "file")
    with pytest.raises(KBError):
        SourceDocument("a.txt", b"x", "")
