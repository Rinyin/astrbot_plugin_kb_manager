"""Behavioural tests for ``JobManager`` using real SQLite and real asyncio."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

import aiosqlite
import pytest

# Import the plugin as a package (``data/plugins``'s parent on sys.path); the
# plugin directory itself must never be on sys.path, and the plugin modules
# only use relative imports internally.
DATA_ROOT = Path(__file__).resolve().parents[3]
if str(DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ROOT))

from plugins.astrbot_plugin_kb_manager.common import (  # noqa: E402
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    STATUS_PARTIAL,
    STATUS_RUNNING,
    STATUS_SUCCEEDED,
    KBError,
    encode_scope,
    fingerprint,
)
from plugins.astrbot_plugin_kb_manager.jobs import JobManager  # noqa: E402

TERMINAL = {STATUS_SUCCEEDED, STATUS_FAILED, STATUS_PARTIAL, STATUS_INTERRUPTED}


def scope(umo: str, sender: str = "10001") -> str:
    return encode_scope(umo, sender)


def _read_job_row(data_dir: Path, request_id: str) -> dict | None:
    db_path = data_dir / "jobs.sqlite3"
    if not db_path.exists():
        return None
    with sqlite3.connect(db_path, timeout=5) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT job_id, status, result, error FROM jobs WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        return dict(row) if row is not None else None


async def _poll_terminal(
    mgr: JobManager, umo: str, job_id: str, timeout: float = 3.0
) -> dict:
    deadline = time.monotonic() + timeout
    result = await mgr.get(umo, job_id)
    while result["status"] not in TERMINAL and time.monotonic() < deadline:
        await asyncio.sleep(0.02)
        result = await mgr.get(umo, job_id)
    return result


# ---------------------------------------------------------------------------
# Persistence, restart, idempotency
# ---------------------------------------------------------------------------


async def _persistence_scenario(tmp_path: Path) -> None:
    data_dir = tmp_path / "plugin_data"
    mgr = JobManager(data_dir)
    await mgr.initialize()
    umo = scope("aiocqhttp:GroupMessage:1")
    calls: list[str] = []

    async def work(progress):
        calls.append("run")
        await progress("chunking", 1, 2)
        return {"kb_id": "kb1", "added": 1}

    result = await mgr.submit(
        umo, "req-1", "add_document", {"kb_id": "kb1"}, "kb:kb1", work
    )
    assert result["status"] == STATUS_SUCCEEDED
    assert result["job_id"]
    assert result["data"] == {"kb_id": "kb1", "added": 1}
    assert result["error"] is None
    assert calls == ["run"]
    await mgr.close()

    db_path = data_dir / "jobs.sqlite3"
    assert db_path.exists()
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT status, progress, fingerprint, payload FROM jobs WHERE request_id = ?",
            ("req-1",),
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == STATUS_SUCCEEDED
    assert json.loads(rows[0][1]) == {"stage": "chunking", "current": 1, "total": 2}
    assert rows[0][2] == fingerprint(
        {"operation": "add_document", "payload": {"kb_id": "kb1"}}
    )
    assert json.loads(rows[0][3]) == {"kb_id": "kb1"}

    rebuilt = JobManager(data_dir)
    await rebuilt.initialize()
    got = await rebuilt.get(umo, result["job_id"])
    assert got["status"] == STATUS_SUCCEEDED
    assert got["data"] == {"kb_id": "kb1", "added": 1}

    reruns: list[str] = []

    async def rerun(progress):
        reruns.append("run")
        return {"should": "not run"}

    reused = await rebuilt.submit(
        umo, "req-1", "add_document", {"kb_id": "kb1"}, "kb:kb1", rerun
    )
    assert reused["job_id"] == result["job_id"]
    assert reused["status"] == STATUS_SUCCEEDED
    assert reused["data"] == {"kb_id": "kb1", "added": 1}

    conflict = await rebuilt.submit(
        umo, "req-1", "add_document", {"kb_id": "other"}, "kb:kb1", rerun
    )
    assert conflict["status"] == STATUS_FAILED
    assert conflict["error"]["code"] == "idempotency_conflict"
    assert conflict["error"]["details"]["existing_job_id"] == result["job_id"]
    assert reruns == []
    await rebuilt.close()


def test_persistence_and_reuse_across_restart(tmp_path):
    asyncio.run(_persistence_scenario(tmp_path))


async def _restart_scenario(tmp_path: Path) -> None:
    umo = scope("umo-restart")
    mgr = JobManager(tmp_path)
    await mgr.initialize()
    await mgr.close()

    stale_fp = fingerprint({"operation": "op", "payload": {}})
    with sqlite3.connect(tmp_path / "jobs.sqlite3") as conn:
        conn.execute(
            "INSERT INTO jobs (job_id, scope, request_id, operation, fingerprint, payload,"
            " lock_key, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "stale-job",
                umo,
                "stale-req",
                "op",
                stale_fp,
                "{}",
                "kb:1",
                "running",
                1,
                1,
            ),
        )
        conn.commit()

    rebuilt = JobManager(tmp_path)
    await rebuilt.initialize()
    recovered = await rebuilt.get(umo, "stale-job")
    assert recovered["status"] == STATUS_INTERRUPTED
    assert recovered["error"]["code"] == "interrupted"

    async def must_not_run(progress):
        raise AssertionError("interrupted jobs must not be replayed")

    same = await rebuilt.submit(
        umo, "stale-req", "op", {}, "kb:1", must_not_run, wait_seconds=0
    )
    assert same["job_id"] == "stale-job"
    assert same["status"] == STATUS_INTERRUPTED

    changed = await rebuilt.submit(
        umo, "stale-req", "op", {"changed": True}, "kb:1", must_not_run, wait_seconds=0
    )
    assert changed["status"] == STATUS_FAILED
    assert changed["error"]["code"] == "idempotency_conflict"
    await rebuilt.close()


def test_restart_marks_non_terminal_interrupted_without_replay(tmp_path):
    asyncio.run(_restart_scenario(tmp_path))


# ---------------------------------------------------------------------------
# Idempotency and scope isolation
# ---------------------------------------------------------------------------


async def _concurrent_same_key_scenario(tmp_path: Path) -> None:
    mgr = JobManager(tmp_path)
    await mgr.initialize()
    umo = scope("umo-concurrent")
    counter = {"runs": 0}

    async def work(progress):
        counter["runs"] += 1
        await asyncio.sleep(0.1)
        return {"ok": True}

    results = await asyncio.gather(
        *(mgr.submit(umo, "same-key", "op", {"x": 1}, "kb:1", work) for _ in range(5))
    )
    assert counter["runs"] == 1
    assert len({r["job_id"] for r in results}) == 1
    assert all(r["status"] == STATUS_SUCCEEDED for r in results)
    await mgr.close()


def test_concurrent_same_key_runs_work_once(tmp_path):
    asyncio.run(_concurrent_same_key_scenario(tmp_path))


async def _cross_scope_scenario(tmp_path: Path) -> None:
    mgr = JobManager(tmp_path)
    await mgr.initialize()
    runs: list[int] = []

    async def work(progress):
        runs.append(1)
        return {"ok": True}

    first = await mgr.submit(scope("umo-a"), "shared-req", "op", {"x": 1}, "kb:1", work)
    second = await mgr.submit(
        scope("umo-b"), "shared-req", "op", {"x": 1}, "kb:1", work
    )
    assert first["job_id"] != second["job_id"]
    assert first["status"] == STATUS_SUCCEEDED
    assert second["status"] == STATUS_SUCCEEDED
    assert len(runs) == 2
    await mgr.close()


def test_cross_scope_same_request_id_are_independent(tmp_path):
    asyncio.run(_cross_scope_scenario(tmp_path))


async def _isolation_scenario(tmp_path: Path) -> None:
    mgr = JobManager(tmp_path)
    await mgr.initialize()
    owner = scope("umo-owner")
    intruder = scope("umo-intruder")

    async def work(progress):
        return {"ok": True}

    submitted = await mgr.submit(owner, "isolated", "op", {}, "kb:1", work)
    leaked = await mgr.get(intruder, submitted["job_id"])
    assert leaked["status"] == STATUS_FAILED
    assert leaked["error"]["code"] == "job_not_found"

    visible = await mgr.get(owner, submitted["job_id"])
    assert visible["status"] == STATUS_SUCCEEDED

    missing = await mgr.get(owner, "no-such-job")
    assert missing["status"] == STATUS_FAILED
    assert missing["error"]["code"] == "job_not_found"
    await mgr.close()


def test_get_is_scope_isolated(tmp_path):
    asyncio.run(_isolation_scenario(tmp_path))


# ---------------------------------------------------------------------------
# Scheduling: per-key FIFO and global concurrency
# ---------------------------------------------------------------------------


async def _serialization_scenario(tmp_path: Path) -> None:
    mgr = JobManager(tmp_path, max_concurrent=3)
    await mgr.initialize()
    order: list[str] = []
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def first_work(progress):
        order.append("w1-start")
        first_started.set()
        await release_first.wait()
        order.append("w1-end")
        return {"n": 1}

    async def second_work(progress):
        order.append("w2-start")
        return {"n": 2}

    first_task = asyncio.create_task(
        mgr.submit(
            scope("umo-a"), "r1", "op", {}, "kb:shared", first_work, wait_seconds=5
        )
    )
    await asyncio.wait_for(first_started.wait(), 2)
    second_task = asyncio.create_task(
        mgr.submit(
            scope("umo-b"), "r2", "op", {}, "kb:shared", second_work, wait_seconds=5
        )
    )
    await asyncio.sleep(0.1)
    assert "w2-start" not in order
    release_first.set()
    first = await asyncio.wait_for(first_task, 3)
    second = await asyncio.wait_for(second_task, 3)
    assert first["status"] == STATUS_SUCCEEDED
    assert second["status"] == STATUS_SUCCEEDED
    assert order.index("w1-end") < order.index("w2-start")
    await mgr.close()


def test_same_lock_key_serializes_across_scopes(tmp_path):
    asyncio.run(_serialization_scenario(tmp_path))


async def _parallel_keys_scenario(tmp_path: Path) -> None:
    mgr = JobManager(tmp_path, max_concurrent=2)
    await mgr.initialize()
    started: list[int] = []
    both_running = asyncio.Event()
    gate = asyncio.Event()

    async def work(progress):
        started.append(1)
        if len(started) == 2:
            both_running.set()
        await gate.wait()
        return {"ok": True}

    tasks = [
        asyncio.create_task(
            mgr.submit(
                scope(f"umo-{i}"), f"r{i}", "op", {}, f"kb:{i}", work, wait_seconds=5
            )
        )
        for i in range(2)
    ]
    await asyncio.wait_for(both_running.wait(), 2)
    gate.set()
    results = [await asyncio.wait_for(task, 3) for task in tasks]
    assert all(r["status"] == STATUS_SUCCEEDED for r in results)
    await mgr.close()


def test_different_lock_keys_run_concurrently(tmp_path):
    asyncio.run(_parallel_keys_scenario(tmp_path))


async def _global_cap_scenario(tmp_path: Path) -> None:
    mgr = JobManager(tmp_path, max_concurrent=2)
    await mgr.initialize()
    started: list[int] = []
    two_running = asyncio.Event()
    gate = asyncio.Event()

    async def work(progress):
        started.append(1)
        if len(started) == 2:
            two_running.set()
        await gate.wait()
        return {"ok": True}

    tasks = [
        asyncio.create_task(
            mgr.submit(
                scope(f"umo-cap-{i}"),
                f"r{i}",
                "op",
                {},
                f"kb:cap{i}",
                work,
                wait_seconds=5,
            )
        )
        for i in range(3)
    ]
    await asyncio.wait_for(two_running.wait(), 2)
    await asyncio.sleep(0.1)
    assert len(started) == 2
    gate.set()
    results = await asyncio.gather(*tasks)
    assert all(r["status"] == STATUS_SUCCEEDED for r in results)
    assert len(started) == 3
    await mgr.close()


def test_global_concurrency_limit_is_respected(tmp_path):
    asyncio.run(_global_cap_scenario(tmp_path))


# ---------------------------------------------------------------------------
# Timeout / cancellation shield
# ---------------------------------------------------------------------------


async def _timeout_scenario(tmp_path: Path) -> None:
    mgr = JobManager(tmp_path)
    await mgr.initialize()
    umo = scope("umo-timeout")
    done: list[str] = []

    async def slow(progress):
        await asyncio.sleep(0.25)
        done.append("done")
        return {"ok": True}

    quick = await mgr.submit(umo, "slow", "op", {}, "kb:1", slow, wait_seconds=0.05)
    assert quick["status"] in ("queued", "running")
    assert quick["job_id"]
    # Non-terminal envelopes expose live job metadata inside ``data``.
    assert quick["data"]["operation"] == "op"
    assert quick["data"]["progress"] is None
    assert isinstance(quick["data"]["updated_at"], int)

    # get() stays responsive while work runs (transactions are not held open).
    current = await asyncio.wait_for(mgr.get(umo, quick["job_id"]), 1)
    assert current["status"] in ("queued", "running")
    assert current["data"]["operation"] == "op"

    final = await _poll_terminal(mgr, umo, quick["job_id"])
    assert final["status"] == STATUS_SUCCEEDED
    assert final["data"] == {"ok": True}
    assert done == ["done"]

    async def slow2(progress):
        await asyncio.sleep(0.2)
        done.append("done2")
        return {"ok": 2}

    waiter = asyncio.create_task(
        mgr.submit(umo, "slow-2", "op", {}, "kb:1", slow2, wait_seconds=5)
    )
    await asyncio.sleep(0.05)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    await asyncio.sleep(0.35)

    row = _read_job_row(tmp_path, "slow-2")
    assert row is not None
    assert row["status"] == STATUS_SUCCEEDED
    assert json.loads(row["result"]) == {"ok": 2}
    assert done == ["done", "done2"]
    await mgr.close()


def test_timeout_and_cancel_do_not_cancel_submitted_work(tmp_path):
    asyncio.run(_timeout_scenario(tmp_path))


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


async def _error_mapping_scenario(tmp_path: Path) -> None:
    mgr = JobManager(tmp_path)
    await mgr.initialize()
    umo = scope("umo-errors")

    async def partial_work(progress):
        raise KBError("embedding_failed", "provider unreachable", partial=True)

    partial = await mgr.submit(umo, "partial", "op", {}, "kb:1", partial_work)
    assert partial["status"] == STATUS_PARTIAL
    assert partial["error"]["code"] == "embedding_failed"
    assert partial["error"]["partial"] is True
    assert partial["data"] is None

    async def boom(progress):
        raise ValueError("http://user:secret@example.com/kb?token=abc")

    failed = await mgr.submit(umo, "boom", "op", {}, "kb:1", boom)
    assert failed["status"] == STATUS_FAILED
    assert failed["error"]["code"] == "internal"
    serialized = json.dumps(failed)
    assert "secret" not in serialized
    assert "http" not in serialized
    await mgr.close()


def test_error_mapping_and_sanitization(tmp_path):
    asyncio.run(_error_mapping_scenario(tmp_path))


def test_unknown_exception_logs_traceback(tmp_path, caplog, monkeypatch):
    import plugins.astrbot_plugin_kb_manager.jobs as jobs_module

    # Pin a stdlib logger so the assertion is independent of whether the
    # AstrBot framework logger happens to be importable in the test session.
    test_logger = logging.getLogger("kb-jobs-traceback-test")
    monkeypatch.setattr(jobs_module, "logger", test_logger)

    async def scenario():
        mgr = JobManager(tmp_path)
        await mgr.initialize()

        async def boom(progress):
            raise ValueError("explosive-detail")

        await mgr.submit(scope("umo-log"), "boom", "op", {}, "kb:1", boom)
        await mgr.close()

    with caplog.at_level(logging.ERROR, logger="kb-jobs-traceback-test"):
        asyncio.run(scenario())
    assert "ValueError" in caplog.text
    assert "explosive-detail" in caplog.text


# ---------------------------------------------------------------------------
# Lifecycle and validation
# ---------------------------------------------------------------------------


async def _close_scenario(tmp_path: Path) -> None:
    umo = scope("umo-close")
    mgr = JobManager(tmp_path)
    await mgr.initialize()
    started = asyncio.Event()
    progress_refs: list = []

    async def long_work(progress):
        progress_refs.append(progress)
        await progress("embedding", 1, 10)
        started.set()
        await asyncio.sleep(30)
        return {"never": True}

    submitted = await mgr.submit(
        umo, "long", "op", {}, "kb:1", long_work, wait_seconds=0
    )
    await asyncio.wait_for(started.wait(), 2)
    await mgr.close()
    await mgr.close()  # idempotent

    rejected = await mgr.submit(
        umo, "after-close", "op", {}, "kb:1", long_work, wait_seconds=0
    )
    assert rejected["status"] == STATUS_FAILED
    assert rejected["error"]["code"] == "unavailable"

    leftovers = [
        t for t in asyncio.all_tasks() if (t.get_name() or "").startswith("kb-job:")
    ]
    assert leftovers == []

    # A stale progress callback must not touch the database after close.
    await progress_refs[0]("storing", 2, 10)

    db_path = tmp_path / "jobs.sqlite3"
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT status, progress FROM jobs WHERE job_id = ?", (submitted["job_id"],)
        ).fetchone()
    assert row[0] == STATUS_INTERRUPTED
    assert json.loads(row[1])["stage"] == "embedding"

    rebuilt = JobManager(tmp_path)
    await rebuilt.initialize()
    got = await rebuilt.get(umo, submitted["job_id"])
    assert got["status"] == STATUS_INTERRUPTED
    await rebuilt.close()


def test_close_is_idempotent_and_cleans_up(tmp_path):
    asyncio.run(_close_scenario(tmp_path))


async def _validation_scenario(tmp_path: Path) -> None:
    umo = scope("umo-invalid")

    async def work(progress):
        return {}

    uninitialized = JobManager(tmp_path / "never_initialized")
    result = await uninitialized.submit(umo, "r", "op", {}, "kb:1", work)
    assert result["status"] == STATUS_FAILED
    assert result["error"]["code"] == "unavailable"
    got = await uninitialized.get(umo, "whatever")
    assert got["error"]["code"] == "unavailable"
    await uninitialized.close()

    with pytest.raises(KBError):
        JobManager(tmp_path, max_concurrent=0)

    mgr = JobManager(tmp_path)
    await mgr.initialize()
    with pytest.raises(KBError) as scope_error:
        await mgr.submit("not-json", "r", "op", {}, "kb:1", work)
    assert scope_error.value.code == "invalid_argument"
    with pytest.raises(KBError):
        await mgr.submit(umo, "", "op", {}, "kb:1", work)
    with pytest.raises(KBError):
        await mgr.submit(umo, "r", "op", {"bad": b"bytes"}, "kb:1", work)
    with pytest.raises(KBError):
        await mgr.submit(umo, "r", "op", {}, "kb:1", "not-callable")
    await mgr.initialize()  # repeated calls are safe
    await mgr.close()
    with pytest.raises(KBError) as closed_error:
        await mgr.initialize()
    assert closed_error.value.code == "unavailable"


def test_submit_validation_and_unavailable_manager(tmp_path):
    asyncio.run(_validation_scenario(tmp_path))


# ---------------------------------------------------------------------------
# Live metadata for queued/running jobs
# ---------------------------------------------------------------------------


async def _progress_scenario(tmp_path: Path) -> None:
    mgr = JobManager(tmp_path)
    await mgr.initialize()
    umo = scope("umo-progress")
    release = asyncio.Event()
    reported = asyncio.Event()

    async def work(progress):
        await progress("embedding", 3, 10)
        reported.set()
        await release.wait()
        return {"ok": True}

    submitted = await mgr.submit(
        umo, "progress", "add_document", {}, "kb:1", work, wait_seconds=0
    )
    await asyncio.wait_for(reported.wait(), 2)

    current = await mgr.get(umo, submitted["job_id"])
    assert current["status"] in ("queued", STATUS_RUNNING)
    assert current["data"]["operation"] == "add_document"
    assert current["data"]["progress"] == {
        "stage": "embedding",
        "current": 3,
        "total": 10,
    }
    assert current["data"]["updated_at"] >= current["data"]["created_at"]

    release.set()
    final = await _poll_terminal(mgr, umo, submitted["job_id"])
    assert final["status"] == STATUS_SUCCEEDED
    assert final["data"] == {"ok": True}
    await mgr.close()


def test_running_job_returns_operation_progress_and_timestamps(tmp_path):
    asyncio.run(_progress_scenario(tmp_path))


# ---------------------------------------------------------------------------
# Commit failures, rollback and durable outcomes
# ---------------------------------------------------------------------------


class _FlakyCommit:
    """Inject real commit failures into every live aiosqlite connection."""

    def __init__(self, monkeypatch) -> None:
        self.armed = False
        self.remaining = 0
        self._real = aiosqlite.Connection.commit

        async def fake_commit(connection) -> None:
            if self.armed and self.remaining > 0:
                self.remaining -= 1
                raise sqlite3.OperationalError("injected commit failure")
            await self._real(connection)

        monkeypatch.setattr(aiosqlite.Connection, "commit", fake_commit)


async def _commit_failure_once_scenario(tmp_path: Path, monkeypatch) -> None:
    mgr = JobManager(tmp_path)
    await mgr.initialize()
    umo = scope("umo-commit")
    runs: list[int] = []
    flaky = _FlakyCommit(monkeypatch)

    async def work(progress):
        runs.append(1)
        flaky.armed = True
        flaky.remaining = 1  # fail exactly the terminal succeeded write
        return {"written": True}

    result = await mgr.submit(umo, "commit-fail", "op", {}, "kb:1", work)
    assert result["status"] == STATUS_PARTIAL
    assert result["error"]["code"] == "storage_failed"
    assert result["error"]["partial"] is True
    assert result["data"] is None
    assert runs == [1]

    # The caller must never see the failed, uncommitted succeeded row; the
    # durable fallback marks the job as an explicit uncertain outcome.
    row = _read_job_row(tmp_path, "commit-fail")
    assert row is not None
    assert row["status"] == STATUS_PARTIAL
    assert json.loads(row["error"])["code"] == "storage_failed"

    async def must_not_run(progress):
        runs.append(2)
        return {}

    again = await mgr.submit(umo, "commit-fail", "op", {}, "kb:1", must_not_run)
    assert again["job_id"] == result["job_id"]
    assert again["status"] == STATUS_PARTIAL
    assert again["error"]["code"] == "storage_failed"
    assert runs == [1]
    await mgr.close()


def test_terminal_commit_failure_reports_uncertain_outcome_without_replay(
    tmp_path, monkeypatch
):
    asyncio.run(_commit_failure_once_scenario(tmp_path, monkeypatch))


async def _rollback_scenario(tmp_path: Path, monkeypatch) -> None:
    mgr = JobManager(tmp_path)
    await mgr.initialize()
    umo = scope("umo-rollback")
    flaky = _FlakyCommit(monkeypatch)

    async def failing(progress):
        flaky.armed = True
        flaky.remaining = 2  # both the succeeded write and the fallback fail
        return {"written": True}

    result = await mgr.submit(umo, "rollback", "op", {}, "kb:1", failing)
    assert result["status"] == STATUS_PARTIAL
    assert result["error"]["code"] == "storage_failed"

    flaky.armed = False

    async def later(progress):
        return {"ok": True}

    done = await mgr.submit(umo, "later", "op", {}, "kb:1", later)
    assert done["status"] == STATUS_SUCCEEDED

    # The later successful commit must not flush the earlier failed writes:
    # the failed job stays at its last durable state (running) and its
    # uncertain outcome only lives in process memory.
    failed_row = _read_job_row(tmp_path, "rollback")
    assert failed_row is not None
    assert failed_row["status"] == STATUS_RUNNING
    assert _read_job_row(tmp_path, "later")["status"] == STATUS_SUCCEEDED
    await mgr.close()


def test_failed_short_transactions_are_rolled_back(tmp_path, monkeypatch):
    asyncio.run(_rollback_scenario(tmp_path, monkeypatch))


async def _store_down_scenario(tmp_path: Path, monkeypatch) -> None:
    mgr = JobManager(tmp_path)
    await mgr.initialize()
    umo = scope("umo-store-down")
    runs: list[int] = []
    flaky = _FlakyCommit(monkeypatch)

    loop = asyncio.get_running_loop()
    unhandled: list[dict] = []
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))

    async def work(progress):
        runs.append(1)
        await progress("storing", 1, 1)
        flaky.armed = True  # every following commit fails
        flaky.remaining = 10_000
        return {"written": True}

    result = await mgr.submit(umo, "store-down", "op", {}, "kb:1", work)
    assert result["status"] == STATUS_PARTIAL
    assert result["error"]["code"] == "storage_failed"

    # The uncertain outcome stays queryable from process memory.
    fetched = await mgr.get(umo, result["job_id"])
    assert fetched["status"] == STATUS_PARTIAL
    assert fetched["error"]["code"] == "storage_failed"

    async def must_not_run(progress):
        runs.append(2)
        return {}

    again = await mgr.submit(umo, "store-down", "op", {}, "kb:1", must_not_run)
    assert again["job_id"] == result["job_id"]
    assert runs == [1]

    # A key the store cannot confirm as unused must never start work.
    blocked = await mgr.submit(umo, "fresh-key", "op", {}, "kb:1", must_not_run)
    assert blocked["status"] == STATUS_FAILED
    assert blocked["error"]["code"] == "unavailable"
    assert blocked["job_id"] is None
    assert runs == [1]

    await mgr.close()
    leftovers = [
        t for t in asyncio.all_tasks() if (t.get_name() or "").startswith("kb-job:")
    ]
    assert leftovers == []
    assert unhandled == []

    # Restart recovery: the row was never durably finalized, so it comes back
    # as interrupted (the work is not replayed).
    flaky.armed = False
    rebuilt = JobManager(tmp_path)
    await rebuilt.initialize()
    recovered = await rebuilt.get(umo, result["job_id"])
    assert recovered["status"] == STATUS_INTERRUPTED
    assert recovered["error"]["code"] == "interrupted"
    await rebuilt.close()


def test_persistent_store_failure_is_queryable_and_never_orphans_tasks(
    tmp_path, monkeypatch
):
    asyncio.run(_store_down_scenario(tmp_path, monkeypatch))


# ---------------------------------------------------------------------------
# Memory cleanup and lifecycle concurrency
# ---------------------------------------------------------------------------


async def _cleanup_scenario(tmp_path: Path) -> None:
    mgr = JobManager(tmp_path)
    await mgr.initialize()

    async def work(progress):
        return {"ok": True}

    first = await mgr.submit(scope("umo-clean-a"), "r1", "op", {}, "kb:clean", work)
    second = await mgr.submit(scope("umo-clean-b"), "r2", "op", {}, "kb:clean", work)
    assert first["status"] == STATUS_SUCCEEDED
    assert second["status"] == STATUS_SUCCEEDED

    for _ in range(100):
        if not mgr._events and "kb:clean" not in mgr._kb_locks:
            break
        await asyncio.sleep(0.01)
    # Finished jobs release their events; the lock entry is dropped once no
    # task holds or waits for it.
    assert mgr._events == {}
    assert "kb:clean" not in mgr._kb_locks
    await mgr.close()


def test_finished_jobs_release_events_and_idle_locks(tmp_path):
    asyncio.run(_cleanup_scenario(tmp_path))


async def _lifecycle_scenario(tmp_path: Path) -> None:
    mgr = JobManager(tmp_path)
    await asyncio.gather(mgr.initialize(), mgr.initialize(), mgr.initialize())
    assert mgr._initialized

    umo = scope("umo-lifecycle")
    started = asyncio.Event()
    release = asyncio.Event()

    async def long_work(progress):
        started.set()
        await release.wait()
        return {"never": True}

    waiter = asyncio.create_task(
        mgr.submit(umo, "life", "op", {}, "kb:1", long_work, wait_seconds=5)
    )
    await asyncio.wait_for(started.wait(), 2)
    getter = asyncio.create_task(mgr.get(umo, "unknown"))
    await asyncio.gather(mgr.close(), mgr.close())

    result = await asyncio.wait_for(waiter, 3)
    if result["status"] == STATUS_INTERRUPTED:
        assert result["error"]["code"] == "interrupted"
    else:
        assert result["status"] == STATUS_FAILED
        assert result["error"]["code"] == "unavailable"

    got = await getter
    assert got["status"] == STATUS_FAILED
    assert got["error"]["code"] in ("unavailable", "job_not_found")

    with pytest.raises(KBError) as excinfo:
        await mgr.initialize()
    assert excinfo.value.code == "unavailable"
    assert mgr._db is None

    # initialize racing close is safe in either order and never leaves a
    # freshly opened connection behind.
    other = JobManager(tmp_path / "race")
    results = await asyncio.gather(
        other.initialize(), other.close(), return_exceptions=True
    )
    for item in results:
        if isinstance(item, BaseException):
            assert isinstance(item, KBError)
            assert item.code == "unavailable"
    assert other._db is None


def test_initialize_and_close_are_concurrency_safe(tmp_path):
    asyncio.run(_lifecycle_scenario(tmp_path))
