"""SQLite-backed async job manager with idempotent, serialised submissions.

See ``docs/dev/contract.md`` §5 for the frozen contract. Jobs are stored in
``<data_dir>/jobs.sqlite3``; user input is never used in filesystem paths.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import aiosqlite

from .common import (
    CODE_INTERNAL,
    CODE_INTERRUPTED,
    CODE_INVALID_ARGUMENT,
    CODE_JOB_NOT_FOUND,
    CODE_STORAGE_FAILED,
    CODE_UNAVAILABLE,
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_WAIT_SECONDS,
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    STATUS_PARTIAL,
    STATUS_QUEUED,
    STATUS_RUNNING,
    STATUS_SUCCEEDED,
    KBError,
    ProgressCallback,
    WorkFunc,
    decode_scope,
    error_result,
    fingerprint,
    json_dumps,
    make_result,
)

try:  # pragma: no cover - inside AstrBot the framework logger is available
    from astrbot.api import logger
except ImportError:  # pragma: no cover - tests run without AstrBot installed
    import logging

    logger = logging.getLogger(__name__)

_DB_FILENAME = "jobs.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    request_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    payload TEXT NOT NULL,
    lock_key TEXT NOT NULL,
    status TEXT NOT NULL,
    progress TEXT,
    result TEXT,
    error TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    UNIQUE (scope, request_id)
)
"""

_INDEX = "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs (status)"


def _now_ms() -> int:
    """Return the current UTC time as Unix epoch milliseconds."""

    return time.time_ns() // 1_000_000


def _require_str(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise KBError(CODE_INVALID_ARGUMENT, f"{name} must be a non-empty string")
    return value


class JobManager:
    """Run submitted coroutines with idempotency, locking and persistence.

    Args:
        data_dir: Plugin data directory; the SQLite database is created here.
        max_concurrent: Maximum number of jobs executing at the same time.
    """

    def __init__(
        self, data_dir: str | Path, max_concurrent: int = DEFAULT_MAX_CONCURRENT
    ) -> None:
        if (
            not isinstance(max_concurrent, int)
            or isinstance(max_concurrent, bool)
            or max_concurrent < 1
        ):
            raise KBError(
                CODE_INVALID_ARGUMENT, "max_concurrent must be a positive integer"
            )
        self._data_dir = Path(data_dir)
        self._db_path = self._data_dir / _DB_FILENAME
        self._max_concurrent = max_concurrent

        self._db: aiosqlite.Connection | None = None
        self._initialized = False
        self._closing = False
        self._closed = False

        # initialize/close share one lock so a close can never race a fresh
        # database connection into existence after the manager shut down.
        self._lifecycle_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._db_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._kb_locks: dict[str, asyncio.Lock] = {}
        self._kb_lock_refs: dict[str, int] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._events: dict[str, asyncio.Event] = {}
        self._job_scopes: dict[str, str] = {}
        # Jobs whose terminal state could not be written durably; the recorded
        # outcome stays queryable in-process until the process restarts.
        self._terminal_overrides: dict[
            str, tuple[str, str, str | None, str | None]
        ] = {}
        # (scope, request_id) -> job_id for jobs seen during this process, which
        # prevents replaying work when the database is temporarily unavailable.
        self._request_index: dict[tuple[str, str], str] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create the database and recover stale non-terminal jobs."""

        async with self._lifecycle_lock:
            if self._closed:
                raise KBError(CODE_UNAVAILABLE, "job manager is closed")
            if self._initialized:
                return
            self._data_dir.mkdir(parents=True, exist_ok=True)
            db = await aiosqlite.connect(self._db_path)
            try:
                db.row_factory = aiosqlite.Row
                await db.execute(_SCHEMA)
                await db.execute(_INDEX)
                # A restart must never replay in-flight work.
                await db.execute(
                    "UPDATE jobs SET status = ?, error = ?, updated_at = ? "
                    "WHERE status IN (?, ?)",
                    (
                        STATUS_INTERRUPTED,
                        json_dumps(self._interrupted_error().to_dict()),
                        _now_ms(),
                        STATUS_QUEUED,
                        STATUS_RUNNING,
                    ),
                )
                await db.commit()
            except BaseException:
                await db.close()
                raise
            self._db = db
            self._initialized = True

    async def close(self) -> None:
        """Reject new jobs, stop running work and close the connection."""

        async with self._lifecycle_lock:
            if self._closed:
                return
            async with self._state_lock:
                self._closing = True
                tasks = list(self._tasks.values())
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await self._persist_all_interrupted()
            for event in self._events.values():
                event.set()
            self._events.clear()
            self._job_scopes.clear()
            self._tasks.clear()
            self._kb_locks.clear()
            self._kb_lock_refs.clear()
            async with self._db_lock:
                db, self._db = self._db, None
                if db is not None:
                    await db.close()
            self._closed = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit(
        self,
        scope: str,
        request_id: str,
        operation: str,
        payload: Mapping[str, Any],
        lock_key: str,
        work: WorkFunc,
        wait_seconds: float = DEFAULT_WAIT_SECONDS,
    ) -> dict[str, Any]:
        """Submit ``work`` idempotently and optionally wait for its result."""

        decode_scope(scope)
        _require_str("request_id", request_id)
        _require_str("operation", operation)
        _require_str("lock_key", lock_key)
        if not isinstance(payload, Mapping):
            raise KBError(CODE_INVALID_ARGUMENT, "payload must be a mapping")
        if not callable(work):
            raise KBError(CODE_INVALID_ARGUMENT, "work must be callable")
        try:
            wait_seconds = float(wait_seconds)
        except (TypeError, ValueError) as exc:
            raise KBError(
                CODE_INVALID_ARGUMENT, "wait_seconds must be a number"
            ) from exc
        payload_dict = dict(payload)
        try:
            payload_fingerprint = fingerprint(
                {"operation": operation, "payload": payload_dict}
            )
        except KBError as exc:
            raise KBError(
                CODE_INVALID_ARGUMENT,
                "payload must be JSON-serializable",
                details=exc.details,
            ) from exc

        request_key = (scope, request_id)
        async with self._state_lock:
            if self._closing or self._closed:
                return error_result(KBError(CODE_UNAVAILABLE, "job manager is closed"))
            if not self._initialized or self._db is None:
                return error_result(
                    KBError(CODE_UNAVAILABLE, "job manager is not initialized")
                )

            existing: aiosqlite.Row | None = None
            store_error: Exception | None = None
            try:
                existing = await self._select_by_request(scope, request_id)
            except Exception as exc:  # the store may be temporarily unavailable
                store_error = exc
                logger.exception("idempotency lookup failed for request %s", request_id)

            if existing is not None:
                self._request_index[request_key] = existing["job_id"]
                if existing["fingerprint"] != payload_fingerprint:
                    return error_result(
                        KBError(
                            "idempotency_conflict",
                            "request_id was already used with different parameters",
                            details={"existing_job_id": existing["job_id"]},
                        ),
                        job_id=existing["job_id"],
                    )
                job_id = existing["job_id"]
            elif store_error is not None:
                # Never start work when the store cannot confirm this key was
                # not already used; report the known job instead when possible.
                known = self._request_index.get(request_key)
                if known is None:
                    return error_result(
                        KBError(CODE_UNAVAILABLE, "job store temporarily unavailable")
                    )
                job_id = known
            else:
                job_id = uuid.uuid4().hex
                try:
                    await self._insert_job(
                        job_id=job_id,
                        scope=scope,
                        request_id=request_id,
                        operation=operation,
                        payload_fingerprint=payload_fingerprint,
                        payload=payload_dict,
                        lock_key=lock_key,
                    )
                except Exception:
                    logger.exception("failed to persist submitted job")
                    return error_result(
                        KBError(CODE_UNAVAILABLE, "job store temporarily unavailable")
                    )
                self._request_index[request_key] = job_id
                self._events[job_id] = asyncio.Event()
                self._job_scopes[job_id] = scope
                # The job runs in its own task: timing out or cancelling this
                # caller never cancels the submitted work.
                task = asyncio.create_task(
                    self._run_job(job_id, scope, lock_key, operation, work),
                    name=f"kb-job:{job_id}",
                )
                self._tasks[job_id] = task

        await self._wait_for_terminal(job_id, wait_seconds)
        return await self._current_envelope(scope, job_id)

    async def get(self, scope: str, job_id: str) -> dict[str, Any]:
        """Return the result envelope for ``job_id`` within ``scope``."""

        decode_scope(scope)
        _require_str("job_id", job_id)
        if self._closing or self._closed or not self._initialized:
            return error_result(
                KBError(CODE_UNAVAILABLE, "job manager is unavailable"), job_id=job_id
            )
        return await self._current_envelope(scope, job_id)

    # ------------------------------------------------------------------
    # Job execution
    # ------------------------------------------------------------------

    async def _run_job(
        self, job_id: str, scope: str, lock_key: str, operation: str, work: WorkFunc
    ) -> None:
        kb_lock = self._kb_locks.get(lock_key)
        if kb_lock is None:
            kb_lock = asyncio.Lock()
            self._kb_locks[lock_key] = kb_lock
        self._kb_lock_refs[lock_key] = self._kb_lock_refs.get(lock_key, 0) + 1
        try:
            # Acquire the per-key lock *before* a global slot so that waiters
            # for the same knowledge base never occupy concurrency capacity.
            async with kb_lock, self._semaphore:
                await self._execute_job(job_id, scope, operation, work)
        except asyncio.CancelledError:
            await self._finalize_terminal(
                job_id,
                scope,
                STATUS_INTERRUPTED,
                None,
                json_dumps(KBError(CODE_INTERRUPTED, "job was cancelled").to_dict()),
            )
            raise
        except Exception:
            logger.exception("job runner crashed for %s", job_id)
            await self._finalize_terminal(
                job_id,
                scope,
                STATUS_FAILED,
                None,
                json_dumps(
                    KBError(CODE_INTERNAL, "internal job runner failure").to_dict()
                ),
            )
        finally:
            self._release_kb_lock(lock_key, kb_lock)
            self._tasks.pop(job_id, None)

    def _release_kb_lock(self, lock_key: str, kb_lock: asyncio.Lock) -> None:
        refs = self._kb_lock_refs.get(lock_key, 1) - 1
        if refs > 0:
            self._kb_lock_refs[lock_key] = refs
            return
        self._kb_lock_refs.pop(lock_key, None)
        # Drop the lock reference once nobody holds or waits for it.
        if not kb_lock.locked():
            self._kb_locks.pop(lock_key, None)

    async def _execute_job(
        self, job_id: str, scope: str, operation: str, work: WorkFunc
    ) -> None:
        if self._closing or self._closed:
            await self._finalize_terminal(
                job_id,
                scope,
                STATUS_INTERRUPTED,
                None,
                json_dumps(
                    KBError(
                        CODE_INTERRUPTED, "job interrupted before execution"
                    ).to_dict()
                ),
            )
            return
        try:
            await self._persist_status(job_id, STATUS_RUNNING)
        except Exception:
            # Not fatal: the terminal write below carries the real outcome.
            logger.exception("failed to mark job %s as running", job_id)
        try:
            data = await work(self._make_progress_callback(job_id))
        except asyncio.CancelledError:
            raise
        except KBError as exc:
            await self._finalize_kb_error(job_id, scope, exc)
        except Exception:
            # Never echo raw exception text (may contain URLs or content) back.
            logger.exception("job %s failed during %s", job_id, operation)
            await self._finalize_kb_error(
                job_id,
                scope,
                KBError(CODE_INTERNAL, f"internal error during '{operation}'"),
            )
        else:
            await self._finalize_success(job_id, scope, data)

    async def _finalize_success(self, job_id: str, scope: str, data: Any) -> None:
        if not isinstance(data, Mapping):
            await self._finalize_kb_error(
                job_id, scope, KBError(CODE_INTERNAL, "work must return a dict")
            )
            return
        try:
            result = json_dumps(dict(data))
        except KBError as exc:
            await self._finalize_kb_error(
                job_id,
                scope,
                KBError(
                    CODE_INTERNAL,
                    "work result is not JSON-serializable",
                    details=exc.details,
                ),
            )
            return
        await self._finalize_terminal(job_id, scope, STATUS_SUCCEEDED, result, None)

    async def _finalize_kb_error(self, job_id: str, scope: str, exc: KBError) -> None:
        status = STATUS_PARTIAL if exc.partial else STATUS_FAILED
        await self._finalize_terminal(
            job_id, scope, status, None, json_dumps(exc.to_dict())
        )

    async def _finalize_terminal(
        self,
        job_id: str,
        scope: str,
        status: str,
        result: str | None,
        error: str | None,
    ) -> None:
        """Persist a terminal state, or record an in-memory uncertain outcome.

        Waiters are only woken once a durable terminal state exists or the
        outcome is explicitly kept in process memory; a failed commit is rolled
        back so it can never be committed later by an unrelated transaction.
        """

        try:
            await self._execute(
                "UPDATE jobs SET status = ?, result = ?, error = ?, updated_at = ? "
                "WHERE job_id = ?",
                (status, result, error, _now_ms(), job_id),
            )
            self._wake(job_id)
            return
        except Exception:
            logger.exception("failed to persist terminal state for job %s", job_id)

        storage_error = KBError(
            CODE_STORAGE_FAILED,
            "job finished but its result could not be stored; outcome is uncertain",
            details={"stage": "persist_result"},
            partial=True,
        )
        fallback_status = STATUS_PARTIAL
        fallback_error = json_dumps(storage_error.to_dict())
        try:
            await self._execute(
                "UPDATE jobs SET status = ?, result = NULL, error = ?, updated_at = ? "
                "WHERE job_id = ?",
                (fallback_status, fallback_error, _now_ms(), job_id),
            )
        except Exception:
            logger.exception("failed to persist uncertainty for job %s", job_id)
            self._terminal_overrides[job_id] = (
                scope,
                fallback_status,
                None,
                fallback_error,
            )
        self._wake(job_id)

    def _wake(self, job_id: str) -> None:
        """Wake waiters of a job whose terminal outcome is durably known."""

        event = self._events.pop(job_id, None)
        self._job_scopes.pop(job_id, None)
        if event is not None:
            event.set()

    def _make_progress_callback(self, job_id: str) -> ProgressCallback:
        async def report(stage: str, current: int, total: int) -> None:
            if self._closing or self._closed or self._db is None:
                return
            try:
                await self._persist_progress(
                    job_id,
                    {"stage": str(stage), "current": int(current), "total": int(total)},
                )
            except Exception:
                logger.warning(
                    "progress update failed for job %s", job_id, exc_info=True
                )

        return report

    async def _wait_for_terminal(self, job_id: str, wait_seconds: float) -> None:
        if wait_seconds <= 0:
            return
        event = self._events.get(job_id)
        if event is None:
            return
        try:
            await asyncio.wait_for(event.wait(), timeout=wait_seconds)
        except asyncio.TimeoutError:
            return

    # ------------------------------------------------------------------
    # Persistence helpers (short, serialised transactions)
    # ------------------------------------------------------------------

    async def _execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        """Run one short write transaction, rolling back on any failure."""

        async with self._db_lock:
            db = self._db
            if db is None:
                raise KBError(CODE_UNAVAILABLE, "job manager is unavailable")
            try:
                await db.execute(sql, params)
                await db.commit()
            except BaseException:
                await self._rollback_quietly(db)
                raise

    @staticmethod
    async def _rollback_quietly(db: aiosqlite.Connection) -> None:
        try:
            await db.rollback()
        except Exception:
            logger.exception("rollback failed")

    async def _fetchone(
        self, sql: str, params: tuple[Any, ...] = ()
    ) -> aiosqlite.Row | None:
        async with self._db_lock:
            db = self._db
            if db is None:
                raise KBError(CODE_UNAVAILABLE, "job manager is unavailable")
            cursor = await db.execute(sql, params)
            try:
                return await cursor.fetchone()
            finally:
                await cursor.close()

    async def _insert_job(
        self,
        job_id: str,
        scope: str,
        request_id: str,
        operation: str,
        payload_fingerprint: str,
        payload: dict[str, Any],
        lock_key: str,
    ) -> None:
        now = _now_ms()
        await self._execute(
            "INSERT INTO jobs (job_id, scope, request_id, operation, fingerprint, "
            "payload, lock_key, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                job_id,
                scope,
                request_id,
                operation,
                payload_fingerprint,
                json_dumps(payload),
                lock_key,
                STATUS_QUEUED,
                now,
                now,
            ),
        )

    async def _persist_status(self, job_id: str, status: str) -> None:
        await self._execute(
            "UPDATE jobs SET status = ?, updated_at = ? WHERE job_id = ?",
            (status, _now_ms(), job_id),
        )

    async def _persist_progress(self, job_id: str, progress: dict[str, Any]) -> None:
        await self._execute(
            "UPDATE jobs SET progress = ?, updated_at = ? WHERE job_id = ?",
            (json_dumps(progress), _now_ms(), job_id),
        )

    async def _persist_all_interrupted(self) -> None:
        error_json = json_dumps(self._interrupted_error().to_dict())
        try:
            await self._execute(
                "UPDATE jobs SET status = ?, result = NULL, error = ?, updated_at = ? "
                "WHERE status IN (?, ?)",
                (
                    STATUS_INTERRUPTED,
                    error_json,
                    _now_ms(),
                    STATUS_QUEUED,
                    STATUS_RUNNING,
                ),
            )
        except Exception:
            logger.exception("failed to persist interrupted jobs during close")
            for job_id in list(self._events):
                scope = self._job_scopes.get(job_id)
                if scope is not None:
                    self._terminal_overrides.setdefault(
                        job_id, (scope, STATUS_INTERRUPTED, None, error_json)
                    )

    async def _select_by_request(
        self, scope: str, request_id: str
    ) -> aiosqlite.Row | None:
        return await self._fetchone(
            "SELECT * FROM jobs WHERE scope = ? AND request_id = ?",
            (scope, request_id),
        )

    async def _select_job(self, scope: str, job_id: str) -> aiosqlite.Row | None:
        return await self._fetchone(
            "SELECT * FROM jobs WHERE scope = ? AND job_id = ?",
            (scope, job_id),
        )

    async def _current_envelope(self, scope: str, job_id: str) -> dict[str, Any]:
        override = self._terminal_overrides.get(job_id)
        if override is not None:
            override_scope, status, result, error = override
            if override_scope != scope:
                return error_result(
                    KBError(CODE_JOB_NOT_FOUND, "job not found"), job_id=job_id
                )
            data = json.loads(result) if result is not None else None
            error_payload = json.loads(error) if error is not None else None
            return make_result(status, job_id=job_id, data=data, error=error_payload)
        try:
            row = await self._select_job(scope, job_id)
        except KBError as exc:
            return error_result(exc, job_id=job_id)
        except Exception:
            logger.exception("failed to read job %s", job_id)
            return error_result(
                KBError(CODE_UNAVAILABLE, "job store temporarily unavailable"),
                job_id=job_id,
            )
        if row is None:
            return error_result(
                KBError(CODE_JOB_NOT_FOUND, "job not found"), job_id=job_id
            )
        return self._row_envelope(row)

    @staticmethod
    def _row_envelope(row: aiosqlite.Row) -> dict[str, Any]:
        status = row["status"]
        job_id = row["job_id"]
        if status in (STATUS_QUEUED, STATUS_RUNNING):
            progress = (
                json.loads(row["progress"]) if row["progress"] is not None else None
            )
            return make_result(
                status,
                job_id=job_id,
                data={
                    "operation": row["operation"],
                    "progress": progress,
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                },
            )
        data = json.loads(row["result"]) if row["result"] is not None else None
        error = json.loads(row["error"]) if row["error"] is not None else None
        return make_result(status, job_id=job_id, data=data, error=error)

    @staticmethod
    def _interrupted_error() -> KBError:
        return KBError(CODE_INTERRUPTED, "job interrupted by restart or shutdown")
