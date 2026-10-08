"""Shared, dependency-free contract primitives for the KB Manager plugin.

This module is the single source of truth for the data shapes shared by
``backend.py``, ``sources.py``, ``jobs.py`` and the plugin entry point. It
imports only the Python standard library and must never import AstrBot or a
plugin-local module, so it stays cycle-free and unit-testable.

The behaviour behind these primitives is specified in
``docs/dev/contract.md``. Everything public in this module is frozen
contract; change it only together with a contract revision.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

__all__ = [
    "CODE_ATTACHMENT_EXPIRED",
    "CODE_ATTACHMENT_NOT_FOUND",
    "CODE_BACKEND_UNAVAILABLE",
    "CODE_CHUNK_NOT_FOUND",
    "CODE_DOCUMENT_NOT_FOUND",
    "CODE_EMBEDDING_FAILED",
    "CODE_EMBEDDING_PROVIDER_MISSING",
    "CODE_IDEMPOTENCY_CONFLICT",
    "CODE_INTERNAL",
    "CODE_INTERRUPTED",
    "CODE_INVALID_ARGUMENT",
    "CODE_JOB_NOT_FOUND",
    "CODE_KB_NOT_FOUND",
    "CODE_NAME_CONFLICT",
    "CODE_PAYLOAD_TOO_LARGE",
    "CODE_STORAGE_FAILED",
    "CODE_UNAVAILABLE",
    "CODE_UNSUPPORTED_SOURCE",
    "CODE_URL_FETCH_FAILED",
    "DEFAULT_ATTACHMENT_TTL",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_HTTP_TIMEOUT",
    "DEFAULT_LIST_LIMIT",
    "DEFAULT_MAX_CONCURRENT",
    "DEFAULT_MAX_FILE_BYTES",
    "DEFAULT_MAX_URL_BYTES",
    "DEFAULT_TOP_K",
    "DEFAULT_WAIT_SECONDS",
    "JOB_STATES",
    "JobManagerProtocol",
    "KBError",
    "LOCK_GLOBAL",
    "MAX_LIST_LIMIT",
    "NativeKBBackendProtocol",
    "PROGRESS_STAGES",
    "ProgressCallback",
    "SOURCE_ATTACHMENT",
    "SOURCE_FILE",
    "SOURCE_URL",
    "STATUS_FAILED",
    "STATUS_INTERRUPTED",
    "STATUS_PARTIAL",
    "STATUS_QUEUED",
    "STATUS_RUNNING",
    "STATUS_SUCCEEDED",
    "Scope",
    "SourceDocument",
    "SourceManagerProtocol",
    "TERMINAL_STATES",
    "WorkFunc",
    "as_kb_error",
    "decode_scope",
    "encode_scope",
    "ensure_jsonable",
    "error_result",
    "fingerprint",
    "is_terminal",
    "json_dumps",
    "kb_lock_key",
    "make_result",
    "ok_result",
    "request_key",
]

# ---------------------------------------------------------------------------
# Frozen defaults (must mirror the signatures fixed in docs/dev/contract.md)
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 5
DEFAULT_LIST_LIMIT = 20
MAX_LIST_LIMIT = 100
DEFAULT_MAX_FILE_BYTES = 20971520  # 20 MiB
DEFAULT_MAX_URL_BYTES = 5242880  # 5 MiB
DEFAULT_ATTACHMENT_TTL = 1800  # seconds
DEFAULT_HTTP_TIMEOUT = 30  # seconds
DEFAULT_MAX_CONCURRENT = 3
DEFAULT_WAIT_SECONDS = 3

LOCK_GLOBAL = "kb:*"

SOURCE_FILE = "file"
SOURCE_URL = "url"
SOURCE_ATTACHMENT = "attachment"

# ---------------------------------------------------------------------------
# Job states
# ---------------------------------------------------------------------------

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_PARTIAL = "partial"
STATUS_INTERRUPTED = "interrupted"

JOB_STATES = frozenset(
    {
        STATUS_QUEUED,
        STATUS_RUNNING,
        STATUS_SUCCEEDED,
        STATUS_FAILED,
        STATUS_PARTIAL,
        STATUS_INTERRUPTED,
    }
)

TERMINAL_STATES = frozenset(
    {
        STATUS_SUCCEEDED,
        STATUS_FAILED,
        STATUS_PARTIAL,
        STATUS_INTERRUPTED,
    }
)

# Recommended progress stages for long-running work.
PROGRESS_STAGES = ("chunking", "embedding", "storing")

# ---------------------------------------------------------------------------
# Error codes (reserved; implementations may add new codes, never reuse these)
# ---------------------------------------------------------------------------

CODE_INVALID_ARGUMENT = "invalid_argument"
CODE_INTERNAL = "internal"
CODE_INTERRUPTED = "interrupted"
CODE_KB_NOT_FOUND = "kb_not_found"
CODE_DOCUMENT_NOT_FOUND = "document_not_found"
CODE_CHUNK_NOT_FOUND = "chunk_not_found"
CODE_JOB_NOT_FOUND = "job_not_found"
CODE_ATTACHMENT_NOT_FOUND = "attachment_not_found"
CODE_ATTACHMENT_EXPIRED = "attachment_expired"
CODE_IDEMPOTENCY_CONFLICT = "idempotency_conflict"
CODE_PAYLOAD_TOO_LARGE = "payload_too_large"
CODE_STORAGE_FAILED = "storage_failed"
CODE_UNAVAILABLE = "unavailable"
CODE_UNSUPPORTED_SOURCE = "unsupported_source"
CODE_URL_FETCH_FAILED = "url_fetch_failed"
CODE_BACKEND_UNAVAILABLE = "backend_unavailable"
CODE_EMBEDDING_FAILED = "embedding_failed"
CODE_EMBEDDING_PROVIDER_MISSING = "embedding_provider_missing"
CODE_NAME_CONFLICT = "name_conflict"


# ---------------------------------------------------------------------------
# Small validation helpers
# ---------------------------------------------------------------------------


def _require_non_empty_str(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise KBError(
            CODE_INVALID_ARGUMENT,
            f"{name} must be a non-empty string",
            details={"argument": name, "type": type(value).__name__},
        )
    return value


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class KBError(Exception):
    """Canonical error type crossing every module boundary.

    ``partial=True`` means the operation failed after doing some work; when
    converted into a result envelope this maps to status ``partial``.
    """

    def __init__(
        self,
        code: str,
        message: str,
        details: Any = None,
        partial: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details
        self.partial = bool(partial)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": _safe_jsonable(self.details),
            "partial": self.partial,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> KBError:
        return cls(
            code=str(payload.get("code", CODE_INTERNAL)),
            message=str(payload.get("message", "")),
            details=payload.get("details"),
            partial=bool(payload.get("partial", False)),
        )

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


def as_kb_error(exc: BaseException, code: str = CODE_INTERNAL) -> KBError:
    """Coerce an arbitrary exception into a KBError (pass-through if possible)."""

    if isinstance(exc, KBError):
        return exc
    return KBError(
        code,
        str(exc) or exc.__class__.__name__,
        details={"type": exc.__class__.__name__},
    )


# ---------------------------------------------------------------------------
# Source documents
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SourceDocument:
    """Materialised input document handed to backend write operations."""

    filename: str
    content: bytes
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.filename, str) or not self.filename:
            raise KBError(CODE_INVALID_ARGUMENT, "filename must be a non-empty string")
        if not isinstance(self.source, str) or not self.source:
            raise KBError(CODE_INVALID_ARGUMENT, "source must be a non-empty string")
        if not isinstance(self.content, (bytes, bytearray)):
            raise KBError(CODE_INVALID_ARGUMENT, "content must be bytes")
        self.content = bytes(self.content)

    @property
    def size(self) -> int:
        return len(self.content)

    def to_meta(self) -> dict[str, Any]:
        return {"filename": self.filename, "size": self.size, "source": self.source}


# ---------------------------------------------------------------------------
# Scope codec (UMO + sender_id, JSON array; never LLM-provided)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Scope:
    """Decoded request scope: unified message origin plus platform sender id."""

    umo: str
    sender_id: str

    def encode(self) -> str:
        return encode_scope(self.umo, self.sender_id)

    @classmethod
    def decode(cls, raw: str) -> Scope:
        return decode_scope(raw)


def encode_scope(umo: str, sender_id: str) -> str:
    """Encode ``[umo, sender_id]`` as a compact JSON array string."""

    _require_non_empty_str("umo", umo)
    _require_non_empty_str("sender_id", sender_id)
    return json.dumps([umo, sender_id], ensure_ascii=False, separators=(",", ":"))


def decode_scope(raw: str) -> Scope:
    """Decode a scope produced by :func:`encode_scope`."""

    _require_non_empty_str("scope", raw)
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise KBError(
            CODE_INVALID_ARGUMENT,
            "scope is not valid JSON",
            details={"scope": raw},
        ) from exc
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise KBError(
            CODE_INVALID_ARGUMENT,
            "scope must be a JSON array of two non-empty strings",
            details={"scope": raw},
        )
    return Scope(umo=value[0], sender_id=value[1])


# ---------------------------------------------------------------------------
# Idempotency and locking keys
# ---------------------------------------------------------------------------


def request_key(scope: str, request_id: str) -> str:
    """Idempotency key: ``(scope, request_id)`` encoded as a JSON array."""

    _require_non_empty_str("scope", scope)
    _require_non_empty_str("request_id", request_id)
    return json.dumps([scope, request_id], ensure_ascii=False, separators=(",", ":"))


def kb_lock_key(kb_id: str) -> str:
    """Serialisation key for write jobs touching one knowledge base."""

    _require_non_empty_str("kb_id", kb_id)
    return f"kb:{kb_id}"


def fingerprint(payload: Any) -> str:
    """Stable fingerprint used to detect reused request ids with new params."""

    return json_dumps(payload, sort_keys=True)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def ensure_jsonable(value: Any, _path: str = "$") -> Any:
    """Return a strict-JSON copy of ``value`` or raise KBError.

    Mappings become dicts with string keys, sequences become lists, path-like
    values become strings, and bytes / non-finite floats are rejected because
    they cannot travel through the result envelope.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise KBError(
                CODE_INTERNAL,
                f"non-finite float at {_path} is not JSON-serializable",
                details={"path": _path},
            )
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise KBError(
                    CODE_INTERNAL,
                    f"non-string mapping key at {_path}",
                    details={"path": _path, "key": repr(key)},
                )
            result[key] = ensure_jsonable(item, f"{_path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            ensure_jsonable(item, f"{_path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    raise KBError(
        CODE_INTERNAL,
        f"value at {_path} is not JSON-serializable",
        details={"path": _path, "type": type(value).__name__},
    )


def json_dumps(value: Any, ensure_ascii: bool = False, sort_keys: bool = False) -> str:
    """Strict-JSON serializer: rejects bytes, NaN/Inf and non-string keys."""

    try:
        return json.dumps(
            ensure_jsonable(value),
            ensure_ascii=ensure_ascii,
            sort_keys=sort_keys,
            separators=(",", ":"),
            allow_nan=False,
        )
    except KBError:
        raise
    except (TypeError, ValueError) as exc:
        raise KBError(
            CODE_INTERNAL,
            "value is not JSON-serializable",
            details={"error": str(exc)},
        ) from exc


def _safe_jsonable(value: Any) -> Any:
    if value is None:
        return None
    try:
        return ensure_jsonable(value)
    except KBError:
        return repr(value)


# ---------------------------------------------------------------------------
# Result envelope
# ---------------------------------------------------------------------------


def make_result(
    status: str,
    job_id: str | None = None,
    data: Any = None,
    error: KBError | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the unified envelope: ``status, job_id, data, error``."""

    if status not in JOB_STATES:
        raise KBError(
            CODE_INVALID_ARGUMENT,
            f"unknown job status: {status!r}",
            details={"status": status, "allowed": sorted(JOB_STATES)},
        )
    if isinstance(error, KBError):
        error_payload: Any = error.to_dict()
    else:
        error_payload = ensure_jsonable(error) if error is not None else None
    return {
        "status": status,
        "job_id": job_id,
        "data": ensure_jsonable(data) if data is not None else None,
        "error": error_payload,
    }


def ok_result(data: Any = None, job_id: str | None = None) -> dict[str, Any]:
    return make_result(STATUS_SUCCEEDED, job_id=job_id, data=data)


def error_result(
    error: BaseException | KBError,
    job_id: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    kb_error = as_kb_error(error)
    if status is None:
        status = STATUS_PARTIAL if kb_error.partial else STATUS_FAILED
    return make_result(status, job_id=job_id, error=kb_error)


def is_terminal(status: str) -> bool:
    return status in TERMINAL_STATES


# ---------------------------------------------------------------------------
# Callable contracts
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[str, int, int], Awaitable[None]]
WorkFunc = Callable[[ProgressCallback], Awaitable[dict[str, Any]]]


# ---------------------------------------------------------------------------
# Structural protocols (the concrete classes live in backend/sources/jobs)
# ---------------------------------------------------------------------------


class NativeKBBackendProtocol(Protocol):
    """Contract for ``backend.py``'s ``NativeKBBackend``.

    All methods are async and return a JSON-serialisable payload dict that
    becomes the ``data`` field of the result envelope; failures raise
    :class:`KBError`.
    """

    def __init__(
        self, context: Any, default_embedding_provider_id: str = ""
    ) -> None: ...

    async def initialize(self) -> dict[str, Any]: ...

    async def list_kbs(self) -> dict[str, Any]: ...

    async def create_kb(
        self,
        name: str,
        description: str = "",
        embedding_provider_id: str = "",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> dict[str, Any]: ...

    async def update_kb(
        self,
        kb_id: str,
        description: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> dict[str, Any]: ...

    async def delete_kb(self, kb_id: str) -> dict[str, Any]: ...

    async def list_documents(
        self,
        kb_id: str,
        offset: int = 0,
        limit: int = 20,
        search: str = "",
    ) -> dict[str, Any]: ...

    async def read_document(
        self,
        kb_id: str,
        doc_id: str,
        offset: int = 0,
        limit: int = 20,
    ) -> dict[str, Any]: ...

    async def search(
        self,
        query: str,
        kb_ids: list[str],
        top_k: int = 5,
    ) -> dict[str, Any]: ...

    async def add_document(
        self,
        kb_id: str,
        filename: str,
        content: bytes,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]: ...

    async def replace_document(
        self,
        kb_id: str,
        doc_id: str,
        filename: str,
        content: bytes,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]: ...

    async def delete_document(self, kb_id: str, doc_id: str) -> dict[str, Any]: ...

    async def add_chunk(
        self, kb_id: str, doc_id: str, content: str
    ) -> dict[str, Any]: ...

    async def update_chunk(
        self,
        kb_id: str,
        doc_id: str,
        chunk_id: str,
        content: str,
    ) -> dict[str, Any]: ...

    async def delete_chunk(
        self,
        kb_id: str,
        doc_id: str,
        chunk_id: str,
    ) -> dict[str, Any]: ...


class SourceManagerProtocol(Protocol):
    """Contract for ``sources.py``'s ``SourceManager``."""

    def __init__(
        self,
        data_dir: str | os.PathLike[str],
        max_file_bytes: int = 20971520,
        max_url_bytes: int = 5242880,
        attachment_ttl: int = 1800,
        http_timeout: float = 30,
    ) -> None: ...

    async def initialize(self) -> None: ...

    async def remember_attachments(
        self,
        scope: str,
        components: list[Any],
    ) -> dict[str, Any]: ...

    async def load_attachment(
        self, scope: str, attachment_id: str
    ) -> SourceDocument: ...

    async def load_url(self, url: str) -> SourceDocument: ...

    def list_attachments(self, scope: str) -> list[dict[str, Any]]: ...

    async def close(self) -> None: ...


class JobManagerProtocol(Protocol):
    """Contract for ``jobs.py``'s ``JobManager``."""

    def __init__(
        self, data_dir: str | os.PathLike[str], max_concurrent: int = 3
    ) -> None: ...

    async def initialize(self) -> None: ...

    async def submit(
        self,
        scope: str,
        request_id: str,
        operation: str,
        payload: dict[str, Any],
        lock_key: str,
        work: WorkFunc,
        wait_seconds: float = 3,
    ) -> dict[str, Any]: ...

    async def get(self, scope: str, job_id: str) -> dict[str, Any]: ...

    async def close(self) -> None: ...
