"""Trusted source access for KB Manager: attachment cache and restricted URLs.

The :class:`SourceManager` owns two responsibilities:

1. Remember ``File`` message segments handed in by the trusted entry layer,
   keyed by the encoded request scope, and later materialise them as
   :class:`SourceDocument` objects. Attachment records live in memory only;
   only downloaded bytes may touch ``data_dir``.
2. Fetch public ``http(s)`` URLs under strict limits (public IP only, manual
   redirect handling, size caps) and turn static HTML / plain text into a
   UTF-8 ``.md`` / ``.txt`` document.

Two independent HTTP sessions are used so the trusted-attachment exception
can never widen the public-web policy, even under concurrency:

* the *public* session always resolves through :class:`_PublicResolver`,
  which validates DNS answers at connection time and pins the checked IPs;
* the *internal* session is only reachable through
  :meth:`SourceManager.load_attachment` for URLs taken from a File segment,
  so adapter-internal file services (for example ``http://localhost``) work.

All failures raise :class:`common.KBError`; no function logs or returns a raw
URL (credentials, query signatures) or document body.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import ipaddress
import mimetypes
import os
import re
import socket
import ssl
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlsplit

import aiohttp
import certifi
from astrbot.api import logger

from .common import (
    CODE_ATTACHMENT_EXPIRED,
    CODE_ATTACHMENT_NOT_FOUND,
    CODE_BACKEND_UNAVAILABLE,
    CODE_INVALID_ARGUMENT,
    CODE_PAYLOAD_TOO_LARGE,
    CODE_UNSUPPORTED_SOURCE,
    CODE_URL_FETCH_FAILED,
    DEFAULT_ATTACHMENT_TTL,
    DEFAULT_HTTP_TIMEOUT,
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_MAX_URL_BYTES,
    SOURCE_ATTACHMENT,
    SOURCE_URL,
    KBError,
    SourceDocument,
)

__all__ = ["SourceManager"]

MAX_REDIRECTS = 5
_DOWNLOAD_CHUNK = 65536
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_HTML_MIMES = frozenset({"text/html", "application/xhtml+xml"})
_TEXT_MIMES = frozenset({"text/html", "application/xhtml+xml", "text/plain"})
_HTML_EXTS = frozenset({".html", ".htm", ".xhtml"})
_PLAIN_EXTS = frozenset({".txt", ".md", ".markdown", ".text"})
_BINARY_EXTS = frozenset(
    {
        ".7z",
        ".bin",
        ".bz2",
        ".doc",
        ".docx",
        ".epub",
        ".exe",
        ".gif",
        ".gz",
        ".ico",
        ".jpeg",
        ".jpg",
        ".mp3",
        ".mp4",
        ".pdf",
        ".png",
        ".ppt",
        ".pptx",
        ".rar",
        ".tar",
        ".wav",
        ".webm",
        ".webp",
        ".xls",
        ".xlsx",
        ".xz",
        ".zip",
    }
)
_BINARY_MAGIC = (
    b"%pdf-",
    b"pk\x03\x04",
    b"pk\x05\x06",
    b"\x1f\x8b",
    b"rar!",
    b"7z\xbc\xaf\x27\x1c",
    b"\x89png",
    b"gif87a",
    b"gif89a",
    b"\xff\xd8\xff",
)
_HTML_MARKERS = (b"<!doctype html", b"<html", b"<head", b"<body")
_INVALID_FILENAME = re.compile(r'[\x00-\x1f\x7f<>:"/\\|?*]')
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_MAX_FILENAME = 200
_CHARSET_IN_HEAD = re.compile(rb"""charset\s*=\s*["']?([A-Za-z0-9_\-]+)""")


# ---------------------------------------------------------------------------
# IP policy and DNS resolver
# ---------------------------------------------------------------------------


def _is_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True only for globally routable unicast addresses."""
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return _is_public_ip(ip.ipv4_mapped)
    return ip.is_global


def _public_addrinfos(infos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter resolved addresses, rejecting every non-public address.

    Args:
        infos: ``aiohttp`` resolver entries (``ResolveResult`` shape, whose
            ``host`` key holds the resolved IP; ``hostname`` is the original
            name and is used only as a fallback for older aiohttp).

    Returns:
        Copies of the original entries whose resolved address is a public IP.

    Raises:
        OSError: If any entry is not publicly routable, or if the list has
            no usable IP address at all. A mixed answer is rejected entirely
            so a DNS answer cannot smuggle an internal address.
    """
    verified: list[dict[str, Any]] = []
    for info in infos:
        address = info.get("host") or info.get("hostname")
        if not address:
            continue
        try:
            ip = ipaddress.ip_address(str(address))
        except ValueError:
            continue
        if not _is_public_ip(ip):
            raise OSError("resolved address is not publicly routable")
        verified.append(dict(info))
    if not verified:
        raise OSError("host did not resolve to a usable public address")
    return verified


class _PublicResolver(aiohttp.abc.AbstractResolver):
    """Resolver that validates and pins the addresses used by the connector."""

    def __init__(self) -> None:
        self._delegate = aiohttp.resolver.ThreadedResolver()

    async def resolve(
        self,
        host: str,
        port: int = 0,
        family: int = socket.AF_UNSPEC,
    ) -> list[dict[str, Any]]:
        infos = await self._delegate.resolve(host, port, family)
        return _public_addrinfos(infos)

    async def close(self) -> None:
        await self._delegate.close()


def _build_resolver() -> aiohttp.abc.AbstractResolver:
    """Create the resolver used by the public session (test seam)."""
    return _PublicResolver()


# ---------------------------------------------------------------------------
# URL and filename helpers
# ---------------------------------------------------------------------------


def _is_http_url(value: str) -> bool:
    lower = value.lower()
    return lower.startswith("http://") or lower.startswith("https://")


def _validate_url(url: str, *, allow_internal: bool) -> str:
    """Validate syntax and literal hosts before any network access.

    Control characters are rejected before any trimming so a control prefix or
    suffix cannot be smuggled through ``strip()``. Malformed URLs (bad IPv6
    brackets, invalid ports, unparsable input) are normalised to ``KBError``.

    Args:
        url: Candidate URL.
        allow_internal: ``True`` only for URLs obtained from a trusted File
            segment. Never set from tool input.

    Returns:
        The trimmed URL.

    Raises:
        KBError: ``unsupported_source`` for non-http(s), credentials, control
            characters, missing host, invalid port, or a literal non-public IP.
    """
    if not isinstance(url, str) or not url:
        raise KBError(CODE_UNSUPPORTED_SOURCE, "url must be a non-empty string")
    if _CONTROL_CHARS.search(url):
        raise KBError(
            CODE_UNSUPPORTED_SOURCE, "url must not contain control characters"
        )
    url = url.strip()
    if not url:
        raise KBError(CODE_UNSUPPORTED_SOURCE, "url must be a non-empty string")
    if not _is_http_url(url):
        raise KBError(CODE_UNSUPPORTED_SOURCE, "only http/https urls are supported")
    try:
        parts = urlsplit(url)
    except ValueError as exc:
        raise KBError(CODE_UNSUPPORTED_SOURCE, "url could not be parsed") from exc
    if parts.scheme.lower() not in {"http", "https"}:
        raise KBError(CODE_UNSUPPORTED_SOURCE, "only http/https urls are supported")
    if parts.username is not None or parts.password is not None:
        raise KBError(CODE_UNSUPPORTED_SOURCE, "url must not embed credentials")
    try:
        host = parts.hostname
        port = parts.port
    except ValueError as exc:
        raise KBError(
            CODE_UNSUPPORTED_SOURCE, "url has an invalid host or port"
        ) from exc
    if not host:
        raise KBError(CODE_UNSUPPORTED_SOURCE, "url must include a host")
    if port is not None and not 0 < port < 65536:
        raise KBError(CODE_UNSUPPORTED_SOURCE, "url has an invalid port")
    if not allow_internal:
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            ip = None
        if ip is not None and not _is_public_ip(ip):
            raise KBError(CODE_UNSUPPORTED_SOURCE, "url host is not publicly routable")
    return url


def _sanitize_filename(name: str) -> str:
    """Strip path components and unsafe characters, keeping the extension."""
    name = str(name).replace("\\", "/").rsplit("/", 1)[-1]
    name = _INVALID_FILENAME.sub("_", name)
    name = name.strip().strip(".")
    if not name:
        return ""
    if len(name) <= _MAX_FILENAME:
        return name
    suffix = Path(name).suffix
    if suffix and len(suffix) < _MAX_FILENAME // 2:
        return f"{name[: _MAX_FILENAME - len(suffix)]}{suffix}"
    return name[:_MAX_FILENAME]


def _has_extension(name: str) -> bool:
    return bool(Path(name).suffix)


def _guess_media_type(filename: str) -> str:
    mime, _ = mimetypes.guess_type(filename)
    return mime or "application/octet-stream"


def _url_filename(url: str) -> str:
    path = urlsplit(url).path
    candidate = unquote(path.rsplit("/", 1)[-1]) if path else ""
    return _sanitize_filename(candidate) or "download"


def _filename_from_source(name: str, source: str) -> str:
    candidate = _sanitize_filename(name) if name else ""
    if not candidate and source:
        if _is_http_url(source):
            candidate = _url_filename(source)
        else:
            candidate = _sanitize_filename(os.path.basename(source))
    if not candidate:
        candidate = "attachment"
    if not _has_extension(candidate):
        suffix = Path(unquote(source)).suffix
        if suffix and suffix.lower() in _BINARY_EXTS | _PLAIN_EXTS | _HTML_EXTS:
            candidate += suffix
    return candidate


def _mime_of(content_type: str) -> str:
    return (content_type or "").split(";", 1)[0].strip().lower()


def _looks_binary(content: bytes) -> bool:
    """Detect obvious binary payloads without a Content-Type header."""
    head = content[:2048]
    if b"\x00" in head:
        return True
    stripped = head.lstrip().lower()
    return stripped.startswith(_BINARY_MAGIC)


def _looks_like_html_bytes(content: bytes) -> bool:
    head = content[:4096].lower()
    return any(marker in head for marker in _HTML_MARKERS)


def _reject_binary_hint(content_type: str, url: str) -> None:
    """Reject an explicit non-text type or a known binary URL suffix early."""
    mime = _mime_of(content_type)
    if mime:
        if mime not in _TEXT_MIMES:
            raise KBError(
                CODE_UNSUPPORTED_SOURCE,
                "only static HTML or plain text pages are supported",
            )
        return
    suffix = Path(urlsplit(url).path).suffix.lower()
    if suffix and suffix in _BINARY_EXTS:
        raise KBError(
            CODE_UNSUPPORTED_SOURCE,
            "only static HTML or plain text pages are supported",
        )


def _classify_web_content(content: bytes, content_type: str, url: str) -> str:
    """Classify a fetched page as ``"html"`` or ``"text"``.

    When the server omits ``Content-Type`` the payload is sniffed so an
    extensionless HTML page is still extracted and an obvious binary document
    (PDF/ZIP/...) is rejected instead of stored as plain text.

    Raises:
        KBError: ``unsupported_source`` for binary or unsupported payloads.
    """
    mime = _mime_of(content_type)
    if mime:
        if mime not in _TEXT_MIMES:
            raise KBError(
                CODE_UNSUPPORTED_SOURCE,
                "only static HTML or plain text pages are supported",
            )
        return "html" if mime in _HTML_MIMES else "text"
    if _looks_binary(content):
        raise KBError(
            CODE_UNSUPPORTED_SOURCE,
            "only static HTML or plain text pages are supported",
        )
    if _looks_like_html_bytes(content):
        return "html"
    if Path(urlsplit(url).path).suffix.lower() in _HTML_EXTS:
        return "html"
    return "text"


def _decode_text(raw: bytes, content_type: str) -> str:
    """Decode response bytes with header charset, meta sniff, then chardet."""
    charset = ""
    for part in (content_type or "").split(";")[1:]:
        key, _, value = part.partition("=")
        if key.strip().lower() == "charset":
            charset = value.strip().strip("\"'")
            break
    if charset:
        with contextlib.suppress(LookupError, UnicodeDecodeError):
            return raw.decode(charset)
    head = raw[:4096]
    match = _CHARSET_IN_HEAD.search(head)
    if match:
        with contextlib.suppress(LookupError, UnicodeDecodeError):
            return raw.decode(match.group(1).decode("ascii", "ignore"))
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:
        import chardet

        detected = chardet.detect(raw)
        encoding = detected.get("encoding")
        if encoding and (detected.get("confidence") or 0) >= 0.5:
            return raw.decode(encoding, errors="replace")
    except Exception:  # pragma: no cover - optional dependency
        pass
    return raw.decode("utf-8", errors="replace")


def _extract_article(html_text: str) -> tuple[str, str]:
    """Extract title and main text from static HTML (runs in a thread)."""
    from bs4 import BeautifulSoup
    from readability import Document

    try:
        article = Document(html_text)
        title = (article.short_title() or "").strip()
        summary = article.summary(html_partial=True)
    except Exception:
        return "", ""
    soup = BeautifulSoup(summary, "lxml")
    for tag in soup(
        ["script", "style", "noscript", "template", "nav", "aside", "footer"]
    ):
        tag.decompose()
    body = soup.get_text("\n", strip=True)
    return title, re.sub(r"\n{3,}", "\n\n", body).strip()


def _normalize_plain(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _render_document(title: str, body: str) -> bytes:
    parts: list[str] = []
    if title:
        parts.append(f"# {title}")
        parts.append("")
    parts.append(body)
    return ("\n".join(parts).strip() + "\n").encode("utf-8")


def _now_ms() -> int:
    return int(time.time() * 1000)


# ---------------------------------------------------------------------------
# SourceManager
# ---------------------------------------------------------------------------


class _Attachment:
    """In-memory attachment record; never serialised with its URL."""

    __slots__ = ("attachment_id", "component", "filename", "media_type", "size")

    def __init__(
        self,
        attachment_id: str,
        component: Any,
        filename: str,
        media_type: str,
        size: int | None,
    ) -> None:
        self.attachment_id = attachment_id
        self.component = component
        self.filename = filename
        self.media_type = media_type
        self.size = size


class SourceManager:
    """Scope-isolated attachment cache plus restricted public URL loading."""

    def __init__(
        self,
        data_dir: str | os.PathLike[str],
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        max_url_bytes: int = DEFAULT_MAX_URL_BYTES,
        attachment_ttl: int = DEFAULT_ATTACHMENT_TTL,
        http_timeout: float = DEFAULT_HTTP_TIMEOUT,
    ) -> None:
        """Initialise the manager.

        Args:
            data_dir: Plugin data directory (only place this module may write).
            max_file_bytes: Attachment size cap in bytes (default 20 MiB).
            max_url_bytes: Web page size cap in bytes (default 5 MiB).
            attachment_ttl: Attachment lifetime in seconds (default 1800).
            http_timeout: Total timeout per fetch in seconds (default 30).
        """
        for name, value in (
            ("max_file_bytes", max_file_bytes),
            ("max_url_bytes", max_url_bytes),
            ("attachment_ttl", attachment_ttl),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise KBError(CODE_INVALID_ARGUMENT, f"{name} must be a number")
            if value <= 0:
                raise KBError(CODE_INVALID_ARGUMENT, f"{name} must be > 0")
        if http_timeout is None or http_timeout <= 0:
            raise KBError(CODE_INVALID_ARGUMENT, "http_timeout must be > 0")
        self.data_dir = Path(data_dir)
        self.max_file_bytes = int(max_file_bytes)
        self.max_url_bytes = int(max_url_bytes)
        self.attachment_ttl = float(attachment_ttl)
        self.http_timeout = float(http_timeout)
        self._attachments: dict[str, dict[str, _Attachment]] = {}
        self._expiry: dict[str, dict[str, int]] = {}
        self._public_session: aiohttp.ClientSession | None = None
        self._internal_session: aiohttp.ClientSession | None = None
        self._public_resolver: aiohttp.abc.AbstractResolver | None = None
        self._cleanup_task: asyncio.Task[None] | None = None
        self._initialized = False
        self._closed = False
        self._init_lock = asyncio.Lock()

    # -- lifecycle ---------------------------------------------------------

    async def initialize(self) -> None:
        """Create the cache directory and start the TTL sweeper (idempotent)."""
        async with self._init_lock:
            if self._closed:
                raise KBError(CODE_BACKEND_UNAVAILABLE, "source manager is closed")
            if self._initialized:
                return
            cache_root = self.data_dir / "attachments"
            await asyncio.to_thread(cache_root.mkdir, parents=True, exist_ok=True)
            self._purge_expired(_now_ms())
            self._cleanup_task = asyncio.create_task(
                self._cleanup_loop(),
                name="kb-manager-attachment-cleanup",
            )
            self._initialized = True

    async def close(self) -> None:
        """Cancel the sweeper and close both sessions and the resolver.

        Idempotent. Every cleanup step is attempted independently; failures
        are logged so a stuck resource stays observable instead of vanishing.
        """
        async with self._init_lock:
            if self._closed:
                return
            self._closed = True
            task, self._cleanup_task = self._cleanup_task, None
            sessions: list[aiohttp.ClientSession] = []
            for name in ("_public_session", "_internal_session"):
                session = getattr(self, name)
                setattr(self, name, None)
                if session is not None:
                    sessions.append(session)
            resolver, self._public_resolver = self._public_resolver, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        for session in sessions:
            try:
                await session.close()
            except Exception:
                logger.warning(
                    "failed to close HTTP session during shutdown", exc_info=True
                )
        if resolver is not None:
            try:
                await resolver.close()
            except Exception:
                logger.warning(
                    "failed to close DNS resolver during shutdown", exc_info=True
                )
        self._attachments.clear()
        self._expiry.clear()

    async def _ensure_ready(self) -> None:
        if self._closed:
            raise KBError(CODE_BACKEND_UNAVAILABLE, "source manager is closed")
        if not self._initialized:
            await self.initialize()

    async def _cleanup_loop(self) -> None:
        interval = max(1.0, min(self.attachment_ttl, 60.0))
        try:
            while True:
                await asyncio.sleep(interval)
                try:
                    self._purge_expired(_now_ms())
                except Exception:  # pragma: no cover - defensive
                    logger.warning("attachment cleanup failed", exc_info=True)
        except asyncio.CancelledError:
            return

    def _purge_expired(self, now_ms: int) -> None:
        for scope, entries in list(self._attachments.items()):
            expiry = self._expiry.get(scope, {})
            for attachment_id, expires_at in list(expiry.items()):
                if now_ms >= expires_at:
                    entries.pop(attachment_id, None)
                    expiry.pop(attachment_id, None)
            if not entries:
                self._attachments.pop(scope, None)
                self._expiry.pop(scope, None)

    # -- attachments -------------------------------------------------------

    async def remember_attachments(
        self, scope: str, components: list[Any]
    ) -> dict[str, Any]:
        """Record trusted File segments for ``scope`` and return their info.

        Only File-protocol objects (with a callable ``get_file``) are
        accepted; plain strings and arbitrary path-like values are ignored so
        the tool layer cannot turn this into a general file reader.

        Args:
            scope: Encoded scope produced by the entry layer.
            components: Real AstrBot File message segments.

        Returns:
            ``{"attachments": [AttachmentInfo, ...]}`` where each info has at
            least ``attachment_id`` and ``filename``.

        Raises:
            KBError: ``invalid_argument`` for a bad scope or non-list input.
        """
        if not isinstance(scope, str) or not scope:
            raise KBError(CODE_INVALID_ARGUMENT, "scope must be a non-empty string")
        if components is None:
            components = []
        if isinstance(components, (str, bytes)) or not hasattr(components, "__iter__"):
            raise KBError(CODE_INVALID_ARGUMENT, "components must be a list")
        await self._ensure_ready()
        now_ms = _now_ms()
        expires_at = now_ms + int(self.attachment_ttl * 1000)
        results: list[dict[str, Any]] = []
        for component in components:
            hint = self._source_hint(component)
            if hint is None:
                continue
            url, source = hint
            name = getattr(component, "name", "") or ""
            filename = _filename_from_source(str(name), source)
            size = (
                self._stat_size(source) if source and not _is_http_url(source) else None
            )
            media_type = _guess_media_type(filename)
            attachment_id = self._make_attachment_id(scope, url, source, filename)
            record = _Attachment(attachment_id, component, filename, media_type, size)
            self._attachments.setdefault(scope, {})[attachment_id] = record
            self._expiry.setdefault(scope, {})[attachment_id] = expires_at
            results.append(
                {
                    "attachment_id": attachment_id,
                    "filename": filename,
                    "size": size,
                    "media_type": media_type,
                    "expires_at": expires_at,
                }
            )
        logger.info("remembered %d attachment(s)", len(results))
        return {"attachments": results}

    def list_attachments(self, scope: str) -> list[dict[str, Any]]:
        """Return non-expired AttachmentInfo dicts for ``scope`` (no fetch)."""
        if not isinstance(scope, str) or not scope:
            raise KBError(CODE_INVALID_ARGUMENT, "scope must be a non-empty string")
        now_ms = _now_ms()
        entries = self._attachments.get(scope, {})
        expiry = self._expiry.get(scope, {})
        result = []
        for attachment_id, record in entries.items():
            expires_at = expiry.get(attachment_id)
            if expires_at is None or now_ms >= expires_at:
                continue
            result.append(
                {
                    "attachment_id": attachment_id,
                    "filename": record.filename,
                    "size": record.size,
                    "media_type": record.media_type,
                    "expires_at": expires_at,
                }
            )
        return result

    async def load_attachment(self, scope: str, attachment_id: str) -> SourceDocument:
        """Materialise one remembered attachment as a SourceDocument.

        Args:
            scope: Encoded scope that recorded the attachment.
            attachment_id: Opaque id returned by :meth:`remember_attachments`.

        Returns:
            A ``SourceDocument`` with ``source="attachment"``.

        Raises:
            KBError: ``invalid_argument``, ``attachment_not_found``,
                ``attachment_expired``, ``payload_too_large`` or
                ``unsupported_source``.
        """
        if not isinstance(scope, str) or not scope:
            raise KBError(CODE_INVALID_ARGUMENT, "scope must be a non-empty string")
        if not isinstance(attachment_id, str) or not attachment_id:
            raise KBError(
                CODE_INVALID_ARGUMENT, "attachment_id must be a non-empty string"
            )
        await self._ensure_ready()
        now_ms = _now_ms()
        expires_at = self._expiry.get(scope, {}).get(attachment_id)
        record = self._attachments.get(scope, {}).get(attachment_id)
        if expires_at is not None and now_ms >= expires_at:
            self._attachments.get(scope, {}).pop(attachment_id, None)
            self._expiry.get(scope, {}).pop(attachment_id, None)
            raise KBError(CODE_ATTACHMENT_EXPIRED, "attachment has expired")
        if record is None:
            raise KBError(CODE_ATTACHMENT_NOT_FOUND, "attachment was not found")
        source = await self._resolve_component(record.component)
        if _is_http_url(source):
            content, _ = await self._fetch(
                source,
                self.max_file_bytes,
                internal=True,
                require_text=False,
            )
        else:
            content = await self._read_local(source, self.max_file_bytes)
        return SourceDocument(
            filename=record.filename,
            content=content,
            source=SOURCE_ATTACHMENT,
        )

    # -- public URL --------------------------------------------------------

    async def load_url(self, url: str) -> SourceDocument:
        """Fetch a public web page and return its extracted text document.

        Args:
            url: Public ``http(s)`` URL.

        Returns:
            A ``SourceDocument`` with ``source="url"`` and a ``.md`` (HTML) or
            ``.txt`` (plain text) filename.

        Raises:
            KBError: ``invalid_argument``, ``unsupported_source``,
                ``payload_too_large`` or ``url_fetch_failed``.
        """
        if not isinstance(url, str) or not url.strip():
            raise KBError(CODE_INVALID_ARGUMENT, "url must be a non-empty string")
        if _CONTROL_CHARS.search(url):
            raise KBError(
                CODE_UNSUPPORTED_SOURCE, "url must not contain control characters"
            )
        url = url.strip()
        await self._ensure_ready()
        content, content_type = await self._fetch(
            url,
            self.max_url_bytes,
            internal=False,
            require_text=True,
        )
        kind = _classify_web_content(content, content_type, url)
        text = _decode_text(content, content_type)
        if kind == "html":
            title, body = await asyncio.to_thread(_extract_article, text)
            extension = ".md"
        else:
            title, body = "", _normalize_plain(text)
            extension = ".txt"
        if not body:
            raise KBError(
                CODE_URL_FETCH_FAILED,
                "no readable content found in the page",
            )
        stem = _url_filename(url)
        if _has_extension(stem):
            stem = stem.rsplit(".", 1)[0] or "download"
        filename = f"{stem}{extension}"
        rendered = _render_document(title, body)
        if len(rendered) > self.max_url_bytes:
            raise KBError(
                CODE_PAYLOAD_TOO_LARGE,
                "extracted content exceeds the size limit",
                details={"limit": self.max_url_bytes},
            )
        return SourceDocument(
            filename=filename,
            content=rendered,
            source=SOURCE_URL,
        )

    # -- internals ---------------------------------------------------------

    def _source_hint(self, component: Any) -> tuple[str, str] | None:
        """Return ``(url_hint, source_hint)`` for a trusted File component.

        Returns ``None`` when the object is not a File-protocol component; a
        File always exposes a callable ``get_file``. The hints are only used
        for metadata and the stable attachment id -- loading always goes
        through ``get_file(allow_return_url=True)``.
        """
        if not callable(getattr(component, "get_file", None)):
            return None
        url = getattr(component, "url", "") or ""
        if not isinstance(url, str):
            url = ""
        url = url.strip()
        local = ""
        for attr in ("file_", "path"):
            value = getattr(component, attr, "") or ""
            if isinstance(value, str) and value.strip():
                local = value.strip()
                break
        if url:
            return url, url
        if local:
            return "", local
        name = getattr(component, "name", "") or ""
        if isinstance(name, str) and name.strip():
            return "", ""
        return None

    @staticmethod
    def _stat_size(source: str) -> int | None:
        try:
            return os.path.getsize(source)
        except (OSError, ValueError):
            return None

    @staticmethod
    def _make_attachment_id(scope: str, url: str, source: str, filename: str) -> str:
        digest = hashlib.sha256(
            f"{scope}\x00{url}\x00{source}\x00{filename}".encode()
        ).hexdigest()
        return f"att_{digest[:32]}"

    async def _resolve_component(self, component: Any) -> str:
        """Resolve a File segment into a trusted path or URL via ``get_file``."""
        getter = getattr(component, "get_file", None)
        if not callable(getter):
            raise KBError(
                CODE_ATTACHMENT_NOT_FOUND,
                "attachment does not expose get_file()",
            )
        try:
            value = await getter(allow_return_url=True)
        except TypeError:
            value = await getter()
        except Exception as exc:
            raise KBError(
                CODE_ATTACHMENT_NOT_FOUND,
                "attachment is no longer resolvable",
                details={"error": type(exc).__name__},
            ) from exc
        if not isinstance(value, str) or not value.strip():
            raise KBError(CODE_ATTACHMENT_NOT_FOUND, "attachment source is unavailable")
        return value.strip()

    async def _read_local(self, path: str, limit: int) -> bytes:
        """Read at most ``limit + 1`` bytes from a trusted local file."""
        if path.startswith("file://"):
            path = unquote(urlsplit(path).path)
            if os.name == "nt" and len(path) > 2 and path[0] == "/" and path[2] == ":":
                path = path[1:]

        def _read() -> bytes:
            with open(path, "rb") as handle:
                return handle.read(limit + 1)

        try:
            data = await asyncio.to_thread(_read)
        except (OSError, ValueError) as exc:
            raise KBError(
                CODE_ATTACHMENT_NOT_FOUND,
                "attachment file is not readable",
                details={"error": type(exc).__name__},
            ) from exc
        if len(data) > limit:
            raise KBError(
                CODE_PAYLOAD_TOO_LARGE,
                "attachment exceeds the size limit",
                details={"limit": limit},
            )
        return data

    def _build_session(
        self, resolver: aiohttp.abc.AbstractResolver | None
    ) -> aiohttp.ClientSession:
        context = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(
            resolver=resolver,
            use_dns_cache=False,
            ssl=context,
            limit=8,
        )
        timeout = aiohttp.ClientTimeout(
            total=self.http_timeout,
            connect=min(10.0, self.http_timeout),
        )
        return aiohttp.ClientSession(
            trust_env=False,
            connector=connector,
            timeout=timeout,
        )

    async def _ensure_session(self, *, internal: bool) -> aiohttp.ClientSession:
        """Return the session for ``internal`` mode, never post-close."""
        async with self._init_lock:
            if self._closed:
                raise KBError(CODE_BACKEND_UNAVAILABLE, "source manager is closed")
            if internal:
                if self._internal_session is None:
                    self._internal_session = self._build_session(None)
                return self._internal_session
            if self._public_session is None:
                resolver = _build_resolver()
                self._public_resolver = resolver
                self._public_session = self._build_session(resolver)
            return self._public_session

    async def _fetch(
        self,
        url: str,
        limit: int,
        *,
        internal: bool,
        require_text: bool,
    ) -> tuple[bytes, str]:
        """Download ``url`` with manual redirects and streaming size caps."""
        try:
            return await asyncio.wait_for(
                self._follow_and_read(
                    url,
                    limit,
                    internal=internal,
                    require_text=require_text,
                ),
                timeout=self.http_timeout,
            )
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise KBError(CODE_URL_FETCH_FAILED, "remote fetch timed out") from exc

    async def _follow_and_read(
        self,
        url: str,
        limit: int,
        *,
        internal: bool,
        require_text: bool,
    ) -> tuple[bytes, str]:
        current = _validate_url(url, allow_internal=internal)
        session = await self._ensure_session(internal=internal)
        for _ in range(MAX_REDIRECTS + 1):
            try:
                async with session.get(current, allow_redirects=False) as response:
                    if response.status in _REDIRECT_STATUSES:
                        location = response.headers.get("Location")
                        if not location:
                            raise KBError(
                                CODE_URL_FETCH_FAILED,
                                "redirect response is missing a location",
                            )
                        current = _validate_url(
                            urljoin(current, location), allow_internal=internal
                        )
                        continue
                    if response.status < 200 or response.status >= 300:
                        raise KBError(
                            CODE_URL_FETCH_FAILED,
                            "remote server returned an unexpected status",
                            details={"status": response.status},
                        )
                    content_type = response.headers.get("Content-Type", "")
                    if require_text:
                        _reject_binary_hint(content_type, current)
                    content = await self._read_bounded(response, limit)
                    return content, content_type
            except KBError:
                raise
            except (
                aiohttp.ClientError,
                OSError,
                ValueError,
                RuntimeError,
            ) as exc:
                raise KBError(
                    CODE_URL_FETCH_FAILED,
                    "failed to fetch remote content",
                    details={"error": type(exc).__name__},
                ) from exc
        raise KBError(
            CODE_URL_FETCH_FAILED,
            "too many redirects",
            details={"max_redirects": MAX_REDIRECTS},
        )

    @staticmethod
    async def _read_bounded(response: aiohttp.ClientResponse, limit: int) -> bytes:
        declared = response.headers.get("Content-Length")
        if declared:
            with contextlib.suppress(ValueError):
                if int(declared) > limit:
                    raise KBError(
                        CODE_PAYLOAD_TOO_LARGE,
                        "response exceeds the size limit",
                        details={"limit": limit},
                    )
        chunks: list[bytes] = []
        total = 0
        async for chunk in response.content.iter_chunked(_DOWNLOAD_CHUNK):
            total += len(chunk)
            if total > limit:
                raise KBError(
                    CODE_PAYLOAD_TOO_LARGE,
                    "response exceeds the size limit",
                    details={"limit": limit},
                )
            chunks.append(chunk)
        return b"".join(chunks)
