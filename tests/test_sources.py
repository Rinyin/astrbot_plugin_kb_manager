# ruff: noqa: E402, I001
"""Tests for :mod:`sources` (package import, no plugin-dir sys.path entry).

The plugin is imported as ``data.plugins.astrbot_plugin_kb_manager.sources``
matching the framework's real module root, and ``astrbot.api`` comes from the
upstream checkout with a temporary ``ASTRBOT_ROOT``.

The network boundary is exercised with local servers and an injected public
resolver; no test reaches the public internet. The trusted-attachment session
uses the framework's default resolver so real ``localhost`` file services are
covered end to end.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import os
import socket
import sys
import tempfile
from pathlib import Path
from unittest import mock

import aiohttp
import pytest
from aiohttp import web

_WORKSPACE = Path(__file__).resolve().parents[4]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))
_ASTRBOT_SRC = _WORKSPACE / "upstream" / "AstrBot"
if _ASTRBOT_SRC.is_dir() and str(_ASTRBOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ASTRBOT_SRC))
if "ASTRBOT_ROOT" not in os.environ:
    os.environ["ASTRBOT_ROOT"] = tempfile.mkdtemp(prefix="kb-manager-b-test-")

from data.plugins.astrbot_plugin_kb_manager import sources  # noqa: E402
from data.plugins.astrbot_plugin_kb_manager.common import (  # noqa: E402
    CODE_ATTACHMENT_EXPIRED,
    CODE_ATTACHMENT_NOT_FOUND,
    CODE_BACKEND_UNAVAILABLE,
    CODE_INVALID_ARGUMENT,
    CODE_PAYLOAD_TOO_LARGE,
    CODE_UNSUPPORTED_SOURCE,
    CODE_URL_FETCH_FAILED,
    KBError,
)

HOST = "public.test"
SCOPE_A = '["umo","alice"]'
SCOPE_B = '["umo","bob"]'
_HTML_PAGE = (
    "<html><head><title>Topic Page</title></head><body>"
    "<nav>NAVLINK</nav>"
    "<script>ALERTXYZ()</script>"
    "<article><h1>Topic Page</h1>"
    "<p>The first paragraph explains the subject in depth.</p>"
    "<p>The second paragraph adds supporting detail and context.</p>"
    "<p>A third paragraph concludes the discussion clearly.</p>"
    "</article></body></html>"
)


# ---------------------------------------------------------------------------
# Fakes and helpers
# ---------------------------------------------------------------------------


class _TestResolver(aiohttp.abc.AbstractResolver):
    """Resolver for tests.

    Mapped hosts are answered from the mapping; by default they bypass the
    public-IP policy (they stand in for reachable public hosts), while hosts
    outside the mapping are resolved for real and passed through the real
    policy. ``filter_mapped=True`` applies the policy to the mapping too.
    """

    def __init__(self, mapping: dict[str, object], *, filter_mapped: bool = False):
        self._mapping = {
            key: value if isinstance(value, list) else [value]
            for key, value in mapping.items()
        }
        self._filter_mapped = filter_mapped
        self._delegate = aiohttp.resolver.ThreadedResolver()

    async def resolve(
        self, host: str, port: int = 0, family: int = socket.AF_UNSPEC
    ) -> list[dict]:
        targets = self._mapping.get(host)
        if targets is None:
            infos = await self._delegate.resolve(host, port, family)
            return sources._public_addrinfos(infos)
        loop = asyncio.get_running_loop()
        entries: list[dict] = []
        for target in targets:  # type: ignore[union-attr]
            resolved = await loop.getaddrinfo(
                target, port, family=family, type=socket.SOCK_STREAM
            )
            for fam, _type, proto, _canon, sockaddr in resolved:
                entries.append(
                    {
                        "hostname": host,
                        "host": sockaddr[0],
                        "port": sockaddr[1],
                        "family": fam,
                        "proto": proto,
                        "flags": 0,
                    }
                )
        if self._filter_mapped:
            return sources._public_addrinfos(entries)
        return entries

    async def close(self) -> None:
        await self._delegate.close()


class _FakeFile:
    """Minimal stand-in for an AstrBot File message segment."""

    def __init__(
        self,
        *,
        url: str = "",
        path: str = "",
        name: str = "",
        result: str | None = None,
        fail: bool = False,
    ) -> None:
        self.url = url
        self.file_ = path
        self.name = name
        self._result = result
        self._fail = fail
        self.calls: list[bool] = []

    async def get_file(self, allow_return_url: bool = False) -> str:
        self.calls.append(allow_return_url)
        if self._fail:
            raise RuntimeError("get_file must not be called here")
        if self._result is not None:
            return self._result
        return self.url or self.file_


class _BrokenSession:
    closed = False

    async def close(self) -> None:
        raise RuntimeError("session close boom")


class _BrokenResolver:
    async def close(self) -> None:
        raise RuntimeError("resolver close boom")


def _run(coro):
    return asyncio.run(coro)


async def _start(app: web.Application):
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    return runner, port


async def _start_raw_server(body: bytes):
    """Serve one fixed response without a Content-Type header."""

    async def handle(reader, writer):
        with contextlib.suppress(Exception):
            await reader.read(65536)
            writer.write(
                b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: "
                + str(len(body)).encode()
                + b"\r\n\r\n"
            )
            writer.write(body)
            await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, port


def _patch_public_resolver(
    monkeypatch, mapping: dict[str, object], *, filter_mapped: bool = False
):
    monkeypatch.setattr(
        sources,
        "_build_resolver",
        lambda: _TestResolver(mapping, filter_mapped=filter_mapped),
    )


# ---------------------------------------------------------------------------
# IP policy / DNS resolver
# ---------------------------------------------------------------------------


def test_is_public_ip_policy():
    assert sources._is_public_ip(ipaddress.ip_address("8.8.8.8"))
    assert sources._is_public_ip(ipaddress.ip_address("93.184.216.34"))
    assert sources._is_public_ip(ipaddress.ip_address("2606:4700::1111"))
    for blocked in (
        "127.0.0.1",
        "10.0.0.1",
        "192.168.1.1",
        "172.16.0.1",
        "169.254.1.1",
        "100.64.0.1",
        "224.0.0.1",
        "0.0.0.0",
        "::1",
        "fc00::1",
        "fe80::1",
        "::ffff:127.0.0.1",
    ):
        assert not sources._is_public_ip(ipaddress.ip_address(blocked)), blocked


def test_public_addrinfos_rejects_mixed_and_private_answers():
    public = [{"hostname": "example.test", "host": "93.184.216.34", "port": 80}]
    assert len(sources._public_addrinfos(public)) == 1
    mixed = [
        {"hostname": "example.test", "host": "93.184.216.34", "port": 80},
        {"hostname": "example.test", "host": "10.0.0.1", "port": 80},
    ]
    with pytest.raises(OSError):
        sources._public_addrinfos(mixed)
    with pytest.raises(OSError):
        sources._public_addrinfos(
            [{"hostname": "example.test", "host": "127.0.0.1", "port": 80}]
        )
    with pytest.raises(OSError):
        sources._public_addrinfos([{"hostname": "example.test", "port": 80}])


# ---------------------------------------------------------------------------
# Attachments: remembering, scope, TTL
# ---------------------------------------------------------------------------


def test_remember_and_list_do_not_download(tmp_path):
    async def scenario():
        manager = sources.SourceManager(tmp_path)
        await manager.initialize()
        try:
            comp = _FakeFile(
                url=f"http://{HOST}:1/report.pdf", name="report.pdf", fail=True
            )
            data = await manager.remember_attachments(SCOPE_A, [comp])
            assert set(data) == {"attachments"}
            assert len(data["attachments"]) == 1
            info = data["attachments"][0]
            assert info["attachment_id"].startswith("att_")
            assert info["filename"] == "report.pdf"
            assert info["media_type"] == "application/pdf"
            listed = manager.list_attachments(SCOPE_A)
            assert [item["attachment_id"] for item in listed] == [info["attachment_id"]]
            assert comp.calls == []
            assert (tmp_path / "attachments").is_dir()
        finally:
            await manager.close()

    _run(scenario())


def test_remember_requires_file_protocol_objects(tmp_path):
    async def scenario():
        manager = sources.SourceManager(tmp_path)
        await manager.initialize()
        try:
            valid = _FakeFile(url=f"http://{HOST}:1/ok.pdf", name="ok.pdf")
            data = await manager.remember_attachments(
                SCOPE_A,
                [
                    "http://evil.test/arbitrary.bin",  # plain string
                    "/etc/passwd",  # plain string
                    object(),  # no get_file
                    valid,
                ],
            )
            assert [item["filename"] for item in data["attachments"]] == ["ok.pdf"]
            assert len(manager.list_attachments(SCOPE_A)) == 1
        finally:
            await manager.close()

    _run(scenario())


def test_remember_long_filename_keeps_extension(tmp_path):
    async def scenario():
        manager = sources.SourceManager(tmp_path)
        await manager.initialize()
        try:
            long_name = "x" * 300 + ".pdf"
            comp = _FakeFile(url=f"http://{HOST}:1/doc.pdf", name=long_name)
            info = (await manager.remember_attachments(SCOPE_A, [comp]))["attachments"][
                0
            ]
            assert info["filename"].endswith(".pdf")
            assert len(info["filename"]) <= 200
            assert info["media_type"] == "application/pdf"
        finally:
            await manager.close()

    _run(scenario())


def test_remember_refreshes_ttl_with_stable_id(tmp_path, monkeypatch):
    async def scenario():
        clock = [1_000_000_000_000]
        monkeypatch.setattr(sources, "_now_ms", lambda: clock[0])
        manager = sources.SourceManager(tmp_path, attachment_ttl=1800)
        await manager.initialize()
        try:
            comp = _FakeFile(url=f"http://{HOST}:1/a.pdf", name="a.pdf")
            first = (await manager.remember_attachments(SCOPE_A, [comp]))[
                "attachments"
            ][0]
            clock[0] += 600_000
            second = (await manager.remember_attachments(SCOPE_A, [comp]))[
                "attachments"
            ][0]
            assert first["attachment_id"] == second["attachment_id"]
            assert second["expires_at"] > first["expires_at"]
            assert len(manager.list_attachments(SCOPE_A)) == 1
        finally:
            await manager.close()

    _run(scenario())


def test_scope_isolation(tmp_path):
    async def scenario():
        manager = sources.SourceManager(tmp_path)
        await manager.initialize()
        try:
            comp = _FakeFile(url=f"http://{HOST}:1/a.pdf", name="a.pdf")
            info = (await manager.remember_attachments(SCOPE_A, [comp]))["attachments"][
                0
            ]
            assert manager.list_attachments(SCOPE_B) == []
            with pytest.raises(KBError) as excinfo:
                await manager.load_attachment(SCOPE_B, info["attachment_id"])
            assert excinfo.value.code == CODE_ATTACHMENT_NOT_FOUND
        finally:
            await manager.close()

    _run(scenario())


def test_ttl_expiry(tmp_path, monkeypatch):
    async def scenario():
        clock = [1_000_000_000_000]
        monkeypatch.setattr(sources, "_now_ms", lambda: clock[0])
        manager = sources.SourceManager(tmp_path, attachment_ttl=1800)
        await manager.initialize()
        try:
            payload = tmp_path / "note.txt"
            payload.write_bytes(b"hello")
            comp = _FakeFile(path=str(payload), name="note.txt")
            info = (await manager.remember_attachments(SCOPE_A, [comp]))["attachments"][
                0
            ]
            clock[0] += 1799_000
            assert len(manager.list_attachments(SCOPE_A)) == 1
            clock[0] += 2_000
            assert manager.list_attachments(SCOPE_A) == []
            with pytest.raises(KBError) as excinfo:
                await manager.load_attachment(SCOPE_A, info["attachment_id"])
            assert excinfo.value.code == CODE_ATTACHMENT_EXPIRED
        finally:
            await manager.close()

    _run(scenario())


def test_remember_rejects_bad_input(tmp_path):
    async def scenario():
        manager = sources.SourceManager(tmp_path)
        await manager.initialize()
        try:
            with pytest.raises(KBError) as excinfo:
                await manager.remember_attachments("", [])
            assert excinfo.value.code == CODE_INVALID_ARGUMENT
            with pytest.raises(KBError) as excinfo:
                await manager.remember_attachments(SCOPE_A, "not-a-list")
            assert excinfo.value.code == CODE_INVALID_ARGUMENT
            assert await manager.remember_attachments(SCOPE_A, [object()]) == {
                "attachments": []
            }
        finally:
            await manager.close()

    _run(scenario())


# ---------------------------------------------------------------------------
# Attachments: loading
# ---------------------------------------------------------------------------


def test_load_local_attachment_bounded(tmp_path):
    async def scenario():
        manager = sources.SourceManager(tmp_path)
        await manager.initialize()
        try:
            payload = tmp_path / "Report.PDF"
            payload.write_bytes(b"%PDF-1.4 payload")
            comp = _FakeFile(name="Report.PDF", result=str(payload))
            comp.file_ = str(payload)
            info = (await manager.remember_attachments(SCOPE_A, [comp]))["attachments"][
                0
            ]
            assert info["filename"] == "Report.PDF"
            document = await manager.load_attachment(SCOPE_A, info["attachment_id"])
            assert document.content == b"%PDF-1.4 payload"
            assert document.source == "attachment"
            assert document.filename == "Report.PDF"
            assert comp.calls == [True]
        finally:
            await manager.close()

    _run(scenario())


def test_load_local_attachment_too_large(tmp_path):
    async def scenario():
        manager = sources.SourceManager(tmp_path, max_file_bytes=10)
        await manager.initialize()
        try:
            payload = tmp_path / "big.bin"
            payload.write_bytes(b"x" * 11)
            comp = _FakeFile(path=str(payload), name="big.bin")
            info = (await manager.remember_attachments(SCOPE_A, [comp]))["attachments"][
                0
            ]
            with pytest.raises(KBError) as excinfo:
                await manager.load_attachment(SCOPE_A, info["attachment_id"])
            assert excinfo.value.code == CODE_PAYLOAD_TOO_LARGE
        finally:
            await manager.close()

    _run(scenario())


def test_attachment_localhost_allowed_public_blocked(tmp_path):
    """Real localhost attachment works; the same URL stays blocked for web."""

    async def scenario():
        body = b"%PDF-1.4 local file service"
        app = web.Application()

        async def handler(request):
            return web.Response(body=body, content_type="application/pdf")

        app.router.add_get("/doc.pdf", handler)
        runner, port = await _start(app)
        manager = sources.SourceManager(tmp_path)
        await manager.initialize()
        try:
            url = f"http://localhost:{port}/doc.pdf"
            comp = _FakeFile(url=url, name="doc.pdf")
            info = (await manager.remember_attachments(SCOPE_A, [comp]))["attachments"][
                0
            ]
            document = await manager.load_attachment(SCOPE_A, info["attachment_id"])
            assert document.content == body
            # The public session resolves localhost for real and rejects it.
            with pytest.raises(KBError) as excinfo:
                await manager.load_url(url)
            assert excinfo.value.code == CODE_URL_FETCH_FAILED
        finally:
            await manager.close()
            await runner.cleanup()

    _run(scenario())


def test_attachment_and_public_sessions_are_isolated(tmp_path, monkeypatch):
    """Concurrent internal and public fetches cannot widen each other's policy."""

    async def scenario():
        app = web.Application()

        async def slow_pdf(request):
            await asyncio.sleep(0.15)
            return web.Response(body=b"%PDF-1.4 slow", content_type="application/pdf")

        async def page(request):
            return web.Response(text=_HTML_PAGE, content_type="text/html")

        app.router.add_get("/slow.pdf", slow_pdf)
        app.router.add_get("/page", page)
        runner, port = await _start(app)
        _patch_public_resolver(monkeypatch, {HOST: ["127.0.0.1"]})
        manager = sources.SourceManager(tmp_path)
        await manager.initialize()
        try:
            comp = _FakeFile(url=f"http://localhost:{port}/slow.pdf", name="slow.pdf")
            info = (await manager.remember_attachments(SCOPE_A, [comp]))["attachments"][
                0
            ]
            results = await asyncio.gather(
                manager.load_attachment(SCOPE_A, info["attachment_id"]),
                manager.load_url(f"http://localhost:{port}/page"),
                manager.load_url(f"http://{HOST}:{port}/page"),
                return_exceptions=True,
            )
            internal_doc, blocked, public_doc = results
            assert not isinstance(internal_doc, BaseException)
            assert internal_doc.content == b"%PDF-1.4 slow"
            assert isinstance(blocked, KBError)
            assert blocked.code == CODE_URL_FETCH_FAILED
            assert not isinstance(public_doc, BaseException)
            assert public_doc.filename.endswith(".md")
        finally:
            await manager.close()
            await runner.cleanup()

    _run(scenario())


# ---------------------------------------------------------------------------
# URL loading: validation
# ---------------------------------------------------------------------------


def test_load_url_rejects_bad_syntax(tmp_path):
    async def scenario():
        manager = sources.SourceManager(tmp_path)
        await manager.initialize()
        try:
            with pytest.raises(KBError) as excinfo:
                await manager.load_url("")
            assert excinfo.value.code == CODE_INVALID_ARGUMENT
            for url in (
                "ftp://public.test/x",
                "http://user:secret@public.test/x",
                "http://public.test:99999/x",
                "http://[::1",
                "http:///missing-host",
                "http://127.0.0.1/x",
                "http://[::1]/x",
                "http://10.0.0.1/x",
                "http://169.254.169.254/latest/meta-data",
            ):
                with pytest.raises(KBError) as excinfo:
                    await manager.load_url(url)
                assert excinfo.value.code == CODE_UNSUPPORTED_SOURCE, url
        finally:
            await manager.close()

    _run(scenario())


def test_load_url_rejects_control_characters(tmp_path):
    async def scenario():
        manager = sources.SourceManager(tmp_path)
        await manager.initialize()
        try:
            for url in (
                "\x00http://public.test/x",
                "http://public.test/x\x0b",
                "http://public.test/\x00bad",
            ):
                with pytest.raises(KBError) as excinfo:
                    await manager.load_url(url)
                assert excinfo.value.code == CODE_UNSUPPORTED_SOURCE, repr(url)
        finally:
            await manager.close()

    _run(scenario())


# ---------------------------------------------------------------------------
# URL loading: network boundary
# ---------------------------------------------------------------------------


def test_load_url_blocks_mixed_dns_answer(tmp_path, monkeypatch):
    async def scenario():
        _patch_public_resolver(
            monkeypatch,
            {"mixed.test": ["93.184.216.34", "10.0.0.1"]},
            filter_mapped=True,
        )
        manager = sources.SourceManager(tmp_path)
        await manager.initialize()
        try:
            with pytest.raises(KBError) as excinfo:
                await manager.load_url("http://mixed.test/x")
            assert excinfo.value.code == CODE_URL_FETCH_FAILED
        finally:
            await manager.close()

    _run(scenario())


def test_load_url_rejects_redirect_to_internal_ip(tmp_path, monkeypatch):
    async def scenario():
        app = web.Application()
        secret = "http://127.0.0.1:9/secret"

        async def redirect(request):
            raise web.HTTPFound(secret)

        app.router.add_get("/redirect", redirect)
        runner, port = await _start(app)
        _patch_public_resolver(monkeypatch, {HOST: ["127.0.0.1"]})
        manager = sources.SourceManager(tmp_path)
        await manager.initialize()
        try:
            with pytest.raises(KBError) as excinfo:
                await manager.load_url(f"http://{HOST}:{port}/redirect")
            assert excinfo.value.code == CODE_UNSUPPORTED_SOURCE
        finally:
            await manager.close()
            await runner.cleanup()

    _run(scenario())


def test_load_url_rejects_too_many_redirects(tmp_path, monkeypatch):
    async def scenario():
        app = web.Application()
        state = {"hits": 0}

        async def loop(request):
            state["hits"] += 1
            raise web.HTTPFound("/loop")

        app.router.add_get("/loop", loop)
        runner, port = await _start(app)
        _patch_public_resolver(monkeypatch, {HOST: ["127.0.0.1"]})
        manager = sources.SourceManager(tmp_path)
        await manager.initialize()
        try:
            with pytest.raises(KBError) as excinfo:
                await manager.load_url(f"http://{HOST}:{port}/loop")
            assert excinfo.value.code == CODE_URL_FETCH_FAILED
            assert excinfo.value.details == {"max_redirects": 5}
            assert state["hits"] == 6
        finally:
            await manager.close()
            await runner.cleanup()

    _run(scenario())


def test_load_url_network_failure(tmp_path, monkeypatch):
    async def scenario():
        _patch_public_resolver(monkeypatch, {HOST: ["127.0.0.1"]})
        manager = sources.SourceManager(tmp_path)
        await manager.initialize()
        try:
            with pytest.raises(KBError) as excinfo:
                await manager.load_url(f"http://{HOST}:1/page")
            assert excinfo.value.code == CODE_URL_FETCH_FAILED
        finally:
            await manager.close()

    _run(scenario())


# ---------------------------------------------------------------------------
# URL loading: content extraction and limits
# ---------------------------------------------------------------------------


def test_load_url_rejects_non_text_content_type(tmp_path, monkeypatch):
    async def scenario():
        app = web.Application()

        async def handler(request):
            return web.Response(body=b"%PDF-1.4", content_type="application/pdf")

        app.router.add_get("/file.pdf", handler)
        runner, port = await _start(app)
        _patch_public_resolver(monkeypatch, {HOST: ["127.0.0.1"]})
        manager = sources.SourceManager(tmp_path)
        await manager.initialize()
        try:
            with pytest.raises(KBError) as excinfo:
                await manager.load_url(f"http://{HOST}:{port}/file.pdf")
            assert excinfo.value.code == CODE_UNSUPPORTED_SOURCE
        finally:
            await manager.close()
            await runner.cleanup()

    _run(scenario())


def test_load_url_missing_content_type_html_extracted(tmp_path, monkeypatch):
    async def scenario():
        body = (
            b"<!DOCTYPE html><html><head><title>No Mime</title></head><body>"
            b"<p>Some readable content that is definitely long enough.</p>"
            b"<p>Another paragraph with more supporting words.</p>"
            b"</body></html>"
        )
        server, port = await _start_raw_server(body)
        _patch_public_resolver(monkeypatch, {HOST: ["127.0.0.1"]})
        manager = sources.SourceManager(tmp_path)
        await manager.initialize()
        try:
            document = await manager.load_url(f"http://{HOST}:{port}/page")
            assert document.filename.endswith(".md")
            text = document.content.decode("utf-8")
            assert "readable content" in text
            assert "<html" not in text
        finally:
            await manager.close()
            server.close()
            await server.wait_closed()

    _run(scenario())


def test_load_url_missing_content_type_pdf_rejected(tmp_path, monkeypatch):
    async def scenario():
        server, port = await _start_raw_server(b"%PDF-1.4 fake binary")
        _patch_public_resolver(monkeypatch, {HOST: ["127.0.0.1"]})
        manager = sources.SourceManager(tmp_path)
        await manager.initialize()
        try:
            with pytest.raises(KBError) as excinfo:
                await manager.load_url(f"http://{HOST}:{port}/page")
            assert excinfo.value.code == CODE_UNSUPPORTED_SOURCE
        finally:
            await manager.close()
            server.close()
            await server.wait_closed()

    _run(scenario())


def test_load_url_missing_content_type_zip_rejected(tmp_path, monkeypatch):
    async def scenario():
        server, port = await _start_raw_server(b"PK\x03\x04zip payload")
        _patch_public_resolver(monkeypatch, {HOST: ["127.0.0.1"]})
        manager = sources.SourceManager(tmp_path)
        await manager.initialize()
        try:
            with pytest.raises(KBError) as excinfo:
                await manager.load_url(f"http://{HOST}:{port}/page")
            assert excinfo.value.code == CODE_UNSUPPORTED_SOURCE
        finally:
            await manager.close()
            server.close()
            await server.wait_closed()

    _run(scenario())


def test_load_url_rejects_declared_oversize(tmp_path, monkeypatch):
    async def scenario():
        app = web.Application()

        async def handler(request):
            return web.Response(text="y" * 500, content_type="text/html")

        app.router.add_get("/big", handler)
        runner, port = await _start(app)
        _patch_public_resolver(monkeypatch, {HOST: ["127.0.0.1"]})
        manager = sources.SourceManager(tmp_path, max_url_bytes=100)
        await manager.initialize()
        try:
            with pytest.raises(KBError) as excinfo:
                await manager.load_url(f"http://{HOST}:{port}/big")
            assert excinfo.value.code == CODE_PAYLOAD_TOO_LARGE
        finally:
            await manager.close()
            await runner.cleanup()

    _run(scenario())


def test_load_url_rejects_streamed_oversize(tmp_path, monkeypatch):
    async def scenario():
        app = web.Application()

        async def handler(request):
            response = web.StreamResponse(
                status=200, headers={"Content-Type": "text/html"}
            )
            await response.prepare(request)
            try:
                for _ in range(10):
                    await response.write(b"z" * 64)
                await response.write_eof()
            except (ConnectionResetError, aiohttp.ClientError):
                pass
            return response

        app.router.add_get("/stream", handler)
        runner, port = await _start(app)
        _patch_public_resolver(monkeypatch, {HOST: ["127.0.0.1"]})
        manager = sources.SourceManager(tmp_path, max_url_bytes=100)
        await manager.initialize()
        try:
            with pytest.raises(KBError) as excinfo:
                await manager.load_url(f"http://{HOST}:{port}/stream")
            assert excinfo.value.code == CODE_PAYLOAD_TOO_LARGE
        finally:
            await manager.close()
            await runner.cleanup()

    _run(scenario())


def test_load_url_extracts_article(tmp_path, monkeypatch):
    async def scenario():
        app = web.Application()

        async def handler(request):
            return web.Response(text=_HTML_PAGE, content_type="text/html")

        app.router.add_get("/article", handler)
        runner, port = await _start(app)
        _patch_public_resolver(monkeypatch, {HOST: ["127.0.0.1"]})
        manager = sources.SourceManager(tmp_path)
        await manager.initialize()
        try:
            document = await manager.load_url(f"http://{HOST}:{port}/article")
            assert document.source == "url"
            assert document.filename.endswith(".md")
            text = document.content.decode("utf-8")
            assert "first paragraph" in text
            assert "NAVLINK" not in text
            assert "ALERTXYZ" not in text
        finally:
            await manager.close()
            await runner.cleanup()

    _run(scenario())


def test_load_url_plain_text(tmp_path, monkeypatch):
    async def scenario():
        app = web.Application()

        async def handler(request):
            return web.Response(
                text="line one\n\n\nline two", content_type="text/plain"
            )

        app.router.add_get("/plain.txt", handler)
        runner, port = await _start(app)
        _patch_public_resolver(monkeypatch, {HOST: ["127.0.0.1"]})
        manager = sources.SourceManager(tmp_path)
        await manager.initialize()
        try:
            document = await manager.load_url(f"http://{HOST}:{port}/plain.txt")
            assert document.filename == "plain.txt"
            text = document.content.decode("utf-8")
            assert "line one" in text
            assert "line two" in text
            assert "\n\n\n" not in text
        finally:
            await manager.close()
            await runner.cleanup()

    _run(scenario())


def test_load_url_rejects_page_without_content(tmp_path, monkeypatch):
    async def scenario():
        app = web.Application()

        async def handler(request):
            return web.Response(
                text="<html><head><title>Empty</title></head><body>   </body></html>",
                content_type="text/html",
            )

        app.router.add_get("/empty", handler)
        runner, port = await _start(app)
        _patch_public_resolver(monkeypatch, {HOST: ["127.0.0.1"]})
        manager = sources.SourceManager(tmp_path)
        await manager.initialize()
        try:
            with pytest.raises(KBError) as excinfo:
                await manager.load_url(f"http://{HOST}:{port}/empty")
            assert excinfo.value.code == CODE_URL_FETCH_FAILED
        finally:
            await manager.close()
            await runner.cleanup()

    _run(scenario())


def test_load_url_checks_extracted_size(tmp_path, monkeypatch):
    async def scenario():
        app = web.Application()

        async def handler(request):
            return web.Response(text="short body", content_type="text/plain")

        app.router.add_get("/page.txt", handler)
        runner, port = await _start(app)
        _patch_public_resolver(monkeypatch, {HOST: ["127.0.0.1"]})
        monkeypatch.setattr(
            sources, "_render_document", lambda title, body: b"x" * 1000
        )
        manager = sources.SourceManager(tmp_path, max_url_bytes=100)
        await manager.initialize()
        try:
            with pytest.raises(KBError) as excinfo:
                await manager.load_url(f"http://{HOST}:{port}/page.txt")
            assert excinfo.value.code == CODE_PAYLOAD_TOO_LARGE
        finally:
            await manager.close()
            await runner.cleanup()

    _run(scenario())


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_close_releases_resources_and_is_idempotent(tmp_path):
    async def scenario():
        manager = sources.SourceManager(tmp_path)
        await manager.initialize()
        comp = _FakeFile(url=f"http://{HOST}:1/a.pdf", name="a.pdf")
        await manager.remember_attachments(SCOPE_A, [comp])
        public_session = await manager._ensure_session(internal=False)
        internal_session = await manager._ensure_session(internal=True)
        assert not public_session.closed
        assert not internal_session.closed
        await manager.close()
        assert public_session.closed
        assert internal_session.closed
        assert manager._public_session is None
        assert manager._internal_session is None
        assert manager._public_resolver is None
        assert manager._cleanup_task is None
        assert manager.list_attachments(SCOPE_A) == []
        await manager.close()
        with pytest.raises(KBError) as excinfo:
            await manager.initialize()
        assert excinfo.value.code == CODE_BACKEND_UNAVAILABLE
        with pytest.raises(KBError) as excinfo:
            await manager.load_url("http://public.test/x")
        assert excinfo.value.code == CODE_BACKEND_UNAVAILABLE
        with pytest.raises(KBError) as excinfo:
            await manager._ensure_session(internal=False)
        assert excinfo.value.code == CODE_BACKEND_UNAVAILABLE
        assert manager._public_session is None

    _run(scenario())


def test_close_observes_resource_failures(tmp_path, monkeypatch):
    async def scenario():
        manager = sources.SourceManager(tmp_path)
        fake_logger = mock.Mock()
        monkeypatch.setattr(sources, "logger", fake_logger)
        manager._public_session = _BrokenSession()
        manager._public_resolver = _BrokenResolver()
        await manager.close()
        assert fake_logger.warning.call_count == 2

    _run(scenario())


def test_close_concurrent_with_load_creates_no_new_session(tmp_path, monkeypatch):
    async def scenario():
        app = web.Application()

        async def slow(request):
            await asyncio.sleep(0.2)
            return web.Response(text=_HTML_PAGE, content_type="text/html")

        app.router.add_get("/slow", slow)
        runner, port = await _start(app)
        _patch_public_resolver(monkeypatch, {HOST: ["127.0.0.1"]})
        manager = sources.SourceManager(tmp_path)
        await manager.initialize()
        try:
            await asyncio.gather(
                manager.load_url(f"http://{HOST}:{port}/slow"),
                manager.close(),
                return_exceptions=True,
            )
            assert manager._public_session is None
            assert manager._internal_session is None
            assert manager._cleanup_task is None
            await asyncio.sleep(0)
            assert manager._public_session is None
        finally:
            await runner.cleanup()

    _run(scenario())
