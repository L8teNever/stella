"""Direct MCP tools on a Gemini Live owner call, against a fake Streamable HTTP server."""

from __future__ import annotations

import asyncio
import json
import socket
import time
import pytest
import uvicorn
from mcp.server.fastmcp import FastMCP

from stella.gemini_voice import GeminiVoiceSession
from stella.live_mcp import (
    MSG_TIMEOUT,
    LiveMcpPool,
    live_mcp_enabled,
    parse_mcp_servers,
    prefix_tool,
    to_gemini_schema,
)
from stella.store import JobStore
from tests.test_stella import OWNER, FakeProviderWS, ida_settings


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _fake_mcp() -> FastMCP:
    mcp = FastMCP("fake", stateless_http=True)
    mcp.settings.streamable_http_path = "/"
    mcp.settings.transport_security.enable_dns_rebinding_protection = False

    @mcp.tool()
    def echo(text: str) -> str:
        """Gibt den Text zurück."""
        return f"echo:{text}"

    @mcp.tool()
    def geheim() -> str:
        """Darf nicht in der Allowlist stehen."""
        return "geheim"

    @mcp.tool()
    async def langsam() -> str:
        """Dauert absichtlich zu lange."""
        await asyncio.sleep(30)
        return "spaet"

    return mcp


class _BearerApp:
    def __init__(self, app, token: str) -> None:
        self._app = app
        self._token = token.encode()

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            headers = dict(scope.get("headers") or [])
            if headers.get(b"authorization") != b"Bearer " + self._token:
                await send({"type": "http.response.start", "status": 401, "headers": []})
                await send({"type": "http.response.body", "body": b""})
                return
        await self._app(scope, receive, send)


@pytest.fixture
async def fake_mcp_url():
    mcp = _fake_mcp()
    app = _BearerApp(mcp.streamable_http_app(), "s3cret")
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", lifespan="on")
    )
    task = asyncio.create_task(server.serve())
    for _ in range(50):
        if server.started:
            break
        await asyncio.sleep(0.05)
    else:
        server.should_exit = True
        raise RuntimeError("fake MCP server did not start")
    try:
        yield f"http://127.0.0.1:{port}/"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=5)


def _servers(url: str) -> str:
    return json.dumps(
        [{
            "name": "fake",
            "url": url,
            "token_env": "FAKE_MCP_TOKEN",
            "allow": ["echo", "langsam"],
        }]
    )


def test_parse_mcp_servers_allowlist_and_bad_json(caplog):
    specs = parse_mcp_servers(
        '[{"name":"calendar","url":"https://cal.example/mcp","token_env":"CAL_TOKEN",'
        '"allow":["list_events","get_event"]},{"name":"","url":"https://x"}]'
    )
    assert len(specs) == 1
    assert specs[0].name == "calendar"
    assert specs[0].allow == ("list_events", "get_event")
    assert parse_mcp_servers("") == []
    assert parse_mcp_servers("not-json") == []
    assert "reason=config" in caplog.text
    assert prefix_tool("calendar", "list-events") == "calendar__list_events"
    schema = to_gemini_schema(
        {"type": "object", "properties": {"text": {"type": "string", "description": "hi"}}, "required": ["text"]}
    )
    assert schema["type"] == "OBJECT"
    assert schema["properties"]["text"]["type"] == "STRING"
    assert schema["required"] == ["text"]


def test_live_mcp_gate_matches_ida(tmp_path):
    settings = ida_settings(tmp_path, stella_mcp_servers='[{"name":"c","url":"http://127.0.0.1/mcp"}]')
    assert live_mcp_enabled(settings, OWNER, True)
    assert not live_mcp_enabled(settings, "+14155552671", True)
    assert not live_mcp_enabled(settings, OWNER, False)
    assert not live_mcp_enabled(ida_settings(tmp_path, ask_ida_enabled=False), OWNER, True)
    assert not live_mcp_enabled(ida_settings(tmp_path), OWNER, True)


@pytest.mark.asyncio
async def test_pool_calls_fake_server_and_times_out(fake_mcp_url, monkeypatch, caplog):
    monkeypatch.setenv("FAKE_MCP_TOKEN", "s3cret")
    caplog.set_level("INFO", logger="stella.live_mcp")
    pool = LiveMcpPool(
        parse_mcp_servers(_servers(fake_mcp_url)),
        tool_timeout_s=0.4,
        connect_timeout_s=5,
    )
    await pool.open()
    try:
        names = [decl["name"] for decl in pool.declarations()]
        assert names == ["fake__echo", "fake__langsam"]
        echo = pool.declarations()[0]
        assert echo["parameters"]["properties"]["text"]["type"] == "STRING"
        result = await pool.call("fake__echo", {"text": "hallo"})
        assert result == {"result": "echo:hallo"}
        timed = await pool.call("fake__langsam", {})
        assert timed == {"error": MSG_TIMEOUT}
        assert await pool.call("fake__echo", {"text": "danach"}) == {"result": "echo:danach"}
    finally:
        await pool.close()
    text = caplog.text
    assert "tool=fake__echo" in text and "ok=ok" in text
    assert "tool=fake__langsam" in text and "reason=timeout" in text
    assert "s3cret" not in text
    assert "hallo" not in text
    assert "Bearer" not in text

    # A new call opens its own sessions after the previous ones were closed.
    again = LiveMcpPool(parse_mcp_servers(_servers(fake_mcp_url)), tool_timeout_s=2, connect_timeout_s=5)
    await again.open()
    try:
        assert await again.call("fake__echo", {"text": "nochmal"}) == {"result": "echo:nochmal"}
    finally:
        await again.close()


@pytest.mark.asyncio
async def test_missing_token_skips_server_without_leaking_it(fake_mcp_url, caplog):
    caplog.set_level("INFO", logger="stella.live_mcp")
    pool = LiveMcpPool(parse_mcp_servers(_servers(fake_mcp_url)), tool_timeout_s=2)
    await pool.open()
    try:
        assert pool.declarations() == []
    finally:
        await pool.close()
    assert "reason=token" in caplog.text
    assert "s3cret" not in caplog.text


@pytest.mark.asyncio
async def test_owner_session_prefers_direct_tools_without_blocking_audio(
    tmp_path, fake_mcp_url, monkeypatch
):
    monkeypatch.setenv("FAKE_MCP_TOKEN", "s3cret")
    settings = ida_settings(tmp_path, stella_mcp_servers=_servers(fake_mcp_url), stella_mcp_tool_timeout_s=0.4)
    store = JobStore(settings.stella_db_path)
    job = store.create(kind="call", to_number=OWNER, brief="b", allow_ida=True)
    sess = GeminiVoiceSession(
        settings=settings, job=job, gemini_connect=None,
        hangup_cb=lambda: None, transcript_cb=lambda c: None, outcome_cb=lambda t: None,
    )
    await sess._open_live_mcp()
    try:
        payload = sess._setup_payload()
        names = [d["name"] for t in payload["tools"] for d in t["functionDeclarations"]]
        assert names[:3] == ["hang_up", "frag_ida", "fake__echo"]
        assert "fake__geheim" not in names and "geheim" not in names
        text = payload["systemInstruction"]["parts"][0]["text"]
        assert "Direct tools always come before frag_ida" in text
        assert "fake__echo" in text and "frag_ida" in text
        assert "Was steht heute an?" in text
        assert "google__" in text and "never frag_ida" in text
        frag = next(
            d for t in payload["tools"] for d in t["functionDeclarations"] if d["name"] == "frag_ida"
        )
        assert "Nur Fallback" in frag["description"]
        assert "Was steht heute an?" in frag["description"]

        ws = FakeProviderWS()
        sess._gemini_ws = ws
        started = time.monotonic()
        await sess._handle_gemini_event(
            {"toolCall": {"functionCalls": [{"id": "slow-1", "name": "fake__langsam", "args": {}}]}}
        )
        assert time.monotonic() - started < 0.5
        await sess._handle_gemini_event(
            {"serverContent": {"outputTranscription": {"text": "Moment."}}}
        )
        assert sess._assistant_bits == ["Moment."]
        assert any(not task.done() for task in sess._mcp_tasks)

        await sess._handle_gemini_event(
            {"toolCall": {"functionCalls": [
                {"id": "echo-1", "name": "fake__echo", "args": {"text": "hallo"}}
            ]}}
        )
        await asyncio.wait_for(asyncio.gather(*list(sess._mcp_tasks)), timeout=3)
        # The slow tool is still the one that may have finished or still be running;
        # echo must have answered with the matching id.
        sent = [
            item["toolResponse"]["functionResponses"][0]
            for item in ws.sent
            if isinstance(item, dict) and "toolResponse" in item
        ]
        echo = next(item for item in sent if item["id"] == "echo-1")
        assert echo["name"] == "fake__echo"
        assert echo["response"]["result"] == "echo:hallo"
    finally:
        await sess.close()
        assert sess._mcp is None

    other = store.create(kind="call", to_number="+14155552671", brief="b", allow_ida=True)
    other_sess = GeminiVoiceSession(
        settings=settings, job=other, gemini_connect=None,
        hangup_cb=lambda: None, transcript_cb=lambda c: None, outcome_cb=lambda t: None,
    )
    await other_sess._open_live_mcp()
    names = [
        d["name"]
        for t in other_sess._setup_payload()["tools"]
        for d in t["functionDeclarations"]
    ]
    assert "fake__echo" not in names
    assert other_sess._mcp is None

