"""Per-call MCP clients for Gemini Live.

Each owner call opens its own Streamable HTTP sessions, lists the allowlisted
tools, and closes the sessions when the call ends. Tool results go back to
Gemini as function responses. Nothing here is shared across calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import httpx

from stella.ask_ida import ida_allowed_for
from stella.config import Settings

logger = logging.getLogger(__name__)

_NAME = re.compile(r"[^A-Za-z0-9_]+")
_RESERVED = frozenset({"hang_up", "frag_ida", "aufgabe_planen"})
_TYPE_MAP = {
    "object": "OBJECT",
    "string": "STRING",
    "integer": "INTEGER",
    "number": "NUMBER",
    "boolean": "BOOLEAN",
    "array": "ARRAY",
}
_RESULT_LIMIT = 2000
MSG_TIMEOUT = "Das hat zu lange gedauert."
MSG_UNREACHABLE = "Das Tool ist gerade nicht erreichbar."
MSG_UNKNOWN = "Unbekanntes Tool."


@dataclass(frozen=True)
class McpServerSpec:
    name: str
    url: str
    token_env: str = ""
    # None: every tool on the server. Empty: none.
    allow: tuple[str, ...] | None = None


def parse_mcp_servers(raw: str) -> list[McpServerSpec]:
    """Parse STELLA_MCP_SERVERS. Bad JSON yields no servers and does not raise."""
    text = (raw or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("stella_mcp ok=error reason=config")
        return []
    if not isinstance(data, list):
        logger.warning("stella_mcp ok=error reason=config")
        return []
    specs: list[McpServerSpec] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        url = str(item.get("url") or "").strip()
        if not name or not url:
            continue
        token_env = str(item.get("token_env") or "").strip()
        allow: tuple[str, ...] | None = None
        if "allow" in item and item.get("allow") is not None:
            raw_allow = item.get("allow")
            if not isinstance(raw_allow, list):
                continue
            allow = tuple(str(tool_name).strip() for tool_name in raw_allow if str(tool_name).strip())
        specs.append(McpServerSpec(name=name, url=url, token_env=token_env, allow=allow))
    return specs


def live_mcp_enabled(settings: Settings, to_number: str, allow_ida: bool) -> bool:
    """Same gate as frag_ida: owner number, ask-ida enabled, and this call allows it."""
    if not allow_ida or not parse_mcp_servers(settings.stella_mcp_servers):
        return False
    return ida_allowed_for(settings, to_number)


def prefix_tool(server: str, tool: str) -> str:
    """Gemini function name. Server prefix keeps two servers from colliding."""
    left = _safe_name(server, "server")
    right = _safe_name(tool, "tool")
    return f"{left}__{right}"[:64]


def to_gemini_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    """JSON Schema from MCP → Gemini function-declaration schema."""
    if not isinstance(schema, dict):
        return {"type": "OBJECT", "properties": {}}
    out: dict[str, Any] = {}
    raw_type = schema.get("type")
    if isinstance(raw_type, list):
        raw_type = next((item for item in raw_type if item != "null"), "string")
    if isinstance(raw_type, str):
        out["type"] = _TYPE_MAP.get(raw_type.lower(), "STRING")
    description = schema.get("description")
    if isinstance(description, str) and description.strip():
        out["description"] = description.strip()[:500]
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        out["enum"] = [str(item) for item in enum[:20]]
    properties = schema.get("properties")
    if isinstance(properties, dict):
        out["properties"] = {
            str(key)[:64]: to_gemini_schema(value if isinstance(value, dict) else {})
            for key, value in list(properties.items())[:32]
        }
    required = schema.get("required")
    if isinstance(required, list) and required:
        out["required"] = [str(item) for item in required if str(item).strip()]
    items = schema.get("items")
    if isinstance(items, dict):
        out["items"] = to_gemini_schema(items)
    if "type" not in out:
        out["type"] = "OBJECT" if "properties" in out else "STRING"
    if out["type"] == "OBJECT" and "properties" not in out:
        out["properties"] = {}
    return out


def format_tool_result(result: Any) -> dict[str, str]:
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        piece = getattr(block, "text", None)
        if isinstance(piece, str) and piece.strip():
            parts.append(piece.strip())
    text = "\n".join(parts).strip()
    if not text:
        structured = getattr(result, "structuredContent", None)
        if isinstance(structured, dict) and structured:
            text = json.dumps(structured, ensure_ascii=False)
    text = (text or "Keine Angaben.")[:_RESULT_LIMIT]
    if getattr(result, "isError", False):
        return {"error": text}
    return {"result": text}


def _safe_name(raw: str, fallback: str) -> str:
    cleaned = _NAME.sub("_", (raw or "").strip()).strip("_")
    if not cleaned:
        cleaned = fallback
    if cleaned[0].isdigit():
        cleaned = f"t_{cleaned}"
    return cleaned[:32]


def _bearer(spec: McpServerSpec) -> str | None:
    """Empty string when no token is configured. None when the env var is missing."""
    if not spec.token_env:
        return ""
    token = (os.environ.get(spec.token_env) or "").strip()
    return token or None


@dataclass
class _Bound:
    link: ServerLink
    original: str
    description: str
    declaration: dict[str, Any]


class _Call:
    def __init__(self, name: str, args: dict[str, Any]) -> None:
        self.name = name
        self.args = args
        self.future: asyncio.Future[tuple[Any | None, str]] = asyncio.get_running_loop().create_future()


class ServerLink:
    """One MCP server for a single phone call.

    The session lives in a dedicated task so its cancel scope is entered and
    left there. Tool calls from the audio loop only enqueue work. A timed-out
    or dropped session is reconnected once from that same task.
    """

    def __init__(self, spec: McpServerSpec, tool_timeout_s: float, connect_timeout_s: float) -> None:
        self.spec = spec
        self.tool_timeout_s = tool_timeout_s
        self.connect_timeout_s = connect_timeout_s
        self._queue: asyncio.Queue[_Call | None] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._opened = asyncio.Event()
        self._error: BaseException | None = None
        self._reconnects = 0
        self.tools: list[Any] = []

    async def open(self) -> list[Any]:
        self._task = asyncio.create_task(self._run())
        await self._opened.wait()
        if self._error is not None:
            await self.close()
            raise self._error
        return self.tools

    async def close(self) -> None:
        if self._task is None:
            return
        await self._queue.put(None)
        try:
            await self._task
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.info("stella_mcp server=%s ok=error reason=close", self.spec.name)
        self._task = None

    async def invoke(self, original: str, args: dict[str, Any]) -> tuple[Any | None, str]:
        if self._task is None or self._task.done():
            return None, "unreachable"
        call = _Call(original, args)
        await self._queue.put(call)
        return await call.future

    async def _run(self) -> None:
        try:
            while True:
                try:
                    if await self._serve():
                        return
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if not self._opened.is_set():
                        self._error = exc
                        self._opened.set()
                        return
                    self._reconnects += 1
                    if self._reconnects > 1:
                        return
        finally:
            self._fail_queued("unreachable")
            if not self._opened.is_set():
                self._opened.set()

    async def _serve(self) -> None:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        token = _bearer(self.spec)
        if token is None:
            raise RuntimeError("missing token")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        async with AsyncExitStack() as stack:
            client = httpx.AsyncClient(
                headers=headers,
                timeout=httpx.Timeout(self.connect_timeout_s + self.tool_timeout_s),
            )
            await stack.enter_async_context(client)
            read, write, _sid = await stack.enter_async_context(
                streamable_http_client(self.spec.url, http_client=client)
            )
            session = await stack.enter_async_context(ClientSession(read, write))
            await asyncio.wait_for(session.initialize(), timeout=self.connect_timeout_s)
            if not self._opened.is_set():
                listed = await session.list_tools()
                self.tools = list(listed.tools)
                self._opened.set()
            self._reconnects = 0
            while True:
                item = await self._queue.get()
                if item is None:
                    return True
                timeout = timedelta(seconds=self.tool_timeout_s)
                try:
                    result = await asyncio.wait_for(
                        session.call_tool(item.name, arguments=item.args, read_timeout_seconds=timeout),
                        timeout=self.tool_timeout_s,
                    )
                except TimeoutError:
                    if not item.future.done():
                        item.future.set_result((None, "timeout"))
                    raise TimeoutError("tool timeout")
                except asyncio.CancelledError:
                    if not item.future.done():
                        item.future.cancel()
                    raise
                except Exception:
                    if not item.future.done():
                        item.future.set_result((None, "unreachable"))
                    raise
                else:
                    if not item.future.done():
                        item.future.set_result((result, "ok"))

    def _fail_queued(self, reason: str) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if item is not None and not item.future.done():
                item.future.set_result((None, reason))


class LiveMcpPool:
    """MCP sessions and Gemini function declarations for one call."""

    def __init__(
        self,
        specs: list[McpServerSpec],
        *,
        tool_timeout_s: float = 8.0,
        connect_timeout_s: float = 8.0,
    ) -> None:
        self.specs = specs
        self.tool_timeout_s = tool_timeout_s
        self.connect_timeout_s = connect_timeout_s
        self._links: list[ServerLink] = []
        self._bound: dict[str, _Bound] = {}

    def __contains__(self, name: str) -> bool:
        return name in self._bound

    def catalog(self) -> list[tuple[str, str]]:
        return [(name, bound.description) for name, bound in self._bound.items()]

    def declarations(self) -> list[dict[str, Any]]:
        return [bound.declaration for bound in self._bound.values()]

    async def open(self) -> None:
        for spec in self.specs:
            token = _bearer(spec)
            if token is None:
                logger.info("stella_mcp server=%s ok=error reason=token", spec.name)
                continue
            link = ServerLink(spec, self.tool_timeout_s, self.connect_timeout_s)
            try:
                tools = await link.open()
            except Exception:
                logger.info("stella_mcp server=%s ok=error reason=connect", spec.name)
                await link.close()
                continue
            added = 0
            allow = None if spec.allow is None else set(spec.allow)
            for tool in tools:
                original = str(getattr(tool, "name", "") or "")
                if not original or (allow is not None and original not in allow):
                    continue
                prefixed = prefix_tool(spec.name, original)
                if prefixed in _RESERVED or prefixed in self._bound:
                    logger.info("stella_mcp tool=%s ok=error reason=name", prefixed)
                    continue
                description = str(getattr(tool, "description", "") or original).strip()[:300]
                schema = getattr(tool, "inputSchema", None)
                self._bound[prefixed] = _Bound(
                    link=link,
                    original=original,
                    description=description or original,
                    declaration={
                        "name": prefixed,
                        "description": description or original,
                        "parameters": to_gemini_schema(schema if isinstance(schema, dict) else {}),
                    },
                )
                added += 1
            if added:
                self._links.append(link)
            else:
                await link.close()

    async def close(self) -> None:
        links = self._links
        self._links = []
        self._bound.clear()
        for link in links:
            await link.close()

    async def call(self, name: str, args: dict[str, Any]) -> dict[str, str]:
        bound = self._bound.get(name)
        started = time.perf_counter()
        if bound is None:
            logger.info("stella_mcp tool=%s ms=0 ok=error reason=unknown", name)
            return {"error": MSG_UNKNOWN}
        reason = "unreachable"
        result: Any = None
        try:
            result, reason = await bound.link.invoke(bound.original, args)
            if reason == "ok" and result is not None and getattr(result, "isError", False):
                reason = "tool_error"
            if result is None:
                return {"error": MSG_TIMEOUT if reason == "timeout" else MSG_UNREACHABLE}
            return format_tool_result(result)
        finally:
            ms = int((time.perf_counter() - started) * 1000)
            logger.info(
                "stella_mcp tool=%s ms=%d ok=%s reason=%s",
                name,
                ms,
                "ok" if reason == "ok" else "error",
                reason,
            )
