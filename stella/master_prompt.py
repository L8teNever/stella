"""Master prompt: shared style rules for all of Simon's AIs, stored in Ida Memory.

The rules live as observations of the entity `MASTER_PROMPT_ENTITY` in Ida Memory, so
Simon changes them in one place. Stella reads them over MCP, caches them in-process and
refreshes periodically; if Ida Memory is unreachable the last good text (or the built-in
defaults) is used.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from stella.config import Settings

logger = logging.getLogger(__name__)

# Meta observations that describe the entity itself, not rules for the AI.
_META_PREFIXES = ("ZWECK:", "FORM:")
_FETCH_TIMEOUT_S = 8.0

_text: str | None = None


def current() -> str | None:
    """Latest rules text, or None if never fetched (callers fall back to built-in defaults)."""
    return _text


def set_text(text: str | None) -> None:
    global _text
    _text = (text or "").strip() or None


def parse_entity(payload: Any, entity: str) -> str | None:
    """Extract the rule observations of `entity` from an open_nodes result."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None
    for ent in payload.get("entities") or []:
        if ent.get("name") == entity:
            rules = [
                str(o).strip()
                for o in ent.get("observations") or []
                if str(o).strip() and not str(o).strip().upper().startswith(_META_PREFIXES)
            ]
            return "\n".join(f"- {r}" for r in rules) or None
    return None


async def fetch(settings: Settings) -> str | None:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    try:
        with open(settings.ask_ida_mcp_config, encoding="utf-8") as fh:
            server = json.load(fh)["mcpServers"][settings.master_prompt_server]
    except (OSError, KeyError, ValueError):
        return None

    async def _run() -> str | None:
        async with streamablehttp_client(server["url"], headers=server.get("headers")) as (r, w, _):
            async with ClientSession(r, w) as session:
                await session.initialize()
                result = await session.call_tool(
                    "open_nodes", {"names": [settings.master_prompt_entity]}
                )
                for part in result.content:
                    text = getattr(part, "text", None)
                    if text:
                        parsed = parse_entity(text, settings.master_prompt_entity)
                        if parsed:
                            return parsed
        return None

    return await asyncio.wait_for(_run(), timeout=_FETCH_TIMEOUT_S)


async def refresh(settings: Settings) -> bool:
    """Fetch and cache the master prompt. Returns True if new text was stored."""
    if not settings.master_prompt_enabled:
        return False
    try:
        text = await fetch(settings)
    except Exception as exc:  # noqa: BLE001
        logger.warning("master prompt refresh failed: %s", exc)
        return False
    if text and text != _text:
        set_text(text)
        logger.info("master prompt updated (%d chars)", len(text))
        return True
    return False
