"""Choose Grok vs Gemini once per call job.

Selection happens when MCP starts a call (`stella_call` / `stella_briefing_call`).
Grok Voice realtime is probed **once**. That provider is stored on the job and
used for every media WebSocket of that call. There is no mid-call re-probe and
no fallback after the session is already running.

Grok-down cache
---------------
Probe failures (HTTP 403, auth, quota/limit, websocket connect errors) are
remembered in-process for ``GROK_DOWN_CACHE_TTL_SECONDS`` (default **300s /
5 minutes**). The next MCP call in that window skips the Grok handshake and
selects Gemini immediately (if ``GEMINI_API_KEY`` is set). The cache is not
shared across processes and is cleared when a Grok probe succeeds.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Callable

from stella.config import Settings
from stella.errors import StellaError
from stella.xai_auth import XAIAuth

logger = logging.getLogger(__name__)

# In-memory TTL after a failed Grok realtime probe. Next MCP calls skip Grok.
GROK_DOWN_CACHE_TTL_SECONDS = 300

GrokProbeFn = Callable[[str], None]


def grok_realtime_url(settings: Settings) -> str:
    return f"{settings.xai_realtime_url}?model={settings.xai_voice_model}"


def default_grok_realtime_probe(settings: Settings, token: str) -> None:
    """Open Grok Voice realtime WS and close it. Raises on handshake/connect failure."""

    async def _once() -> None:
        import websockets

        url = grok_realtime_url(settings)
        headers = {"Authorization": f"Bearer {token}"}
        ws = await websockets.connect(
            url,
            additional_headers=headers,
            open_timeout=8,
            close_timeout=2,
        )
        try:
            await ws.close()
        except Exception:
            pass

    asyncio.run(_once())


class VoiceProviderChooser:
    """Process-wide Grok availability cache + one-shot provider pick for a job."""

    def __init__(
        self,
        settings: Settings,
        xai: XAIAuth,
        *,
        grok_probe: GrokProbeFn | None = None,
        clock: Callable[[], float] | None = None,
        down_ttl_seconds: float = GROK_DOWN_CACHE_TTL_SECONDS,
    ) -> None:
        self.settings = settings
        self.xai = xai
        self._probe = grok_probe
        self._clock = clock or time.monotonic
        self.down_ttl_seconds = down_ttl_seconds
        self._lock = threading.Lock()
        self._down_until: float | None = None
        self._down_reason: str = ""

    def grok_down_cached(self) -> str | None:
        with self._lock:
            if self._down_until is None:
                return None
            if self._clock() >= self._down_until:
                self._down_until = None
                self._down_reason = ""
                return None
            return self._down_reason or "Grok Voice unavailable (cached)"

    def mark_grok_down(self, reason: str) -> None:
        with self._lock:
            self._down_until = self._clock() + self.down_ttl_seconds
            self._down_reason = reason
        logger.warning(
            "Caching Grok Voice as down for %.0fs (%s)",
            self.down_ttl_seconds,
            reason,
        )

    def clear_grok_down(self) -> None:
        with self._lock:
            self._down_until = None
            self._down_reason = ""

    def choose(self) -> str:
        """Pick `grok` or `gemini` for a new call. Probes Grok at most once (or uses cache)."""
        gemini_ok = bool((self.settings.gemini_api_key or "").strip())

        cached = self.grok_down_cached()
        if cached:
            if gemini_ok:
                logger.info("Using Gemini for this call; Grok down cache: %s", cached)
                return "gemini"
            raise StellaError(
                f"{cached}. GEMINI_API_KEY is not set.",
                "voice_auth_missing",
            )

        grok_err: Exception | None = None
        token: str | None = None
        try:
            token = self.xai.bearer_token()
        except Exception as exc:
            grok_err = exc
            logger.warning("Grok credentials unavailable (%s)", exc)

        if token:
            try:
                self._run_probe(token)
                self.clear_grok_down()
                return "grok"
            except Exception as exc:
                grok_err = exc
                self.mark_grok_down(str(exc)[:500] or "Grok realtime probe failed")
                logger.warning("Grok Voice probe failed (%s)", exc)

        if gemini_ok:
            return "gemini"

        grok_msg = str(grok_err) if grok_err else "Grok Voice was not used"
        if isinstance(grok_err, StellaError):
            grok_msg = grok_err.message
        raise StellaError(
            f"{grok_msg}. Set GEMINI_API_KEY to enable Gemini Live as a voice fallback.",
            "voice_auth_missing",
        )

    def _run_probe(self, token: str) -> None:
        if self._probe is not None:
            self._probe(token)
            return
        default_grok_realtime_probe(self.settings, token)
