from __future__ import annotations

import asyncio
import audioop
import base64
import json
import logging
import re
from typing import Any, Callable

from stella.config import Settings
from stella.store import CallJob

logger = logging.getLogger(__name__)

# Telnyx RTP: 20 ms per 160-byte PCMU frame. Hang up only after the stream is
# live and queued outbound audio has had time to play — otherwise Gemini/Grok
# can complete (and call hang_up) during setup/ringing with 0 packets out.
TELNYX_PCMU_FRAME_MS = 20
TELNYX_PCMU_FRAME_BYTES = 160
HANGUP_WAIT_AUDIO_S = 8.0
HANGUP_PLAYOUT_PAD_S = 0.35
HANGUP_PLAYOUT_CAP_S = 12.0


def pcmu_payload_duration_ms(payload_b64: str) -> int:
    try:
        n = len(base64.b64decode(payload_b64, validate=False))
    except Exception:
        return TELNYX_PCMU_FRAME_MS
    if n <= 0:
        return TELNYX_PCMU_FRAME_MS
    frames = max(1, n // TELNYX_PCMU_FRAME_BYTES)
    return frames * TELNYX_PCMU_FRAME_MS


class TelnyxMediaGuard:
    """Hold kickoff until Telnyx `start`/`media`; delay hangup until audio playout."""

    def __init__(
        self,
        *,
        hangup_wait_audio_s: float = HANGUP_WAIT_AUDIO_S,
        hangup_playout_pad_s: float = HANGUP_PLAYOUT_PAD_S,
        sleep: Callable[[float], Any] | None = None,
    ) -> None:
        self.telnyx_started = asyncio.Event()
        self.kickoff_sent = False
        self.outbound_ms = 0
        self.hangup_wait_audio_s = hangup_wait_audio_s
        self.hangup_playout_pad_s = hangup_playout_pad_s
        self._kickoff_lock = asyncio.Lock()
        self._sleep = sleep or asyncio.sleep
        self._hangup_task: asyncio.Task | None = None

    def arm_hangup(self, deferred) -> asyncio.Task:
        if self._hangup_task is None or self._hangup_task.done():
            self._hangup_task = asyncio.create_task(deferred())
        return self._hangup_task

    def mark_started(self) -> None:
        self.telnyx_started.set()

    async def send_outbound_pcmu(self, telnyx_ws, payload: str) -> bool:
        """Send one Telnyx media event if the RTP stream has started. Drops early audio."""
        if not telnyx_ws or not payload:
            return False
        if not self.telnyx_started.is_set():
            logger.info("Dropping outbound PCMU until Telnyx stream start")
            return False
        await telnyx_ws.send_text(json.dumps({"event": "media", "media": {"payload": payload}}))
        self.outbound_ms += pcmu_payload_duration_ms(payload)
        return True

    async def kickoff_once(self, closed: Callable[[], bool], send_kickoff) -> None:
        async with self._kickoff_lock:
            if self.kickoff_sent or closed():
                return
            await self.telnyx_started.wait()
            if closed() or self.kickoff_sent:
                return
            await send_kickoff()
            self.kickoff_sent = True

    async def hangup_after_audio(self, hangup_cb, closed: Callable[[], bool]) -> None:
        """Wait for stream start + outbound audio (or timeout), then playout delay."""
        await self.telnyx_started.wait()
        deadline = asyncio.get_running_loop().time() + self.hangup_wait_audio_s
        while self.outbound_ms <= 0 and not closed():
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                logger.warning(
                    "hang_up with no outbound PCMU after stream start; hanging up anyway"
                )
                break
            await self._sleep(min(0.05, remaining))
        if self.outbound_ms > 0:
            wait_s = min(
                self.outbound_ms / 1000.0 + self.hangup_playout_pad_s,
                HANGUP_PLAYOUT_CAP_S,
            )
            await self._sleep(wait_s)
        if closed():
            return
        result = hangup_cb()
        if asyncio.iscoroutine(result):
            await result


STELLA_SYSTEM = """You are Stella, a live phone agent. Not Ida; no memory except the brief below.
If asked something not in the brief/context, say you don't know. Do not invent facts.

Keep turns short (one or two sentences). Answer immediately; never narrate reasoning.

Phone audio is noisy. Do not say you do not understand — including "ich verstehe nicht",
"kannst du das wiederholen", "sorry?", or asking them to repeat — unless the caller
clearly said nothing. If you are unsure, make a short best guess from the brief, or
ask one crisp clarifying question once. Never loop those phrases.

When the task is done, they say goodbye, or you have nothing left to ask: say a
brief goodbye if needed AND call hang_up. Spoken goodbye does not end the PSTN
call — hang_up does. Do not wait for extra confirmation or linger.

Briefings: read the text, ask if they heard it, then call hang_up.
"""

_GERMAN_HINT = re.compile(
    r"[äöüÄÖÜß]|\b(deutsch|german|hallo|bitte|danke|guten|reservier|"
    r"termin|anruf|sprich)\b",
    re.IGNORECASE,
)


def job_wants_german(job: CallJob) -> bool:
    """True when speak_to / brief / context / dest look German (e.g. +49)."""
    blob = " ".join(
        part
        for part in (job.speak_to, job.brief, job.context, getattr(job, "to_number", "") or "")
        if part
    )
    compact = blob.replace(" ", "")
    if "+49" in compact:
        return True
    return bool(_GERMAN_HINT.search(blob))

KICKOFF_TEXT = "They answered. Start from the brief now."


def build_instructions(job: CallJob) -> str:
    speak = f"You are speaking with: {job.speak_to}.\n" if job.speak_to else ""
    lang = ""
    if job_wants_german(job):
        lang = (
            "Speak German unless the other party switches language. "
            "Reply immediately; do not pause to 'think out loud'.\n"
        )
    ctx = job.context.strip() or "(none provided)"
    return (
        f"{STELLA_SYSTEM}\n\n"
        f"{speak}"
        f"{lang}"
        f"Job type: {job.kind}\n"
        f"Brief / task:\n{job.brief.strip()}\n\n"
        f"Extra context from the dispatcher (this is all you have):\n{ctx}\n"
    )


def hang_up_tool_openai() -> dict[str, Any]:
    return {
        "type": "function",
        "name": "hang_up",
        "description": (
            "End the live phone call now. Call this when the task is complete, "
            "the other party says goodbye, a briefing was delivered, or there is "
            "nothing left to ask. Required to actually hang up the PSTN leg."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "outcome": {
                    "type": "string",
                    "description": "One-sentence result of the call.",
                }
            },
            "required": ["outcome"],
        },
    }


def parse_hang_up_args(args: Any) -> str:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return args or "Call completed."
    if not isinstance(args, dict):
        return "Call completed."
    return str(args.get("outcome") or "Call completed.")


FAREWELL_RE = re.compile(
    r"(tsch+ü+ss|tschuss|tschüss|auf wiederh[öo]ren|auf wiedersehen|"
    r"goodbye|good\s*bye|\bbye\b|ciao)",
    re.IGNORECASE,
)


def looks_like_farewell(text: str) -> bool:
    return bool(text and FAREWELL_RE.search(text))


class FarewellHangupWatch:
    """If spoken transcript is a farewell and inbound stays quiet, hang up once."""

    def __init__(
        self,
        *,
        enabled: bool,
        delay_s: float,
        arm,
        sleep: Callable[[float], Any] | None = None,
    ) -> None:
        self.enabled = enabled
        self.delay_s = max(0.05, float(delay_s))
        self._arm = arm
        self._sleep = sleep or asyncio.sleep
        self._task: asyncio.Task | None = None
        self._armed = False

    def note_user_speech(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None

    def note_assistant_text(self, text: str) -> None:
        if not self.enabled or self._armed or not looks_like_farewell(text):
            return
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._fire())

    async def _fire(self) -> None:
        try:
            await self._sleep(self.delay_s)
        except asyncio.CancelledError:
            return
        if self._armed:
            return
        self._armed = True
        self._arm("Call ended after farewell.")


class VoiceSession:
    """Bridge Telnyx media WebSocket <-> xAI Grok Voice realtime WebSocket."""

    def __init__(
        self,
        *,
        settings: Settings,
        job: CallJob,
        bearer_token: str,
        grok_connect,
        hangup_cb,
        transcript_cb,
        outcome_cb,
    ) -> None:
        self.settings = settings
        self.job = job
        self.bearer_token = bearer_token
        self._grok_connect = grok_connect
        self._hangup = hangup_cb
        self._transcript = transcript_cb
        self._outcome = outcome_cb
        self._telnyx_ws = None
        self._grok_ws = None
        self._closed = False
        self._assistant_bits: list[str] = []
        self.provider = "grok"
        self.guard = TelnyxMediaGuard()
        self._farewell = FarewellHangupWatch(
            enabled=bool(settings.stella_farewell_hangup),
            delay_s=settings.stella_farewell_hangup_s,
            arm=self._arm_hangup_from_farewell,
        )

    async def attach_telnyx(self, telnyx_ws) -> None:
        self._telnyx_ws = telnyx_ws
        await self.connect()
        await self.run()

    async def connect(self) -> None:
        url = f"{self.settings.xai_realtime_url}?model={self.settings.xai_voice_model}"
        headers = {"Authorization": f"Bearer {self.bearer_token}"}
        try:
            self._grok_ws = await self._grok_connect(url, headers)
        except Exception as exc:
            logger.exception("Grok Voice connect failed")
            await self._send_telnyx_clear()
            raise RuntimeError(f"Could not open Grok Voice session: {exc}") from exc

        await self._grok_ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "voice": self.settings.xai_voice,
                        "instructions": build_instructions(self.job),
                        "turn_detection": {"type": "server_vad"},
                        "audio": {
                            "input": {"format": {"type": "audio/pcmu", "rate": 8000}},
                            "output": {"format": {"type": "audio/pcmu", "rate": 8000}},
                        },
                        "tools": [hang_up_tool_openai()],
                    },
                }
            )
        )
        # Kickoff waits for Telnyx stream start so the model does not "complete"
        # (and hang_up) during ringing before any RTP can flow.

    async def _send_kickoff(self) -> None:
        assert self._grok_ws is not None
        await self._grok_ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": KICKOFF_TEXT,
                            }
                        ],
                    },
                }
            )
        )
        await self._grok_ws.send(json.dumps({"type": "response.create"}))

    async def run(self) -> None:
        await asyncio.gather(
            self._pump_telnyx(),
            self._pump_grok(),
            self.guard.kickoff_once(lambda: self._closed, self._send_kickoff),
        )

    async def _pump_telnyx(self) -> None:
        assert self._telnyx_ws is not None
        try:
            async for raw in self._telnyx_ws.iter_text():
                if self._closed:
                    break
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                event = msg.get("event")
                if event in {"start", "media"}:
                    self.guard.mark_started()
                if event == "media":
                    media = msg.get("media") or {}
                    track = (media.get("track") or "").lower()
                    if track in {"outbound", "outbound_track"}:
                        continue
                    payload = media.get("payload") or media.get("chunk")
                    if payload and self._grok_ws:
                        try:
                            ulaw = base64.b64decode(payload)
                            rms = audioop.rms(audioop.ulaw2lin(ulaw, 2), 2) if ulaw else 0
                        except Exception:
                            rms = 0
                        if rms >= int(self.settings.stella_client_vad_rms):
                            self._farewell.note_user_speech()
                        await self._grok_ws.send(
                            json.dumps(
                                {
                                    "type": "input_audio_buffer.append",
                                    "audio": payload,
                                }
                            )
                        )
                elif event in {"stop", "closed"}:
                    break
        finally:
            await self.close()

    async def _pump_grok(self) -> None:
        assert self._grok_ws is not None
        try:
            async for raw in self._grok_ws:
                if self._closed:
                    break
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                await self._handle_grok_event(event)
        finally:
            await self.close()

    async def _handle_grok_event(self, event: dict[str, Any]) -> None:
        etype = event.get("type") or ""
        if etype == "response.output_audio.delta":
            delta = event.get("delta")
            if delta:
                await self.guard.send_outbound_pcmu(self._telnyx_ws, delta)
        elif etype in {"response.output_text.delta", "response.audio_transcript.delta"}:
            piece = event.get("delta") or ""
            if piece:
                self._assistant_bits.append(piece)
                self._transcript(piece)
                self._farewell.note_assistant_text(piece)
        elif etype == "response.output_item.done":
            item = event.get("item") or {}
            if item.get("type") == "function_call" and item.get("name") == "hang_up":
                outcome = parse_hang_up_args(item.get("arguments") or "{}")
                self._outcome(outcome)
                if self._grok_ws:
                    await self._grok_ws.send(
                        json.dumps(
                            {
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "function_call_output",
                                    "call_id": item.get("call_id"),
                                    "output": "hanging up",
                                },
                            }
                        )
                    )
                self.guard.arm_hangup(self._deferred_hangup)
        elif etype == "error":
            logger.error("Grok Voice error: %s", event)

    def _arm_hangup_from_farewell(self, outcome: str) -> None:
        if self._closed:
            return
        self._outcome(outcome)
        self.guard.arm_hangup(self._deferred_hangup)

    async def _deferred_hangup(self) -> None:
        try:
            await self.guard.hangup_after_audio(self._hangup, lambda: self._closed)
        finally:
            await self.close()

    async def _send_telnyx_clear(self) -> None:
        if self._telnyx_ws:
            try:
                await self._telnyx_ws.send_text(json.dumps({"event": "clear"}))
            except Exception:
                pass

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.guard.mark_started()
        for ws in (self._grok_ws,):
            if ws is None:
                continue
            try:
                await ws.close()
            except Exception:
                pass


async def start_voice_bridge(
    *,
    settings: Settings,
    job: CallJob,
    telnyx_ws,
    grok_token_fn: Callable[[], str],
    grok_connect,
    hangup_cb,
    transcript_cb,
    outcome_cb,
    provider_cb: Callable[[str], None] | None = None,
    gemini_connect=None,
    locked_provider: str | None = None,
) -> str:
    """Run the already-chosen voice provider for this job.

    ``locked_provider`` is the sticky choice from MCP (`grok` or `gemini`).
    Connect and media pumps use only that provider. A mid-session Grok drop
    does **not** open Gemini (and vice versa). If unset, falls back to the
    job's stored ``voice_provider``.
    """
    from stella.gemini_voice import GeminiVoiceSession

    provider = (locked_provider or job.voice_provider or "").strip().lower()
    if provider not in {"grok", "gemini"}:
        raise RuntimeError(
            "Voice provider was not selected for this call. "
            "MCP place_call must choose grok or gemini before media starts."
        )

    if provider_cb:
        provider_cb(provider)

    if provider == "grok":
        token = grok_token_fn()
        session = VoiceSession(
            settings=settings,
            job=job,
            bearer_token=token,
            grok_connect=grok_connect,
            hangup_cb=hangup_cb,
            transcript_cb=transcript_cb,
            outcome_cb=outcome_cb,
        )
        session._telnyx_ws = telnyx_ws
        await session.connect()
        await session.run()
        return "grok"

    gemini_key = (settings.gemini_api_key or "").strip()
    if not gemini_key:
        raise RuntimeError(
            "This call is locked to Gemini Live but GEMINI_API_KEY is not set."
        )

    gem = GeminiVoiceSession(
        settings=settings,
        job=job,
        gemini_connect=gemini_connect or grok_connect,
        hangup_cb=hangup_cb,
        transcript_cb=transcript_cb,
        outcome_cb=outcome_cb,
    )
    gem._telnyx_ws = telnyx_ws
    await gem.connect()
    await gem.run()
    return "gemini"
