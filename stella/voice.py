from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from typing import Any, Callable

from stella.config import Settings
from stella.store import CallJob

logger = logging.getLogger(__name__)

# Telnyx RTP: 20 ms per 160-byte PCMU frame. Hang up only after the stream is
# live and queued outbound audio has had time to play — otherwise Gemini
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
        # Monotonic time at which everything queued to Telnyx will have finished playing.
        self._play_end = 0.0
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
        dur_ms = pcmu_payload_duration_ms(payload)
        self.outbound_ms += dur_ms
        self._play_end = max(self._play_end, time.monotonic()) + dur_ms / 1000.0
        return True

    def clear_playout(self) -> None:
        """Telnyx `clear` flushed its queue: nothing is left to play out."""
        self._play_end = 0.0

    def remaining_playout_s(self) -> float:
        return max(0.0, self._play_end - time.monotonic())

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
            # Only wait for audio that is still queued, not for the whole call's audio.
            wait_s = min(
                self.remaining_playout_s() + self.hangup_playout_pad_s,
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

Ending the call is YOUR decision. As soon as the conversation is over (task done, they say
goodbye or thanks with nothing left, or you have nothing left to ask): say one brief goodbye
and call hang_up in that very same turn. Whenever you say goodbye (Tschüss, Auf Wiedersehen,
Bis dann ...) you MUST call hang_up right away — never say goodbye without hang_up, never
wait for the other side to hang up, never ask "noch etwas?" after a goodbye. A spoken goodbye
alone does not end the PSTN call; hang_up does.

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


ASK_IDA_INSTRUCTIONS = """
Lookup tool frag_ida: You are talking to Simon himself. If he asks about his calendar
appointments, school timetable or cancellations, homework, e-mails (read only) or smart
home status, first say a short "Moment, ich schau nach" and then call frag_ida with his
question in the `frage` parameter. Afterwards say the result in your own words, but completely and faithfully: do not drop
any item, time or detail from it and do not add anything that is not in it.
After you have given the lookup result, ask once whether he needs anything else; if he
says no/thanks/bye or stays quiet, say goodbye and call hang_up immediately.
Never invent data. If frag_ida reports an error or finds nothing, say so honestly.
If the result contains AKTION_BRAUCHT_BESTAETIGUNG, tell Simon what would be done and ask
"Soll ich das wirklich machen?". Only if he clearly says yes, call frag_ida again with the
same question and bestaetigt=true. Never set bestaetigt=true without his explicit yes.
Not for general knowledge: answer those yourself or say you don't know.
"""


def build_instructions(job: CallJob, *, ask_ida: bool = False) -> str:
    speak = f"You are speaking with: {job.speak_to}.\n" if job.speak_to else ""
    lang = ""
    if job_wants_german(job):
        lang = (
            "Speak German unless the other party switches language. "
            "Short spoken replies only (one or two sentences). "
            "Reply immediately; do not pause to 'think out loud'. "
            "Never loop 'ich verstehe nicht' or ask to repeat unless they said nothing.\n"
        )
    ctx = job.context.strip() or "(none provided)"
    ida = ASK_IDA_INSTRUCTIONS if ask_ida else ""
    return (
        f"{STELLA_SYSTEM}\n\n"
        f"{speak}"
        f"{lang}"
        f"Job type: {job.kind}\n"
        f"Brief / task:\n{job.brief.strip()}\n\n"
        f"Extra context from the dispatcher (this is all you have):\n{ctx}\n"
        f"{ida}"
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


async def start_voice_bridge(
    *,
    settings: Settings,
    job: CallJob,
    telnyx_ws,
    ws_connect,
    hangup_cb,
    transcript_cb,
    outcome_cb,
) -> str:
    """Run the Gemini Live bridge for this job."""
    from stella.gemini_voice import GeminiVoiceSession

    if not (settings.gemini_api_key or "").strip():
        raise RuntimeError("GEMINI_API_KEY is not set.")

    gem = GeminiVoiceSession(
        settings=settings,
        job=job,
        gemini_connect=ws_connect,
        hangup_cb=hangup_cb,
        transcript_cb=transcript_cb,
        outcome_cb=outcome_cb,
    )
    gem._telnyx_ws = telnyx_ws
    await gem.connect()
    await gem.run()
    return "gemini"
