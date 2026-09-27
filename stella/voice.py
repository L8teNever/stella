from __future__ import annotations

import json
import logging
from typing import Any

from stella.config import Settings
from stella.store import CallJob

logger = logging.getLogger(__name__)


STELLA_SYSTEM = """You are Stella, a phone agent placing a live outbound call.
You are NOT Ida and you do NOT share Ida's memory, chat, or MCP tools.
You may use ONLY the job brief and extra context below. If the other party asks
something that is not in that material, say honestly that you don't know.
Do not invent facts, names, times, or numbers.

Complete the task, confirm the outcome in one short sentence if possible,
then say a polite goodbye. When the task is done or the other party hangs up
the conversation, call the hang_up tool.

If this is a briefing call, read the briefing text clearly, ask if they heard
it, then hang up. Do not add extra commentary beyond the briefing text.
"""


def build_instructions(job: CallJob) -> str:
    speak = f"You are speaking with: {job.speak_to}.\n" if job.speak_to else ""
    ctx = job.context.strip() or "(none provided)"
    return (
        f"{STELLA_SYSTEM}\n\n"
        f"{speak}"
        f"Job type: {job.kind}\n"
        f"Brief / task:\n{job.brief.strip()}\n\n"
        f"Extra context from the dispatcher (this is all you have):\n{ctx}\n"
    )


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

    async def attach_telnyx(self, telnyx_ws) -> None:
        self._telnyx_ws = telnyx_ws
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
                        "tools": [
                            {
                                "type": "function",
                                "name": "hang_up",
                                "description": "End the phone call after the task is complete.",
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
                        ],
                    },
                }
            )
        )
        # Kick off with a silent cue so Stella greets / reads the briefing.
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
                                "text": (
                                    "The other party just answered the phone. "
                                    "Begin the call now using only the brief and context."
                                ),
                            }
                        ],
                    },
                }
            )
        )
        await self._grok_ws.send(json.dumps({"type": "response.create"}))

        import asyncio

        await asyncio.gather(self._pump_telnyx(), self._pump_grok())

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
                if event == "media":
                    payload = (msg.get("media") or {}).get("payload")
                    if payload and self._grok_ws:
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
            if delta and self._telnyx_ws:
                await self._telnyx_ws.send_text(
                    json.dumps({"event": "media", "media": {"payload": delta}})
                )
        elif etype in {"response.output_text.delta", "response.audio_transcript.delta"}:
            piece = event.get("delta") or ""
            if piece:
                self._assistant_bits.append(piece)
                self._transcript(piece)
        elif etype == "response.output_item.done":
            item = event.get("item") or {}
            if item.get("type") == "function_call" and item.get("name") == "hang_up":
                args = item.get("arguments") or "{}"
                if isinstance(args, str):
                    try:
                        parsed = json.loads(args)
                    except json.JSONDecodeError:
                        parsed = {"outcome": args}
                else:
                    parsed = args
                outcome = str(parsed.get("outcome") or "Call completed.")
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
                await self._hangup()
                await self.close()
        elif etype == "error":
            logger.error("Grok Voice error: %s", event)

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
        for ws in (self._grok_ws,):
            if ws is None:
                continue
            try:
                await ws.close()
            except Exception:
                pass
