from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any
from urllib.parse import urlencode

from stella.audio_pcmu import GEMINI_OUT_RATE, PcmToPcmu8k, Pcmu8kToPcm16k
from stella.config import Settings
from stella.store import CallJob
from stella.voice import KICKOFF_TEXT, TelnyxMediaGuard, build_instructions, hang_up_tool_openai

logger = logging.getLogger(__name__)


def gemini_ws_url(settings: Settings) -> str:
    base = settings.gemini_realtime_url.rstrip("?")
    query = urlencode({"key": settings.gemini_api_key})
    return f"{base}?{query}"


def gemini_model_name(settings: Settings) -> str:
    model = (settings.gemini_live_model or "").strip()
    if not model:
        model = "gemini-2.5-flash-native-audio-preview-09-2025"
    if not model.startswith("models/"):
        model = f"models/{model}"
    return model


def gemini_realtime_input_config(settings: Settings) -> dict[str, Any]:
    """Phone-tuned Gemini Live VAD (BidiGenerateContent realtimeInputConfig)."""
    return {
        "automaticActivityDetection": {
            "disabled": False,
            "startOfSpeechSensitivity": settings.gemini_vad_start_sensitivity,
            "endOfSpeechSensitivity": settings.gemini_vad_end_sensitivity,
            "prefixPaddingMs": int(settings.gemini_vad_prefix_padding_ms),
            "silenceDurationMs": int(settings.gemini_vad_silence_duration_ms),
        },
        "activityHandling": settings.gemini_vad_activity_handling,
        "turnCoverage": settings.gemini_vad_turn_coverage,
    }


def hang_up_tool_gemini() -> dict[str, Any]:
    openai = hang_up_tool_openai()
    params = openai["parameters"]
    return {
        "functionDeclarations": [
            {
                "name": openai["name"],
                "description": openai["description"],
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "outcome": {
                            "type": "STRING",
                            "description": params["properties"]["outcome"]["description"],
                        }
                    },
                    "required": ["outcome"],
                },
            }
        ]
    }


class GeminiVoiceSession:
    """Bridge Telnyx media WebSocket <-> Gemini Live (BidiGenerateContent)."""

    def __init__(
        self,
        *,
        settings: Settings,
        job: CallJob,
        gemini_connect,
        hangup_cb,
        transcript_cb,
        outcome_cb,
    ) -> None:
        self.settings = settings
        self.job = job
        self._gemini_connect = gemini_connect
        self._hangup = hangup_cb
        self._transcript = transcript_cb
        self._outcome = outcome_cb
        self._telnyx_ws = None
        self._gemini_ws = None
        self._closed = False
        self._assistant_bits: list[str] = []
        self._up = Pcmu8kToPcm16k()
        self._down = PcmToPcmu8k(default_rate=GEMINI_OUT_RATE)
        self.provider = "gemini"
        self.guard = TelnyxMediaGuard()
        self._setup_complete = asyncio.Event()

    async def attach_telnyx(self, telnyx_ws) -> None:
        self._telnyx_ws = telnyx_ws
        await self.connect()
        await self.run()

    async def connect(self) -> None:
        if not (self.settings.gemini_api_key or "").strip():
            raise RuntimeError("GEMINI_API_KEY is not set; cannot open Gemini Live session.")
        url = gemini_ws_url(self.settings)
        try:
            self._gemini_ws = await self._gemini_connect(url, {})
        except Exception as exc:
            logger.exception("Gemini Live connect failed")
            await self._send_telnyx_clear()
            raise RuntimeError(f"Could not open Gemini Live session: {exc}") from exc

        await self._gemini_ws.send(json.dumps({"setup": self._setup_payload()}))

    def _setup_payload(self) -> dict[str, Any]:
        return {
            "model": gemini_model_name(self.settings),
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": self.settings.gemini_voice or "Aoede",
                        }
                    }
                },
            },
            "systemInstruction": {
                "parts": [{"text": build_instructions(self.job)}]
            },
            "tools": [hang_up_tool_gemini()],
            "outputAudioTranscription": {},
            "inputAudioTranscription": {},
            "realtimeInputConfig": gemini_realtime_input_config(self.settings),
        }

    async def run(self) -> None:
        await asyncio.gather(
            self._pump_telnyx(),
            self._pump_gemini(),
            self.guard.kickoff_once(lambda: self._closed, self._send_kickoff),
        )

    async def _send_kickoff(self) -> None:
        await self._setup_complete.wait()
        if self._closed or self._gemini_ws is None:
            return
        await self._gemini_ws.send(
            json.dumps(
                {
                    "clientContent": {
                        "turns": [
                            {
                                "role": "user",
                                "parts": [{"text": KICKOFF_TEXT}],
                            }
                        ],
                        "turnComplete": True,
                    }
                }
            )
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
                    if payload and self._gemini_ws:
                        pcm = self._up.convert_b64(payload)
                        if not pcm:
                            continue
                        await self._gemini_ws.send(
                            json.dumps(
                                {
                                    "realtimeInput": {
                                        "audio": {
                                            "data": base64.b64encode(pcm).decode("ascii"),
                                            "mimeType": "audio/pcm;rate=16000",
                                        }
                                    }
                                }
                            )
                        )
                elif event in {"stop", "closed"}:
                    break
        finally:
            await self.close()

    async def _pump_gemini(self) -> None:
        assert self._gemini_ws is not None
        try:
            async for raw in self._gemini_ws:
                if self._closed:
                    break
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if "setupComplete" in event:
                    self._setup_complete.set()
                    continue
                await self._handle_gemini_event(event)
        finally:
            await self.close()

    async def _handle_gemini_event(self, event: dict[str, Any]) -> None:
        if event.get("error"):
            logger.error("Gemini Live error: %s", event)
            return
        server = event.get("serverContent") or {}
        if server.get("interrupted"):
            await self._on_barge_in()
        model_turn = server.get("modelTurn") or {}
        for part in model_turn.get("parts") or []:
            inline = part.get("inlineData") or {}
            data = inline.get("data")
            mime = inline.get("mimeType") or ""
            if data and "audio" in mime.lower():
                for payload in self._down.convert_frames_b64(data, mime):
                    await self.guard.send_outbound_pcmu(self._telnyx_ws, payload)
            # part.text is often model scratch / thinking, not spoken audio.
            # Spoken text is outputTranscription only.
        out_tx = server.get("outputTranscription") or {}
        piece = out_tx.get("text") or ""
        if piece:
            self._assistant_bits.append(piece)
            self._transcript(piece)

        tool_call = event.get("toolCall") or {}
        for fc in tool_call.get("functionCalls") or []:
            if fc.get("name") != "hang_up":
                continue
            args = fc.get("args") or fc.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"outcome": args}
            outcome = str(args.get("outcome") or "Call completed.")
            self._outcome(outcome)
            if self._gemini_ws:
                await self._gemini_ws.send(
                    json.dumps(
                        {
                            "toolResponse": {
                                "functionResponses": [
                                    {
                                        "id": fc.get("id"),
                                        "name": "hang_up",
                                        "response": {"result": "hanging up"},
                                    }
                                ]
                            }
                        }
                    )
                )
            self.guard.arm_hangup(self._deferred_hangup)
            return

    async def _deferred_hangup(self) -> None:
        try:
            await self.guard.telnyx_started.wait()
            for leftover in self._down.flush_frames_b64():
                await self.guard.send_outbound_pcmu(self._telnyx_ws, leftover)
            await self.guard.hangup_after_audio(self._hangup, lambda: self._closed)
        finally:
            await self.close()

    async def _on_barge_in(self) -> None:
        """Stop queued Telnyx playout when Gemini reports the user interrupted."""
        self._down = PcmToPcmu8k(default_rate=GEMINI_OUT_RATE)
        await self._send_telnyx_clear()

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
        self._setup_complete.set()
        self.guard.mark_started()
        if self._gemini_ws is None:
            return
        try:
            await self._gemini_ws.close()
        except Exception:
            pass
