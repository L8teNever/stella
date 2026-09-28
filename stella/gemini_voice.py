from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any
from urllib.parse import urlencode

from stella.audio_pcmu import PcmToPcmu8k, Pcmu8kToPcm16k
from stella.config import Settings
from stella.store import CallJob
from stella.voice import KICKOFF_TEXT, build_instructions, hang_up_tool_openai

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
        self._down = PcmToPcmu8k(default_rate=24000)
        self.provider = "gemini"

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
                            "voiceName": self.settings.gemini_voice or "Puck",
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
        }

    async def run(self) -> None:
        await asyncio.gather(self._pump_telnyx(), self._pump_gemini())

    async def _send_kickoff(self) -> None:
        assert self._gemini_ws is not None
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
                if event == "media":
                    payload = (msg.get("media") or {}).get("payload")
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
        kickoff_sent = False
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
                if "setupComplete" in event and not kickoff_sent:
                    kickoff_sent = True
                    await self._send_kickoff()
                    continue
                await self._handle_gemini_event(event)
        finally:
            await self.close()

    async def _handle_gemini_event(self, event: dict[str, Any]) -> None:
        if event.get("error"):
            logger.error("Gemini Live error: %s", event)
            return
        server = event.get("serverContent") or {}
        model_turn = server.get("modelTurn") or {}
        for part in model_turn.get("parts") or []:
            inline = part.get("inlineData") or {}
            data = inline.get("data")
            mime = inline.get("mimeType") or ""
            if data and "audio" in mime.lower() and self._telnyx_ws:
                payload = self._down.convert_b64(data, mime)
                if payload:
                    await self._telnyx_ws.send_text(
                        json.dumps({"event": "media", "media": {"payload": payload}})
                    )
            text = part.get("text")
            if text:
                self._assistant_bits.append(text)
                self._transcript(text)
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
            await self._hangup()
            await self.close()
            return

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
        if self._gemini_ws is None:
            return
        try:
            await self._gemini_ws.close()
        except Exception:
            pass
