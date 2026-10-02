from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import Any
from urllib.parse import urlencode

from stella.ask_ida import MSG_BUSY, ask_ida, ida_allowed_for
from stella.audio_pcmu import GEMINI_OUT_RATE, PcmToPcmu8k, Pcmu8kToPcm16k
from stella.config import Settings
from stella.energy_vad import EnergyVad, EnergyVadConfig, pcm_rms
from stella.store import CallJob
from stella.voice import (
    KICKOFF_TEXT,
    FarewellHangupWatch,
    TelnyxMediaGuard,
    build_instructions,
    hang_up_tool_openai,
    parse_hang_up_args,
)

logger = logging.getLogger(__name__)

# https://ai.google.dev/gemini-api/docs/models/gemini-3.8-live
# Live WebSocket get-started uses this id with responseModalities AUDIO.
# Confirmed on Simon's key via v1beta listModels (bidiGenerateContent).
DEFAULT_GEMINI_LIVE_MODEL = "gemini-3.8-live"
# Tried only if the primary model's setup handshake fails (not mid-call).
GEMINI_LIVE_SETUP_FALLBACKS = (
    "gemini-3.1-flash-live-preview",
    "gemini-2.5-flash-native-audio-preview-12-2025",
    "gemini-2.5-flash-native-audio-latest",
)
SETUP_COMPLETE_TIMEOUT_S = 8.0


def gemini_ws_url(settings: Settings) -> str:
    base = settings.gemini_realtime_url.rstrip("?")
    query = urlencode({"key": settings.gemini_api_key})
    return f"{base}?{query}"


def gemini_bare_model_id(model: str) -> str:
    return (model or "").strip().removeprefix("models/")


def gemini_model_id(settings: Settings) -> str:
    return gemini_bare_model_id(settings.gemini_live_model) or DEFAULT_GEMINI_LIVE_MODEL


def gemini_live_try_order(settings: Settings) -> list[str]:
    """Primary GEMINI_LIVE_MODEL, then documented Live fallbacks (no 09-2025)."""
    order: list[str] = []
    seen: set[str] = set()
    for raw in (gemini_model_id(settings), *GEMINI_LIVE_SETUP_FALLBACKS):
        model = gemini_bare_model_id(raw)
        if not model or model in seen:
            continue
        seen.add(model)
        order.append(model)
    return order


def gemini_model_name(settings: Settings, model: str | None = None) -> str:
    raw = gemini_bare_model_id(model) if model else gemini_model_id(settings)
    if not raw.startswith("models/"):
        raw = f"models/{raw}"
    return raw


def gemini_uses_thinking_level(model: str) -> bool:
    """3.1 Live takes thinkingLevel; 3.8 and 2.5 native-audio take thinkingBudget.

    The API rejects setup (close 1007) if both are set, and 3.8 rejects thinkingLevel
    ("Thinking level is not supported for this model").
    """
    raw = gemini_bare_model_id(model).lower()
    return raw.startswith("gemini-3") and not raw.startswith("gemini-3.8")


def gemini_uses_client_vad(settings: Settings) -> bool:
    return bool(settings.stella_client_vad)


def gemini_realtime_input_config(settings: Settings) -> dict[str, Any]:
    """Phone-tuned Gemini Live VAD (BidiGenerateContent realtimeInputConfig)."""
    if gemini_uses_client_vad(settings):
        # activityStart/activityEnd are only accepted when automatic VAD is off.
        return {
            "automaticActivityDetection": {"disabled": True},
            "activityHandling": settings.gemini_vad_activity_handling,
            "turnCoverage": settings.gemini_vad_turn_coverage,
        }
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


def gemini_generation_config(
    settings: Settings, model: str | None = None
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "responseModalities": ["AUDIO"],
        "speechConfig": {
            "voiceConfig": {
                "prebuiltVoiceConfig": {
                    "voiceName": settings.gemini_voice or "Aoede",
                }
            }
        },
    }
    budget = int(settings.gemini_thinking_budget)
    if budget >= 0:
        mid = model if model is not None else gemini_model_id(settings)
        if gemini_uses_thinking_level(mid):
            # 3.1 Live: thinkingLevel only (never together with thinkingBudget).
            cfg["thinkingConfig"] = {"thinkingLevel": "minimal" if budget == 0 else "low"}
        else:
            cfg["thinkingConfig"] = {"thinkingBudget": budget}
    return cfg


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


def ask_ida_tool_gemini() -> dict[str, Any]:
    return {
        "name": "frag_ida",
        "description": (
            "Fragt Simons persönliche Daten ab: Kalendertermine, Stundenplan und Ausfälle, "
            "Hausaufgaben, Mails (nur lesen), Smart-Home-Status. Nicht für Allgemeinwissen. "
            "Dauert einige Sekunden. Vorher kurz sagen, dass du nachschaust."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "frage": {
                    "type": "STRING",
                    "description": (
                        "Simons Frage im Wortlaut, nichts kürzen, hinzufügen oder erweitern. "
                        "Wörter wie 'noch', 'heute noch', 'letzte', 'neueste', 'ungelesen' "
                        "und der Zeitraum (z. B. 'morgen') müssen erhalten bleiben."
                    ),
                },
                "bestaetigt": {
                    "type": "BOOLEAN",
                    "description": (
                        "Nur true, wenn Simon eine zuvor vorgeschlagene Aktion ausdrücklich "
                        "mit Ja bestätigt hat. Sonst weglassen."
                    ),
                },
            },
            "required": ["frage"],
        },
    }


def gemini_tools(*, ask_ida: bool) -> list[dict[str, Any]]:
    tools = hang_up_tool_gemini()
    if ask_ida:
        tools["functionDeclarations"].append(ask_ida_tool_gemini())
    return [tools]


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
        self._gemini_aiter = None
        self._closed = False
        self._assistant_bits: list[str] = []
        self._up = Pcmu8kToPcm16k()
        self._down = PcmToPcmu8k(default_rate=GEMINI_OUT_RATE)
        self.provider = "gemini"
        self.guard = TelnyxMediaGuard()
        self._setup_complete = asyncio.Event()
        self._vad = EnergyVad(
            EnergyVadConfig(
                rms_threshold=int(settings.stella_client_vad_rms),
                min_speech_ms=int(settings.stella_client_vad_min_speech_ms),
                silence_ms=int(settings.stella_client_vad_silence_ms),
            )
        )
        self._farewell = FarewellHangupWatch(
            enabled=bool(settings.stella_farewell_hangup),
            delay_s=settings.stella_farewell_hangup_s,
            arm=self._arm_hangup_from_farewell,
        )
        self._t_setup_ms: float | None = None
        self._t_activity_end_ms: float | None = None
        self._logged_setup_audio = False
        self._logged_turn_audio = False
        self._active_model = gemini_model_id(settings)
        # frag_ida only on calls to the owner number (re-checked here, not just at dial).
        self._ask_ida = bool(job.allow_ida) and ida_allowed_for(settings, job.to_number)
        self._ida_task: asyncio.Task | None = None
        self._ida_cancel = asyncio.Event()

    async def attach_telnyx(self, telnyx_ws) -> None:
        self._telnyx_ws = telnyx_ws
        await self.connect()
        await self.run()

    async def connect(self) -> None:
        if not (self.settings.gemini_api_key or "").strip():
            raise RuntimeError("GEMINI_API_KEY is not set; cannot open Gemini Live session.")
        last_exc: Exception | None = None
        url = gemini_ws_url(self.settings)
        for model in gemini_live_try_order(self.settings):
            self._active_model = model
            try:
                self._gemini_ws = await self._gemini_connect(url, {})
                await self._gemini_ws.send(json.dumps({"setup": self._setup_payload()}))
                await asyncio.wait_for(
                    self._await_setup_complete(),
                    timeout=SETUP_COMPLETE_TIMEOUT_S,
                )
                if model != gemini_model_id(self.settings):
                    logger.warning("Gemini Live using fallback model %s", model)
                return
            except Exception as exc:
                last_exc = exc
                logger.warning("Gemini Live setup failed for %s: %s", model, exc)
                ws = self._gemini_ws
                self._gemini_ws = None
                self._gemini_aiter = None
                if ws is not None:
                    try:
                        await ws.close()
                    except Exception:
                        pass
        await self._send_telnyx_clear()
        raise RuntimeError(
            f"Could not open Gemini Live session: {last_exc}"
        ) from last_exc

    async def _await_setup_complete(self) -> None:
        assert self._gemini_ws is not None
        self._gemini_aiter = self._gemini_ws.__aiter__()
        while True:
            try:
                raw = await self._gemini_aiter.__anext__()
            except StopAsyncIteration as exc:
                raise RuntimeError("Gemini Live closed before setupComplete") from exc
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if event.get("error"):
                raise RuntimeError(f"Gemini Live setup error: {event['error']}")
            if "setupComplete" in event:
                self._setup_complete.set()
                self._t_setup_ms = time.monotonic() * 1000
                self._latency("setup_complete")
                return

    async def _pump_gemini(self) -> None:
        assert self._gemini_ws is not None
        agen = self._gemini_aiter or self._gemini_ws.__aiter__()
        self._gemini_aiter = agen
        try:
            while True:
                if self._closed:
                    break
                try:
                    raw = await agen.__anext__()
                except StopAsyncIteration:
                    break
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if "setupComplete" in event:
                    self._setup_complete.set()
                    if self._t_setup_ms is None:
                        self._t_setup_ms = time.monotonic() * 1000
                    self._latency("setup_complete")
                    continue
                await self._handle_gemini_event(event)
        finally:
            await self.close()

    def _setup_payload(self) -> dict[str, Any]:
        model = self._active_model
        return {
            "model": gemini_model_name(self.settings, model),
            "generationConfig": gemini_generation_config(self.settings, model),
            "systemInstruction": {
                "parts": [{"text": build_instructions(self.job, ask_ida=self._ask_ida)}]
            },
            "tools": gemini_tools(ask_ida=self._ask_ida),
            "outputAudioTranscription": {},
            "inputAudioTranscription": {},
            "realtimeInputConfig": gemini_realtime_input_config(self.settings),
        }

    def _latency(self, kind: str, **fields: Any) -> None:
        if not self.settings.stella_latency_log:
            return
        extra = " ".join(f"{k}={v}" for k, v in fields.items())
        logger.info("stella_latency %s %s", kind, extra)

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
                        await self._forward_inbound_pcm(pcm)
                elif event in {"stop", "closed"}:
                    break
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
                    sent = await self.guard.send_outbound_pcmu(self._telnyx_ws, payload)
                    if sent:
                        self._note_first_outbound()
            # part.text is often model scratch / thinking, not spoken audio.
            # Spoken text is outputTranscription only.
        out_tx = server.get("outputTranscription") or {}
        piece = out_tx.get("text") or ""
        if piece:
            self._assistant_bits.append(piece)
            self._transcript(piece)
            self._farewell.note_assistant_text(piece)

        tool_call = event.get("toolCall") or {}
        for fc in tool_call.get("functionCalls") or []:
            name = fc.get("name")
            if name == "frag_ida" and self._ask_ida:
                self._start_ask_ida(fc)
                continue
            if name != "hang_up":
                continue
            outcome = parse_hang_up_args(fc.get("args") or fc.get("arguments") or {})
            self._outcome(outcome)
            await self._send_function_response(fc.get("id"), "hang_up", {"result": "hanging up"})
            self.guard.arm_hangup(self._deferred_hangup)
            return

    async def _send_function_response(self, call_id, name: str, response: dict[str, Any]) -> None:
        if self._closed or self._gemini_ws is None:
            return
        await self._gemini_ws.send(
            json.dumps(
                {
                    "toolResponse": {
                        "functionResponses": [
                            {"id": call_id, "name": name, "response": response}
                        ]
                    }
                }
            )
        )

    def _start_ask_ida(self, fc: dict[str, Any]) -> None:
        """Run frag_ida in a task so audio, VAD and the hangup guard keep running."""
        if self._ida_task is not None and not self._ida_task.done():
            asyncio.create_task(
                self._send_function_response(fc.get("id"), "frag_ida", {"error": MSG_BUSY})
            )
            return
        self._ida_task = asyncio.create_task(self._run_ask_ida(fc))

    async def _run_ask_ida(self, fc: dict[str, Any]) -> None:
        args = fc.get("args") or fc.get("arguments") or {}
        question = str(args.get("frage") or "").strip()
        confirmed = args.get("bestaetigt") is True
        answer = await ask_ida(
            self.settings, question, confirmed=confirmed, cancel_event=self._ida_cancel
        )
        try:
            await self._send_function_response(fc.get("id"), "frag_ida", {"result": answer})
        except Exception:
            logger.exception("could not send frag_ida result")

    async def _forward_inbound_pcm(self, pcm: bytes) -> None:
        assert self._gemini_ws is not None
        client_vad = gemini_uses_client_vad(self.settings)
        events = self._vad.feed(pcm) if client_vad else []
        if not client_vad and pcm_rms(pcm) >= self.settings.stella_client_vad_rms:
            self._farewell.note_user_speech()
        for ev in events:
            if ev == "start":
                self._farewell.note_user_speech()
                await self._gemini_ws.send(json.dumps({"realtimeInput": {"activityStart": {}}}))
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
        for ev in events:
            if ev == "end":
                self._t_activity_end_ms = time.monotonic() * 1000
                self._logged_turn_audio = False
                self._latency("activity_end")
                await self._gemini_ws.send(json.dumps({"realtimeInput": {"activityEnd": {}}}))

    def _note_first_outbound(self) -> None:
        now = time.monotonic() * 1000
        if not self._logged_setup_audio and self._t_setup_ms is not None:
            self._logged_setup_audio = True
            self._latency("setup_to_first_audio_ms", ms=round(now - self._t_setup_ms))
        if not self._logged_turn_audio and self._t_activity_end_ms is not None:
            self._logged_turn_audio = True
            self._latency(
                "silence_to_first_audio_ms",
                ms=round(now - self._t_activity_end_ms),
            )

    def _arm_hangup_from_farewell(self, outcome: str) -> None:
        if self._closed:
            return
        self._outcome(outcome)
        self.guard.arm_hangup(self._deferred_hangup)

    async def _deferred_hangup(self) -> None:
        try:
            await self.guard.telnyx_started.wait()
            for leftover in self._down.flush_frames_b64():
                sent = await self.guard.send_outbound_pcmu(self._telnyx_ws, leftover)
                if sent:
                    self._note_first_outbound()
            await self.guard.hangup_after_audio(self._hangup, lambda: self._closed)
        finally:
            await self.close()

    async def _on_barge_in(self) -> None:
        """Stop queued Telnyx playout when Gemini reports the user interrupted."""
        self._down = PcmToPcmu8k(default_rate=GEMINI_OUT_RATE)
        await self._send_telnyx_clear()

    async def _send_telnyx_clear(self) -> None:
        self.guard.clear_playout()
        if self._telnyx_ws:
            try:
                await self._telnyx_ws.send_text(json.dumps({"event": "clear"}))
            except Exception:
                pass

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._ida_cancel.set()
        if self._ida_task is not None and not self._ida_task.done():
            self._ida_task.cancel()
        self._setup_complete.set()
        self.guard.mark_started()
        if self._gemini_ws is None:
            return
        try:
            await self._gemini_ws.close()
        except Exception:
            pass
