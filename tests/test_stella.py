from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import base64
from datetime import datetime, timezone

from stella.config import Settings
from stella.errors import StellaError, normalize_e164
from stella.http_app import create_app
from stella.jobs import JobService
from stella.mcp_app import build_mcp
from stella.store import JobStore
from stella.telnyx_client import TelnyxClient
from stella.audio_pcmu import PcmToPcmu8k, Pcmu8kToPcm16k, parse_pcm_rate
from stella.gemini_voice import (
    DEFAULT_GEMINI_LIVE_MODEL,
    GEMINI_LIVE_SETUP_FALLBACKS,
    GeminiVoiceSession,
    gemini_generation_config,
    gemini_live_try_order,
    gemini_model_name,
    gemini_realtime_input_config,
    gemini_uses_thinking_level,
    gemini_ws_url,
)
from stella.voice import TelnyxMediaGuard, build_instructions, job_wants_german, start_voice_bridge
from stella.webhooks import verify_telnyx_signature


class FakeTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        self.calls.append((method, url, kwargs))
        if method == "POST" and url.endswith("/calls"):
            body = {
                "data": {
                    "call_control_id": "cc-test-1",
                    "call_leg_id": "leg-1",
                    "call_session_id": "sess-1",
                }
            }
            return httpx.Response(200, json=body)
        if "hangup" in url:
            return httpx.Response(200, json={"data": {"result": "ok"}})
        return httpx.Response(404, json={"errors": [{"detail": "not mocked"}]})


def make_settings(tmp_path, **kwargs) -> Settings:
    defaults = dict(
        stella_db_path=str(tmp_path / "stella.db"),
        gemini_api_key="test-gemini",
        telnyx_api_key="test-telnyx",
        telnyx_connection_id="conn-1",
        telnyx_from_number="+15551234567",
        stella_public_base_url="https://stella.example",
        stella_skip_webhook_verify=True,
    )
    defaults.update(kwargs)
    return Settings(**defaults)




def test_normalize_e164():
    assert normalize_e164("+1 650 253 0000") == "+16502530000"
    with pytest.raises(StellaError):
        normalize_e164("not-a-number")


def test_place_call_dials_telnyx(tmp_path):
    settings = make_settings(tmp_path)
    store = JobStore(settings.stella_db_path)
    transport = FakeTransport()
    telnyx = TelnyxClient(settings, transport=transport)
    svc = JobService(settings, store, telnyx)
    job = svc.place_call(
        to="+14155552671",
        brief="Reserve a table tomorrow 19:00 for 2, name Franz.",
        speak_to="the restaurant",
    )
    assert job.status == "dialing"
    assert job.voice_provider == "gemini"
    assert job.telnyx_call_control_id == "cc-test-1"
    method, url, kwargs = transport.calls[0]
    assert method == "POST"
    assert url.endswith("/calls")
    payload = kwargs["json"]
    assert payload["to"] == "+14155552671"
    assert payload["stream_url"].endswith(f"/media/{job.id}")
    assert payload["webhook_url"].endswith("/webhooks/telnyx")


def test_place_call_fails_without_telnyx_config(tmp_path):
    settings = make_settings(tmp_path, telnyx_api_key="")
    store = JobStore(settings.stella_db_path)
    svc = JobService(
        settings,
        store,
        TelnyxClient(settings, FakeTransport()),
    )
    with pytest.raises(StellaError) as ei:
        svc.place_call(to="+14155552671", brief="hello")
    assert ei.value.code == "telnyx_config_missing"
    assert "TELNYX_API_KEY" in ei.value.message




def test_place_call_fails_without_gemini_key(tmp_path):
    settings = make_settings(tmp_path, gemini_api_key="")
    store = JobStore(settings.stella_db_path)
    transport = FakeTransport()
    svc = JobService(settings, store, TelnyxClient(settings, transport))
    with pytest.raises(StellaError) as ei:
        svc.place_call(to="+14155552671", brief="hello")
    assert ei.value.code == "gemini_key_missing"
    assert "GEMINI_API_KEY" in ei.value.message
    assert transport.calls == []


def test_place_call_uses_gemini(tmp_path):
    settings = make_settings(tmp_path)
    store = JobStore(settings.stella_db_path)
    svc = JobService(
        settings,
        store,
        TelnyxClient(settings, FakeTransport()),
    )
    job = svc.place_call(to="+14155552671", brief="hello")
    assert job.status == "dialing"
    assert job.voice_provider == "gemini"


def test_webhook_updates_status(tmp_path):
    settings = make_settings(tmp_path)
    store = JobStore(settings.stella_db_path)
    svc = JobService(
        settings,
        store,
        TelnyxClient(settings, FakeTransport()),
    )
    job = svc.place_call(to="+14155552671", brief="Ask if Tuesday 10:00 still works.")
    svc.handle_telnyx_event(
        {
            "data": {
                "event_type": "call.answered",
                "payload": {"call_control_id": "cc-test-1"},
            }
        }
    )
    assert store.get(job.id).status == "in_progress"
    svc.handle_telnyx_event(
        {
            "data": {
                "event_type": "call.hangup",
                "payload": {"call_control_id": "cc-test-1", "hangup_cause": "normal_clearing"},
            }
        }
    )
    done = store.get(job.id)
    assert done.status == "completed"
    assert "normal_clearing" in done.outcome


def test_instructions_contain_only_job_context(tmp_path):
    settings = make_settings(tmp_path)
    store = JobStore(settings.stella_db_path)
    job = store.create(
        kind="call",
        to_number="+14155552671",
        brief="Book a table.",
        context="Name Franz, 2 people, 19:00.",
        speak_to="host",
    )
    text = build_instructions(job)
    assert "Book a table." in text
    assert "Franz" in text
    assert "Ida" in text  # explicit isolation note
    assert "Not Ida" in text or "NOT Ida" in text
    assert "hang_up" in text
    assert "never narrate reasoning" in text.lower()
    assert "ich verstehe nicht" in text.lower()
    assert "Speak German" not in text
    assert not job_wants_german(job)


def test_mcp_tools_registered(tmp_path):
    settings = make_settings(tmp_path)
    store = JobStore(settings.stella_db_path)
    svc = JobService(settings, store, TelnyxClient(settings, FakeTransport()))
    mcp = build_mcp(svc)
    names = {t.name for t in mcp._tool_manager.list_tools()}
    assert names == {"stella_call", "stella_call_status", "stella_briefing_call"}


def test_health_and_call_http(tmp_path):
    settings = make_settings(tmp_path, gemini_api_key="")
    app = create_app(settings)
    from fastapi.testclient import TestClient

    client = TestClient(app)
    h = client.get("/health")
    assert h.status_code == 200
    body = h.json()
    assert body["voice_provider"]["primary"] == "gemini"
    assert body["voice_provider"]["gemini"] == "none"
    assert body["gemini_configured"] is False
    # webhook without skip would 500; skip is on
    r = client.post(
        "/webhooks/telnyx",
        json={"data": {"event_type": "call.initiated", "payload": {}}},
    )
    assert r.status_code == 200


def test_ed25519_webhook_verify():
    key = Ed25519PrivateKey.generate()
    pub = key.public_key().public_bytes_raw()
    ts = str(int(datetime.now(timezone.utc).timestamp()))
    body = b'{"ok":true}'
    signed = f"{ts}|".encode() + body
    sig = key.sign(signed)
    verify_telnyx_signature(
        payload=body,
        timestamp=ts,
        signature_b64=base64.b64encode(sig).decode(),
        public_key_b64=base64.b64encode(pub).decode(),
    )
    with pytest.raises(StellaError):
        verify_telnyx_signature(
            payload=b"tampered",
            timestamp=ts,
            signature_b64=base64.b64encode(sig).decode(),
            public_key_b64=base64.b64encode(pub).decode(),
        )


class FakeProviderWS:
    def __init__(self, messages: list[str] | None = None, *, hold_open: bool = False) -> None:
        self.sent: list[Any] = []
        self._messages = list(messages or [])
        self.closed = False
        self._hold_open = hold_open

    async def send(self, data: str) -> None:
        self.sent.append(json.loads(data) if isinstance(data, str) else data)

    async def close(self) -> None:
        self.closed = True
        self._hold_open = False
        self._messages = []

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        if self._messages:
            return self._messages.pop(0)
        if self._hold_open and not self.closed:
            await asyncio.sleep(0.05)
            if self._messages:
                return self._messages.pop(0)
            if self._hold_open and not self.closed:
                await asyncio.sleep(0.05)
        raise StopAsyncIteration


class FakeTelnyxWS:
    def __init__(self, texts: list[str] | None = None, *, linger_s: float = 0.0) -> None:
        self.sent: list[str] = []
        self._texts = list(texts or [])
        self._linger_s = linger_s

    async def iter_text(self):
        for t in self._texts:
            yield t
        if self._linger_s:
            await asyncio.sleep(self._linger_s)

    async def send_text(self, data: str) -> None:
        self.sent.append(data)




@pytest.mark.asyncio
async def test_start_voice_bridge_runs_gemini(tmp_path):
    settings = make_settings(tmp_path, gemini_api_key="g-key")
    store = JobStore(settings.stella_db_path)
    job = store.create(kind="call", to_number="+14155552671", brief="Book a table.")
    gemini_ws = FakeProviderWS([json.dumps({"setupComplete": {}})])
    async def gemini_connect(url, headers):
        assert "generativelanguage.googleapis.com" in url
        assert "g-key" in url
        return gemini_ws

    provider = await start_voice_bridge(
        settings=settings,
        job=job,
        telnyx_ws=FakeTelnyxWS(),
        hangup_cb=lambda: None,
        transcript_cb=lambda c: None,
        outcome_cb=lambda t: None,
        ws_connect=gemini_connect,
    )
    assert provider == "gemini"
    voice = gemini_ws.sent[0]["setup"]["generationConfig"]["speechConfig"]["voiceConfig"][
        "prebuiltVoiceConfig"
    ]["voiceName"]
    assert voice == "Aoede"






@pytest.mark.asyncio
async def test_start_voice_bridge_second_media_socket(tmp_path):
    settings = make_settings(tmp_path, gemini_api_key="g-key")
    store = JobStore(settings.stella_db_path)
    job = store.create(kind="call", to_number="+14155552671", brief="Book a table.")
    async def gemini_connect(url, headers):
        return FakeProviderWS([json.dumps({"setupComplete": {}})])

    for _ in range(2):
        provider = await start_voice_bridge(
            settings=settings,
            job=job,
            telnyx_ws=FakeTelnyxWS(),
                hangup_cb=lambda: None,
            transcript_cb=lambda c: None,
            outcome_cb=lambda t: None,
            ws_connect=gemini_connect,
        )
        assert provider == "gemini"


@pytest.mark.asyncio
async def test_start_voice_bridge_errors_when_gemini_key_missing(tmp_path):
    settings = make_settings(tmp_path, gemini_api_key="")
    store = JobStore(settings.stella_db_path)
    job = store.create(kind="call", to_number="+14155552671", brief="Book a table.")

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        await start_voice_bridge(
            settings=settings,
            job=job,
            telnyx_ws=FakeTelnyxWS(),
            ws_connect=lambda u, h: None,
            hangup_cb=lambda: None,
            transcript_cb=lambda c: None,
            outcome_cb=lambda t: None,
        )


@pytest.mark.asyncio
async def test_gemini_hang_up_and_audio_out(tmp_path):
    settings = make_settings(tmp_path, gemini_api_key="g-key")
    store = JobStore(settings.stella_db_path)
    job = store.create(kind="call", to_number="+14155552671", brief="Book a table.")
    telnyx = FakeTelnyxWS()
    outcomes: list[str] = []
    hung: list[bool] = []

    async def hangup():
        hung.append(True)

    session = GeminiVoiceSession(
        settings=settings,
        job=job,
        gemini_connect=lambda u, h: None,
        hangup_cb=hangup,
        transcript_cb=lambda c: None,
        outcome_cb=outcomes.append,
    )
    session._telnyx_ws = telnyx
    session.guard.mark_started()
    session.guard.hangup_playout_pad_s = 0.0
    session.guard.hangup_wait_audio_s = 0.2

    pcm24 = b"\x00\x10" * 480  # 20 ms at 24 kHz, 16-bit LE
    payload = base64.b64encode(pcm24).decode("ascii")
    await session._handle_gemini_event(
        {
            "serverContent": {
                "modelTurn": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "audio/pcm;rate=24000",
                                "data": payload,
                            }
                        }
                    ]
                },
                "outputTranscription": {"text": "hello"},
            }
        }
    )
    assert telnyx.sent
    media = json.loads(telnyx.sent[0])
    assert media["event"] == "media"
    raw_ulaw = base64.b64decode(media["media"]["payload"])
    assert len(raw_ulaw) == 160

    class DummyGemini:
        def __init__(self):
            self.sent = []

        async def send(self, data):
            self.sent.append(data)

        async def close(self):
            pass

    session._gemini_ws = DummyGemini()
    await session._handle_gemini_event(
        {
            "toolCall": {
                "functionCalls": [
                    {"id": "1", "name": "hang_up", "args": {"outcome": "Reserved."}}
                ]
            }
        }
    )
    assert outcomes == ["Reserved."]
    assert session.guard._hangup_task is not None
    await session.guard._hangup_task
    assert hung == [True]


def test_pcmu_roundtrip_and_mime_rate():
    assert parse_pcm_rate("audio/pcm;rate=24000", 16000) == 24000
    up = Pcmu8kToPcm16k()
    down = PcmToPcmu8k(default_rate=16000)
    silence = base64.b64encode(b"\xff" * 160).decode("ascii")
    pcm = up.convert_b64(silence)
    assert len(pcm) == 640
    back = down.convert_b64(base64.b64encode(pcm).decode("ascii"), "audio/pcm;rate=16000")
    assert back
    assert len(base64.b64decode(back)) == 160


def test_pcmu_24k_emits_aligned_20ms_frames():
    from stella.audio_pcmu import TELNYX_FRAME_BYTES, downsample_integer, pack_pcm16le, unpack_pcm16le

    # 45 ms of 24 kHz PCM (1080 samples) -> 15 ms leftover at 8 kHz until next chunk.
    samples = [1000, -1000, 500] * 360  # 1080 samples = 45 ms @ 24 kHz
    pcm = pack_pcm16le(samples)
    down = PcmToPcmu8k(default_rate=24000)
    frames = down.convert_frames_b64(base64.b64encode(pcm).decode("ascii"), "audio/pcm;rate=24000")
    assert len(frames) == 2  # 40 ms; 5 ms held
    for f in frames:
        assert len(base64.b64decode(f)) == TELNYX_FRAME_BYTES

    # Odd byte + remaining samples should complete another frame after more data.
    extra = pack_pcm16le([1000, -1000, 500] * 120)  # +15 ms @ 8 kHz with 5 ms leftover -> 1 frame
    more = down.convert_frames_b64(base64.b64encode(extra).decode("ascii"), "audio/pcm;rate=24000")
    assert len(more) == 1
    assert len(base64.b64decode(more[0])) == TELNYX_FRAME_BYTES
    short = pack_pcm16le([0] * 3)  # 1 sample @ 8 kHz, held
    assert down.convert_frames_b64(base64.b64encode(short).decode("ascii"), "audio/pcm;rate=24000") == []
    tail = pack_pcm16le([0] * 477)  # 159 more @ 8 kHz -> one frame
    flushed = down.convert_frames_b64(base64.b64encode(tail).decode("ascii"), "audio/pcm;rate=24000")
    assert len(flushed) == 1
    assert all(len(base64.b64decode(f)) == TELNYX_FRAME_BYTES for f in flushed)

    averaged, leftover = downsample_integer([3, 6, 9, 12, 15], 3)
    assert averaged == [6]
    assert leftover == [12, 15]
    assert unpack_pcm16le(b"\x01\x00\xff\xff") == [1, -1]

    odd = PcmToPcmu8k()
    chunk = pack_pcm16le([0] * 480) + b"\x01"  # trailing unpaired byte
    frames = odd.convert_frames_b64(base64.b64encode(chunk).decode("ascii"), "audio/pcm;rate=24000")
    assert len(frames) == 1
    # leftover byte + rest of a 16-bit sample + 479 samples @24k = 480 samples -> 160 ulaw
    rest = b"\x00" + pack_pcm16le([0] * 479)
    frames2 = odd.convert_frames_b64(base64.b64encode(rest).decode("ascii"), "audio/pcm;rate=24000")
    assert len(frames2) == 1
    assert len(base64.b64decode(frames2[0])) == TELNYX_FRAME_BYTES


def test_gemini_voice_default_is_female(tmp_path):
    settings = make_settings(tmp_path)
    assert settings.gemini_voice == "Aoede"


def test_dial_keeps_inbound_track_for_bidirectional(tmp_path):
    settings = make_settings(tmp_path)
    store = JobStore(settings.stella_db_path)
    transport = FakeTransport()
    telnyx = TelnyxClient(settings, transport=transport)
    svc = JobService(settings, store, telnyx)
    svc.place_call(to="+14155552671", brief="hello")
    payload = transport.calls[0][2]["json"]
    assert payload["stream_track"] == "inbound_track"
    assert payload["stream_bidirectional_mode"] == "rtp"
    assert payload["stream_bidirectional_codec"] == "PCMU"
    assert payload["stream_bidirectional_sampling_rate"] == 8000


def test_gemini_url_hides_nothing_but_uses_query_key(tmp_path):
    settings = make_settings(tmp_path, gemini_api_key="secret-key")
    url = gemini_ws_url(settings)
    assert "secret-key" in url
    assert gemini_model_name(settings) == f"models/{DEFAULT_GEMINI_LIVE_MODEL}"
    assert DEFAULT_GEMINI_LIVE_MODEL == "gemini-3.8-live"
    assert "preview-09-2025" not in DEFAULT_GEMINI_LIVE_MODEL














@pytest.mark.asyncio
async def test_gemini_drops_audio_and_defers_hangup_before_telnyx_start(tmp_path):
    settings = make_settings(tmp_path, gemini_api_key="g-key")
    store = JobStore(settings.stella_db_path)
    job = store.create(kind="call", to_number="+14155552671", brief="Sag Hallo.")
    telnyx = FakeTelnyxWS()
    hung: list[bool] = []

    async def hangup():
        hung.append(True)

    session = GeminiVoiceSession(
        settings=settings,
        job=job,
        gemini_connect=lambda u, h: None,
        hangup_cb=hangup,
        transcript_cb=lambda c: None,
        outcome_cb=lambda t: None,
    )
    session._telnyx_ws = telnyx
    session.guard.hangup_wait_audio_s = 0.15
    session.guard.hangup_playout_pad_s = 0.0

    class DummyGemini:
        def __init__(self):
            self.sent = []

        async def send(self, data):
            self.sent.append(data)

        async def close(self):
            pass

    session._gemini_ws = DummyGemini()
    pcm24 = b"\x00\x10" * 480
    payload = base64.b64encode(pcm24).decode("ascii")
    await session._handle_gemini_event(
        {
            "serverContent": {
                "modelTurn": {
                    "parts": [
                        {"inlineData": {"mimeType": "audio/pcm;rate=24000", "data": payload}}
                    ]
                }
            }
        }
    )
    assert telnyx.sent == []

    await session._handle_gemini_event(
        {
            "toolCall": {
                "functionCalls": [
                    {"id": "1", "name": "hang_up", "args": {"outcome": "Test erfolgreich"}}
                ]
            }
        }
    )
    await asyncio.sleep(0.02)
    assert hung == []
    session.guard.mark_started()
    await session._handle_gemini_event(
        {
            "serverContent": {
                "modelTurn": {
                    "parts": [
                        {"inlineData": {"mimeType": "audio/pcm;rate=24000", "data": payload}}
                    ]
                }
            }
        }
    )
    assert telnyx.sent
    assert session.guard._hangup_task is not None
    await session.guard._hangup_task
    assert hung == [True]


@pytest.mark.asyncio
async def test_gemini_kickoff_waits_for_stream_start(tmp_path):
    settings = make_settings(tmp_path, gemini_api_key="g-key")
    store = JobStore(settings.stella_db_path)
    job = store.create(kind="call", to_number="+14155552671", brief="Hallo Simon")
    gemini_ws = FakeProviderWS([json.dumps({"setupComplete": {}})], hold_open=True)
    telnyx = FakeTelnyxWS(texts=[json.dumps({"event": "connected"})], linger_s=0.15)

    async def gemini_connect(url, headers):
        return gemini_ws

    await start_voice_bridge(
        settings=settings,
        job=job,
        telnyx_ws=telnyx,
        hangup_cb=lambda: None,
        transcript_cb=lambda c: None,
        outcome_cb=lambda t: None,
        ws_connect=gemini_connect,
    )
    assert not any(isinstance(m, dict) and "clientContent" in m for m in gemini_ws.sent)


@pytest.mark.asyncio
async def test_gemini_kickoff_after_telnyx_start(tmp_path):
    settings = make_settings(tmp_path, gemini_api_key="g-key")
    store = JobStore(settings.stella_db_path)
    job = store.create(kind="call", to_number="+14155552671", brief="Hallo Simon")
    gemini_ws = FakeProviderWS([json.dumps({"setupComplete": {}})], hold_open=True)
    telnyx = FakeTelnyxWS(
        texts=[json.dumps({"event": "start", "start": {"mediaFormat": {}}})],
        linger_s=0.15,
    )

    async def gemini_connect(url, headers):
        return gemini_ws

    await start_voice_bridge(
        settings=settings,
        job=job,
        telnyx_ws=telnyx,
        hangup_cb=lambda: None,
        transcript_cb=lambda c: None,
        outcome_cb=lambda t: None,
        ws_connect=gemini_connect,
    )
    assert any(isinstance(m, dict) and "clientContent" in m for m in gemini_ws.sent)






def test_gemini_setup_includes_phone_vad(tmp_path):
    settings = make_settings(
        tmp_path,
        gemini_api_key="g-key",
        stella_client_vad=False,
        gemini_vad_silence_duration_ms=250,
        gemini_vad_prefix_padding_ms=15,
    )
    store = JobStore(settings.stella_db_path)
    job = store.create(kind="call", to_number="+14155552671", brief="Book a table.")
    session = GeminiVoiceSession(
        settings=settings,
        job=job,
        gemini_connect=lambda u, h: None,
        hangup_cb=lambda: None,
        transcript_cb=lambda c: None,
        outcome_cb=lambda t: None,
    )
    payload = session._setup_payload()
    vad = payload["realtimeInputConfig"]
    aad = vad["automaticActivityDetection"]
    assert aad["disabled"] is False
    assert aad["startOfSpeechSensitivity"] == "START_SENSITIVITY_HIGH"
    assert aad["endOfSpeechSensitivity"] == "END_SENSITIVITY_HIGH"
    assert aad["silenceDurationMs"] == 250
    assert aad["prefixPaddingMs"] == 15
    assert vad["activityHandling"] == "START_OF_ACTIVITY_INTERRUPTS"
    assert vad["turnCoverage"] == "TURN_INCLUDES_ONLY_ACTIVITY"
    cfg = gemini_realtime_input_config(settings)
    assert cfg == vad
    assert payload["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert payload["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}
    assert "call hang_up" in payload["systemInstruction"]["parts"][0]["text"]
    assert "ich verstehe nicht" in payload["systemInstruction"]["parts"][0]["text"]


def test_instructions_german_when_brief_is_de(tmp_path):
    settings = make_settings(tmp_path)
    store = JobStore(settings.stella_db_path)
    job = store.create(
        kind="call",
        to_number="+4915112345678",
        brief="Reserviere einen Tisch für morgen 19 Uhr.",
        speak_to="Simon",
    )
    assert job_wants_german(job)
    text = build_instructions(job)
    assert "Speak German" in text
    assert "ich verstehe nicht" in text


@pytest.mark.asyncio
async def test_gemini_transcript_skips_thinking_part_text(tmp_path):
    settings = make_settings(tmp_path, gemini_api_key="g-key")
    store = JobStore(settings.stella_db_path)
    job = store.create(kind="call", to_number="+14155552671", brief="Hi")
    chunks: list[str] = []
    session = GeminiVoiceSession(
        settings=settings,
        job=job,
        gemini_connect=lambda u, h: None,
        hangup_cb=lambda: None,
        transcript_cb=chunks.append,
        outcome_cb=lambda t: None,
    )
    await session._handle_gemini_event(
        {
            "serverContent": {
                "modelTurn": {
                    "parts": [{"text": "**Initiating Pleasantries** I will greet them."}]
                },
                "outputTranscription": {"text": "Hallo, hier ist Stella."},
            }
        }
    )
    assert chunks == ["Hallo, hier ist Stella."]
    assert session._assistant_bits == ["Hallo, hier ist Stella."]


@pytest.mark.asyncio
async def test_gemini_interrupted_sends_telnyx_clear(tmp_path):
    settings = make_settings(tmp_path, gemini_api_key="g-key")
    store = JobStore(settings.stella_db_path)
    job = store.create(kind="call", to_number="+14155552671", brief="Hi")
    telnyx = FakeTelnyxWS()
    session = GeminiVoiceSession(
        settings=settings,
        job=job,
        gemini_connect=lambda u, h: None,
        hangup_cb=lambda: None,
        transcript_cb=lambda c: None,
        outcome_cb=lambda t: None,
    )
    session._telnyx_ws = telnyx
    session.guard.mark_started()
    await session._handle_gemini_event({"serverContent": {"interrupted": True}})
    assert any(json.loads(s).get("event") == "clear" for s in telnyx.sent)


def _pcm16_frame(amplitude: int, frames: int = 1) -> bytes:
    from stella.audio_pcmu import pack_pcm16le

    return pack_pcm16le([amplitude] * (320 * frames))


def test_energy_vad_emits_start_then_end_after_silence():
    from stella.energy_vad import EnergyVad, EnergyVadConfig

    vad = EnergyVad(EnergyVadConfig(rms_threshold=500, min_speech_ms=60, silence_ms=80))
    events: list[str] = []
    for _ in range(4):
        events.extend(vad.feed(_pcm16_frame(8000)))
    assert events == ["start"]
    events = []
    for _ in range(5):
        events.extend(vad.feed(_pcm16_frame(0)))
    assert events == ["end"]


def test_gemini_setup_client_vad_disables_automatic(tmp_path):
    settings = make_settings(tmp_path, gemini_api_key="g-key", stella_client_vad=True)
    store = JobStore(settings.stella_db_path)
    job = store.create(kind="call", to_number="+14155552671", brief="Hi")
    session = GeminiVoiceSession(
        settings=settings,
        job=job,
        gemini_connect=lambda u, h: None,
        hangup_cb=lambda: None,
        transcript_cb=lambda c: None,
        outcome_cb=lambda t: None,
    )
    payload = session._setup_payload()
    aad = payload["realtimeInputConfig"]["automaticActivityDetection"]
    assert aad == {"disabled": True}
    assert payload["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}


@pytest.mark.asyncio
async def test_gemini_client_vad_sends_activity_end(tmp_path):
    settings = make_settings(
        tmp_path,
        gemini_api_key="g-key",
        stella_client_vad=True,
        stella_client_vad_min_speech_ms=20,
        stella_client_vad_silence_ms=40,
        stella_client_vad_rms=200,
    )
    store = JobStore(settings.stella_db_path)
    job = store.create(kind="call", to_number="+14155552671", brief="Hi")
    session = GeminiVoiceSession(
        settings=settings,
        job=job,
        gemini_connect=lambda u, h: None,
        hangup_cb=lambda: None,
        transcript_cb=lambda c: None,
        outcome_cb=lambda t: None,
    )

    class DummyGemini:
        def __init__(self):
            self.sent = []

        async def send(self, data):
            self.sent.append(json.loads(data) if isinstance(data, str) else data)

    session._gemini_ws = DummyGemini()
    await session._forward_inbound_pcm(_pcm16_frame(9000, 2))
    await session._forward_inbound_pcm(_pcm16_frame(0, 4))
    kinds = []
    for msg in session._gemini_ws.sent:
        ri = msg.get("realtimeInput") or {}
        if "activityStart" in ri:
            kinds.append("start")
        elif "activityEnd" in ri:
            kinds.append("end")
        elif "audio" in ri:
            kinds.append("audio")
    assert "start" in kinds and "end" in kinds
    assert kinds.index("start") < kinds.index("end")
    assert any(k == "audio" for k in kinds)


@pytest.mark.asyncio
async def test_gemini_farewell_hangup_backup(tmp_path):
    settings = make_settings(
        tmp_path,
        gemini_api_key="g-key",
        stella_farewell_hangup=True,
        stella_farewell_hangup_s=0.05,
    )
    store = JobStore(settings.stella_db_path)
    job = store.create(kind="call", to_number="+14155552671", brief="Hi")
    hung: list[bool] = []
    outcomes: list[str] = []

    async def hangup():
        hung.append(True)

    session = GeminiVoiceSession(
        settings=settings,
        job=job,
        gemini_connect=lambda u, h: None,
        hangup_cb=hangup,
        transcript_cb=lambda c: None,
        outcome_cb=outcomes.append,
    )
    session.guard.mark_started()
    session.guard.hangup_playout_pad_s = 0.0
    session.guard.hangup_wait_audio_s = 0.05
    pcm24 = b"\x00\x10" * 480
    payload = base64.b64encode(pcm24).decode("ascii")
    session._telnyx_ws = FakeTelnyxWS()
    await session._handle_gemini_event(
        {
            "serverContent": {
                "modelTurn": {
                    "parts": [{"inlineData": {"mimeType": "audio/pcm;rate=24000", "data": payload}}]
                },
                "outputTranscription": {"text": "Tschüss, auf Wiederhören!"},
            }
        }
    )
    await asyncio.sleep(0.2)
    assert session.guard._hangup_task is not None
    await session.guard._hangup_task
    assert hung == [True]
    assert outcomes and "farewell" in outcomes[0].lower()


def test_looks_like_farewell():
    from stella.voice import looks_like_farewell

    assert looks_like_farewell("Ok, tschüss!")
    assert looks_like_farewell("Goodbye")
    assert not looks_like_farewell("What time works for you?")


def test_hang_up_tool_requires_action():
    from stella.voice import hang_up_tool_openai

    desc = hang_up_tool_openai()["description"]
    assert "hang up" in desc.lower() or "End the live" in desc


def test_gemini_model_override_and_thinking_config(tmp_path):
    live = make_settings(tmp_path, gemini_live_model="gemini-3.8-live")
    assert gemini_model_name(live) == "models/gemini-3.8-live"
    assert not gemini_uses_thinking_level("gemini-3.8-live")
    assert gemini_generation_config(live)["thinkingConfig"] == {"thinkingBudget": 0}
    assert gemini_uses_thinking_level("gemini-3.1-flash-live-preview")
    assert gemini_generation_config(live, "gemini-3.1-flash-live-preview")["thinkingConfig"] == {
        "thinkingLevel": "minimal"
    }

    pinned = make_settings(
        tmp_path,
        gemini_live_model="gemini-2.5-flash-native-audio-preview-12-2025",
    )
    assert gemini_model_name(pinned) == (
        "models/gemini-2.5-flash-native-audio-preview-12-2025"
    )
    assert not gemini_uses_thinking_level(pinned.gemini_live_model)
    assert gemini_generation_config(pinned)["thinkingConfig"] == {"thinkingBudget": 0}

    empty = make_settings(tmp_path, gemini_live_model="")
    assert gemini_model_name(empty) == f"models/{DEFAULT_GEMINI_LIVE_MODEL}"

    omit = make_settings(tmp_path, gemini_thinking_budget=-1)
    assert "thinkingConfig" not in gemini_generation_config(omit)

    settings = make_settings(tmp_path)
    order = gemini_live_try_order(settings)
    assert order[0] == "gemini-3.8-live"
    assert order[1:] == list(GEMINI_LIVE_SETUP_FALLBACKS)
    assert "preview-09-2025" not in order
    assert "extended-thinking" not in "".join(order)


@pytest.mark.asyncio
async def test_gemini_setup_falls_back_to_next_live_model(tmp_path):
    settings = make_settings(tmp_path, gemini_api_key="g-key")
    store = JobStore(settings.stella_db_path)
    job = store.create(kind="call", to_number="+14155552671", brief="Hi")
    models: list[str] = []

    async def gemini_connect(url, headers):
        n = len(models)
        if n == 0:
            ws = FakeProviderWS([json.dumps({"error": {"message": "model unavailable"}})])
        else:
            ws = FakeProviderWS([json.dumps({"setupComplete": {}})])

        orig_send = ws.send

        async def send(data):
            await orig_send(data)
            msg = ws.sent[-1]
            if isinstance(msg, dict) and "setup" in msg:
                models.append(msg["setup"]["model"])

        ws.send = send  # type: ignore[method-assign]
        return ws

    session = GeminiVoiceSession(
        settings=settings,
        job=job,
        gemini_connect=gemini_connect,
        hangup_cb=lambda: None,
        transcript_cb=lambda c: None,
        outcome_cb=lambda t: None,
    )
    await session.connect()
    assert models[0] == "models/gemini-3.8-live"
    assert models[1] == "models/gemini-3.1-flash-live-preview"
    assert session._active_model == "gemini-3.1-flash-live-preview"
    await session.close()


@pytest.mark.asyncio
async def test_hangup_guard_waits_for_outbound():
    hung: list[int] = []

    async def hangup():
        hung.append(1)

    guard = TelnyxMediaGuard(hangup_wait_audio_s=0.2, hangup_playout_pad_s=0.0)
    guard.mark_started()
    await guard.hangup_after_audio(hangup, lambda: False)
    assert hung == [1]


def test_mcp_token_protects_mcp_and_calls(tmp_path):
    settings = make_settings(tmp_path, stella_mcp_token="s3cret")
    from fastapi.testclient import TestClient

    client = TestClient(create_app(settings))
    assert client.get("/health").status_code == 200
    assert client.get("/calls/x").status_code == 401
    assert client.get("/calls/x?Token=wrong").status_code == 401
    assert client.get("/calls/x?Token=s3cret").status_code == 404
    assert client.get("/calls/x", headers={"Authorization": "Bearer s3cret"}).status_code == 404


# --- frag_ida (Claude Code as thinker) ---------------------------------------

from stella import ask_ida as ask_ida_mod  # noqa: E402
from stella.gemini_voice import gemini_tools  # noqa: E402

OWNER = "+4917612345678"


class FakeProc:
    def __init__(self, stdout=b"Morgen um acht Uhr Mathe.", stderr=b"", rc=0, hang=False):
        self._out, self._err, self.returncode, self._hang = stdout, stderr, rc, hang
        self.killed = False
        self.stdin_data: bytes | None = None
        self._done = asyncio.Event()

    async def communicate(self, data=None):
        self.stdin_data = data
        if self._hang:
            await self._done.wait()
        return self._out, self._err

    def kill(self):
        self.killed = True
        self._done.set()

    async def wait(self):
        return self.returncode


def ida_settings(tmp_path, **kw):
    opts = dict(
        ask_ida_enabled=True,
        stella_owner_number=OWNER,
        ask_ida_allowed_tools="mcp__Ida_Untis__stundenplan,mcp__Ida_SSH__ssh_befehl_ausfuehren",
        ask_ida_write_tools="mcp__Ida_Dashboard__create_event",
        ask_ida_mcp_config=str(tmp_path / "ida-mcp.json"),
    )
    opts.update(kw)
    return make_settings(tmp_path, **opts)


def make_spawner(proc):
    calls: list[tuple] = []

    async def spawn(*argv, **kwargs):
        calls.append((argv, kwargs))
        return proc

    return spawn, calls


@pytest.mark.asyncio
async def test_ask_ida_success_args_and_no_shell(tmp_path):
    settings = ida_settings(tmp_path)
    proc = FakeProc()
    spawn, calls = make_spawner(proc)
    q = 'Was habe ich morgen? "; rm -rf / #'
    out = await ask_ida_mod.ask_ida(settings, q, spawn=spawn)
    assert out == "Morgen um acht Uhr Mathe."
    argv, kwargs = calls[0]
    assert argv[0] == "claude" and "-p" in argv
    assert "--strict-mcp-config" in argv
    assert argv[argv.index("--mcp-config") + 1] == settings.ask_ida_mcp_config
    assert argv[argv.index("--tools") + 1] == ""
    allowed = argv[argv.index("--allowedTools") + 1]
    assert allowed == "mcp__Ida_Untis__stundenplan"  # SSH denied, write tool not confirmed
    assert "Bash" in argv[argv.index("--disallowedTools") + 1]
    assert "mcp__Ida_SSH" in argv[argv.index("--disallowedTools") + 1]
    assert q not in argv and not any(q in a for a in argv)  # stdin only
    assert proc.stdin_data == q.encode()
    assert "shell" not in kwargs
    assert "GEMINI_API_KEY" not in kwargs["env"]


def test_ask_ida_write_tools_only_when_confirmed(tmp_path):
    settings = ida_settings(tmp_path)
    assert "mcp__Ida_Dashboard__create_event" not in ask_ida_mod.resolve_tools(settings, False)[0]
    assert "mcp__Ida_Dashboard__create_event" in ask_ida_mod.resolve_tools(settings, True)[0]
    # deny always wins, even when confirmed
    settings = ida_settings(tmp_path, ask_ida_write_tools="mcp__X__termin_loeschen")
    assert "mcp__X__termin_loeschen" not in ask_ida_mod.resolve_tools(settings, True)[0]
    cmd = ask_ida_mod.build_command(settings, False)
    assert ask_ida_mod.NEEDS_CONFIRMATION in cmd[cmd.index("--append-system-prompt") + 1]


@pytest.mark.asyncio
async def test_ask_ida_timeout_kills_process(tmp_path):
    settings = ida_settings(tmp_path, ask_ida_timeout_s=0.05)
    proc = FakeProc(hang=True)
    spawn, _ = make_spawner(proc)
    out = await ask_ida_mod.ask_ida(settings, "Termine?", spawn=spawn)
    assert out == ask_ida_mod.MSG_TIMEOUT
    assert proc.killed


@pytest.mark.asyncio
async def test_ask_ida_errors_empty_and_clip(tmp_path):
    settings = ida_settings(tmp_path)
    spawn, _ = make_spawner(FakeProc(stderr=b"boom sk-secret", rc=1))
    assert await ask_ida_mod.ask_ida(settings, "x", spawn=spawn) == ask_ida_mod.MSG_FAILED
    spawn, _ = make_spawner(FakeProc(stdout=b"  \n"))
    assert await ask_ida_mod.ask_ida(settings, "x", spawn=spawn) == ask_ida_mod.MSG_EMPTY
    spawn, _ = make_spawner(FakeProc(stdout=b"wort " * 400))
    out = await ask_ida_mod.ask_ida(settings, "x", spawn=spawn)
    assert len(out) <= ask_ida_mod.MAX_ANSWER_CHARS


@pytest.mark.asyncio
async def test_ask_ida_cancel_event_kills_process(tmp_path):
    settings = ida_settings(tmp_path)
    proc = FakeProc(hang=True)
    spawn, _ = make_spawner(proc)
    cancel = asyncio.Event()
    task = asyncio.create_task(
        ask_ida_mod.ask_ida(settings, "x", spawn=spawn, cancel_event=cancel)
    )
    await asyncio.sleep(0.05)
    cancel.set()
    await task
    assert proc.killed


def test_ida_allowed_only_for_owner(tmp_path):
    s = ida_settings(tmp_path)
    assert ask_ida_mod.ida_allowed_for(s, OWNER)
    assert not ask_ida_mod.ida_allowed_for(s, "+4915112345678")
    assert not ask_ida_mod.ida_allowed_for(s, OWNER, False)
    assert not ask_ida_mod.ida_allowed_for(ida_settings(tmp_path, ask_ida_enabled=False), OWNER)
    assert not ask_ida_mod.ida_allowed_for(ida_settings(tmp_path, stella_owner_number=""), OWNER)


def test_place_call_sets_allow_ida_only_for_owner(tmp_path):
    settings = ida_settings(tmp_path)
    svc = JobService(settings, JobStore(settings.stella_db_path), TelnyxClient(settings, FakeTransport()))
    assert svc.place_call(to=OWNER, brief="hi").allow_ida is True
    assert svc.place_call(to="+14155552671", brief="hi", allow_ida=True).allow_ida is False


def test_gemini_setup_includes_frag_ida_only_when_allowed(tmp_path):
    names = lambda ask: [  # noqa: E731
        d["name"] for t in gemini_tools(ask_ida=ask) for d in t["functionDeclarations"]
    ]
    assert names(False) == ["hang_up"]
    assert names(True) == ["hang_up", "frag_ida"]

    settings = ida_settings(tmp_path)
    store = JobStore(settings.stella_db_path)

    def setup_tools(to, allow):
        job = store.create(kind="call", to_number=to, brief="b", allow_ida=allow)
        sess = GeminiVoiceSession(
            settings=settings, job=job, gemini_connect=None,
            hangup_cb=lambda: None, transcript_cb=lambda c: None, outcome_cb=lambda t: None,
        )
        payload = sess._setup_payload()
        return (
            [d["name"] for t in payload["tools"] for d in t["functionDeclarations"]],
            payload["systemInstruction"]["parts"][0]["text"],
        )

    tools, text = setup_tools(OWNER, True)
    assert "frag_ida" in tools and "frag_ida" in text
    tools, text = setup_tools("+14155552671", True)  # flag set but not the owner number
    assert tools == ["hang_up"] and "frag_ida" not in text
    tools, _ = setup_tools(OWNER, False)
    assert tools == ["hang_up"]


@pytest.mark.asyncio
async def test_frag_ida_does_not_block_and_hang_up_still_works(tmp_path, monkeypatch):
    settings = ida_settings(tmp_path)
    store = JobStore(settings.stella_db_path)
    job = store.create(kind="call", to_number=OWNER, brief="b", allow_ida=True)
    release = asyncio.Event()
    seen: dict[str, Any] = {}

    async def fake_ask(settings, question, **kw):
        seen["q"], seen["confirmed"] = question, kw.get("confirmed")
        await release.wait()
        return "Du hast morgen Mathe."

    monkeypatch.setattr("stella.gemini_voice.ask_ida", fake_ask)
    ws = FakeProviderWS()
    outcomes: list[str] = []
    sess = GeminiVoiceSession(
        settings=settings, job=job, gemini_connect=None,
        hangup_cb=lambda: None, transcript_cb=lambda c: None, outcome_cb=outcomes.append,
    )
    sess._gemini_ws = ws
    sess._telnyx_ws = FakeTelnyxWS()
    await asyncio.wait_for(
        sess._handle_gemini_event(
            {"toolCall": {"functionCalls": [
                {"id": "c1", "name": "frag_ida", "args": {"frage": "Stundenplan morgen?"}}
            ]}}
        ),
        timeout=1,
    )  # returns immediately although ask_ida is still pending
    assert not ws.sent
    # while frag_ida is pending, hang_up is handled as before
    await sess._handle_gemini_event(
        {"toolCall": {"functionCalls": [
            {"id": "h1", "name": "hang_up", "args": {"outcome": "done"}}
        ]}}
    )
    assert outcomes == ["done"]
    assert ws.sent[-1]["toolResponse"]["functionResponses"][0]["id"] == "h1"
    # a second frag_ida while one is running is rejected immediately
    await sess._handle_gemini_event(
        {"toolCall": {"functionCalls": [{"id": "c2", "name": "frag_ida", "args": {"frage": "x"}}]}}
    )
    await asyncio.sleep(0.05)
    busy = [m for m in ws.sent if m["toolResponse"]["functionResponses"][0]["id"] == "c2"]
    assert busy and "error" in busy[0]["toolResponse"]["functionResponses"][0]["response"]
    release.set()
    await sess._ida_task
    last = ws.sent[-1]["toolResponse"]["functionResponses"][0]
    assert last == {"id": "c1", "name": "frag_ida", "response": {"result": "Du hast morgen Mathe."}}
    assert seen == {"q": "Stundenplan morgen?", "confirmed": False}
    sess.guard.hangup_wait_audio_s = 0
    await sess.close()


@pytest.mark.asyncio
async def test_frag_ida_ignored_when_not_allowed(tmp_path):
    settings = make_settings(tmp_path)  # ask_ida disabled
    store = JobStore(settings.stella_db_path)
    job = store.create(kind="call", to_number="+14155552671", brief="b", allow_ida=True)
    ws = FakeProviderWS()
    sess = GeminiVoiceSession(
        settings=settings, job=job, gemini_connect=None,
        hangup_cb=lambda: None, transcript_cb=lambda c: None, outcome_cb=lambda t: None,
    )
    sess._gemini_ws = ws
    await sess._handle_gemini_event(
        {"toolCall": {"functionCalls": [{"id": "c1", "name": "frag_ida", "args": {"frage": "x"}}]}}
    )
    assert sess._ida_task is None and not ws.sent


def test_old_db_without_allow_ida_column_loads(tmp_path):
    import sqlite3

    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE jobs (id TEXT PRIMARY KEY, kind TEXT NOT NULL, to_number TEXT NOT NULL,
        brief TEXT NOT NULL, context TEXT NOT NULL DEFAULT '', speak_to TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL, telnyx_call_control_id TEXT NOT NULL DEFAULT '',
        telnyx_call_leg_id TEXT NOT NULL DEFAULT '', outcome TEXT NOT NULL DEFAULT '',
        transcript TEXT NOT NULL DEFAULT '', error TEXT NOT NULL DEFAULT '',
        voice_provider TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"""
    )
    conn.execute(
        "INSERT INTO jobs (id, kind, to_number, brief, status, voice_provider, created_at, updated_at)"
        " VALUES ('old1','call','+491','b','completed','grok','t','t')"
    )
    conn.commit()
    conn.close()
    store = JobStore(str(db))
    old = store.get("old1")
    assert old.voice_provider == "grok" and old.allow_ida is False
    assert store.create(kind="call", to_number="+492", brief="b", allow_ida=True).allow_ida is True


@pytest.mark.asyncio
async def test_hangup_waits_only_for_queued_audio_not_whole_call():
    """After a long call, hang_up must not wait for the cumulative audio length."""
    slept: list[float] = []

    async def fake_sleep(s):
        slept.append(s)

    guard = TelnyxMediaGuard(hangup_playout_pad_s=0.35, sleep=fake_sleep)
    guard.mark_started()
    frame = base64.b64encode(b"\xff" * 160).decode()  # 20 ms
    ws = FakeTelnyxWS()
    for _ in range(1500):  # 30 s of audio sent over the call
        await guard.send_outbound_pcmu(ws, frame)
    guard._play_end = 0.0  # ...which has long since finished playing
    guard._play_end = __import__("time").monotonic() + 0.2  # 200 ms still queued
    hung: list[int] = []
    await guard.hangup_after_audio(lambda: hung.append(1), lambda: False)
    assert hung == [1]
    assert slept and slept[-1] <= 0.2 + 0.35 + 0.01
    guard.clear_playout()
    assert guard.remaining_playout_s() == 0.0
