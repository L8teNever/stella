from __future__ import annotations

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
from stella.gemini_voice import GeminiVoiceSession, gemini_model_name, gemini_ws_url
from stella.voice import build_instructions, start_voice_bridge
from stella.voice_provider import GROK_DOWN_CACHE_TTL_SECONDS, VoiceProviderChooser
from stella.webhooks import verify_telnyx_signature
from stella.xai_auth import XAIAuth


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
        xai_oauth_token_path=str(tmp_path / "oauth.json"),
        xai_api_key="test-xai-key",
        telnyx_api_key="test-telnyx",
        telnyx_connection_id="conn-1",
        telnyx_from_number="+15551234567",
        stella_public_base_url="https://stella.example",
        stella_skip_webhook_verify=True,
    )
    defaults.update(kwargs)
    return Settings(**defaults)


def make_chooser(settings, xai, grok_probe=None) -> VoiceProviderChooser:
    return VoiceProviderChooser(
        settings,
        xai,
        grok_probe=grok_probe if grok_probe is not None else (lambda token: None),
    )


def test_normalize_e164():
    assert normalize_e164("+1 650 253 0000") == "+16502530000"
    with pytest.raises(StellaError):
        normalize_e164("not-a-number")


def test_place_call_dials_telnyx(tmp_path):
    settings = make_settings(tmp_path)
    store = JobStore(settings.stella_db_path)
    transport = FakeTransport()
    telnyx = TelnyxClient(settings, transport=transport)
    xai = XAIAuth(settings)
    svc = JobService(settings, store, telnyx, xai, voice_chooser=make_chooser(settings, xai))
    job = svc.place_call(
        to="+14155552671",
        brief="Reserve a table tomorrow 19:00 for 2, name Franz.",
        speak_to="the restaurant",
    )
    assert job.status == "dialing"
    assert job.voice_provider == "grok"
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
    xai = XAIAuth(settings)
    svc = JobService(
        settings,
        store,
        TelnyxClient(settings, FakeTransport()),
        xai,
        voice_chooser=make_chooser(settings, xai),
    )
    with pytest.raises(StellaError) as ei:
        svc.place_call(to="+14155552671", brief="hello")
    assert ei.value.code == "telnyx_config_missing"
    assert "TELNYX_API_KEY" in ei.value.message


def test_place_call_fails_without_xai(tmp_path):
    settings = make_settings(tmp_path, xai_api_key="")
    store = JobStore(settings.stella_db_path)
    xai = XAIAuth(settings)
    svc = JobService(
        settings,
        store,
        TelnyxClient(settings, FakeTransport()),
        xai,
        voice_chooser=make_chooser(settings, xai),
    )
    with pytest.raises(StellaError) as ei:
        svc.place_call(to="+14155552671", brief="hello")
    assert ei.value.code == "voice_auth_missing"
    assert "GEMINI_API_KEY" in ei.value.message


def test_place_call_allows_gemini_only(tmp_path):
    settings = make_settings(tmp_path, xai_api_key="", gemini_api_key="test-gemini")
    store = JobStore(settings.stella_db_path)
    xai = XAIAuth(settings)
    svc = JobService(
        settings,
        store,
        TelnyxClient(settings, FakeTransport()),
        xai,
        voice_chooser=make_chooser(settings, xai),
    )
    job = svc.place_call(to="+14155552671", brief="hello")
    assert job.status == "dialing"
    assert job.voice_provider == "gemini"


def test_webhook_updates_status(tmp_path):
    settings = make_settings(tmp_path)
    store = JobStore(settings.stella_db_path)
    xai = XAIAuth(settings)
    svc = JobService(
        settings,
        store,
        TelnyxClient(settings, FakeTransport()),
        xai,
        voice_chooser=make_chooser(settings, xai),
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
    assert "do NOT share" in text or "do not share" in text.lower() or "NOT Ida" in text


def test_mcp_tools_registered(tmp_path):
    settings = make_settings(tmp_path, xai_api_key="")
    store = JobStore(settings.stella_db_path)
    svc = JobService(settings, store, TelnyxClient(settings, FakeTransport()), XAIAuth(settings))
    mcp = build_mcp(svc)
    names = {t.name for t in mcp._tool_manager.list_tools()}
    assert names == {"stella_call", "stella_call_status", "stella_briefing_call"}


def test_health_and_call_http(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings)
    from fastapi.testclient import TestClient

    client = TestClient(app)
    h = client.get("/health")
    assert h.status_code == 200
    body = h.json()
    assert body["xai_auth"] == "api_key"
    assert body["voice_provider"]["primary"] == "grok"
    assert body["voice_provider"]["fallback"] == "gemini"
    assert body["voice_provider"]["sticky"] == "per_call_job"
    assert body["voice_provider"]["grok_down_cache_ttl_seconds"] == GROK_DOWN_CACHE_TTL_SECONDS
    assert body["voice_provider"]["grok_down_cached"] is False
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
    def __init__(self, messages: list[str] | None = None) -> None:
        self.sent: list[Any] = []
        self._messages = list(messages or [])
        self.closed = False

    async def send(self, data: str) -> None:
        self.sent.append(json.loads(data) if isinstance(data, str) else data)

    async def close(self) -> None:
        self.closed = True
        self._messages = []

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)


class FakeTelnyxWS:
    def __init__(self, texts: list[str] | None = None) -> None:
        self.sent: list[str] = []
        self._texts = list(texts or [])

    async def iter_text(self):
        for t in self._texts:
            yield t

    async def send_text(self, data: str) -> None:
        self.sent.append(data)


@pytest.mark.asyncio
async def test_start_voice_bridge_uses_grok_when_locked(tmp_path):
    settings = make_settings(tmp_path)
    store = JobStore(settings.stella_db_path)
    job = store.create(kind="call", to_number="+14155552671", brief="Book a table.")
    store.update(job.id, voice_provider="grok")
    job = store.get(job.id)
    grok_ws = FakeProviderWS()
    seen: list[str] = []

    async def grok_connect(url, headers):
        assert "api.x.ai" in url
        assert headers["Authorization"].startswith("Bearer ")
        return grok_ws

    async def gemini_connect(url, headers):
        raise AssertionError("Gemini should not be used")

    provider = await start_voice_bridge(
        settings=settings,
        job=job,
        telnyx_ws=FakeTelnyxWS(),
        grok_token_fn=lambda: "tok",
        grok_connect=grok_connect,
        hangup_cb=lambda: None,
        transcript_cb=lambda c: None,
        outcome_cb=lambda t: None,
        provider_cb=seen.append,
        gemini_connect=gemini_connect,
        locked_provider="grok",
    )
    assert provider == "grok"
    assert seen == ["grok"]
    assert grok_ws.sent[0]["type"] == "session.update"


@pytest.mark.asyncio
async def test_start_voice_bridge_locked_gemini_skips_grok(tmp_path):
    settings = make_settings(tmp_path, gemini_api_key="g-key")
    store = JobStore(settings.stella_db_path)
    job = store.create(kind="call", to_number="+14155552671", brief="Book a table.")
    gemini_ws = FakeProviderWS([json.dumps({"setupComplete": {}})])
    grok_hits: list[str] = []
    seen: list[str] = []

    async def grok_connect(url, headers):
        grok_hits.append(url)
        raise AssertionError("Grok must not be probed mid-call")

    async def gemini_connect(url, headers):
        assert "generativelanguage.googleapis.com" in url
        assert "g-key" in url
        return gemini_ws

    provider = await start_voice_bridge(
        settings=settings,
        job=job,
        telnyx_ws=FakeTelnyxWS(),
        grok_token_fn=lambda: "tok",
        grok_connect=grok_connect,
        hangup_cb=lambda: None,
        transcript_cb=lambda c: None,
        outcome_cb=lambda t: None,
        provider_cb=seen.append,
        gemini_connect=gemini_connect,
        locked_provider="gemini",
    )
    assert provider == "gemini"
    assert seen == ["gemini"]
    assert grok_hits == []
    voice = gemini_ws.sent[0]["setup"]["generationConfig"]["speechConfig"]["voiceConfig"][
        "prebuiltVoiceConfig"
    ]["voiceName"]
    assert voice == "Aoede"


@pytest.mark.asyncio
async def test_start_voice_bridge_locked_grok_does_not_fallback_on_403(tmp_path):
    settings = make_settings(tmp_path, gemini_api_key="g-key")
    store = JobStore(settings.stella_db_path)
    job = store.create(kind="call", to_number="+14155552671", brief="Book a table.")
    gemini_hits = []

    async def grok_connect(url, headers):
        raise ConnectionError("HTTP 403 quota exceeded")

    async def gemini_connect(url, headers):
        gemini_hits.append(url)
        raise AssertionError("must not fall back")

    with pytest.raises(RuntimeError, match="Grok Voice"):
        await start_voice_bridge(
            settings=settings,
            job=job,
            telnyx_ws=FakeTelnyxWS(),
            grok_token_fn=lambda: "tok",
            grok_connect=grok_connect,
            hangup_cb=lambda: None,
            transcript_cb=lambda c: None,
            outcome_cb=lambda t: None,
            gemini_connect=gemini_connect,
            locked_provider="grok",
        )
    assert gemini_hits == []


@pytest.mark.asyncio
async def test_start_voice_bridge_no_mid_call_gemini_after_grok_connect(tmp_path):
    settings = make_settings(tmp_path, gemini_api_key="g-key")
    store = JobStore(settings.stella_db_path)
    job = store.create(kind="call", to_number="+14155552671", brief="Book a table.")
    gemini_hits: list[str] = []

    class DroppingGrok(FakeProviderWS):
        async def __anext__(self) -> str:
            raise ConnectionError("Grok dropped mid-call")

    grok_ws = DroppingGrok()

    async def grok_connect(url, headers):
        return grok_ws

    async def gemini_connect(url, headers):
        gemini_hits.append(url)
        raise AssertionError("no mid-call Gemini")

    with pytest.raises(ConnectionError, match="mid-call"):
        await start_voice_bridge(
            settings=settings,
            job=job,
            telnyx_ws=FakeTelnyxWS(),
            grok_token_fn=lambda: "tok",
            grok_connect=grok_connect,
            hangup_cb=lambda: None,
            transcript_cb=lambda c: None,
            outcome_cb=lambda t: None,
            gemini_connect=gemini_connect,
            locked_provider="grok",
        )
    assert gemini_hits == []


@pytest.mark.asyncio
async def test_start_voice_bridge_second_media_socket_reuses_lock(tmp_path):
    settings = make_settings(tmp_path, gemini_api_key="g-key")
    store = JobStore(settings.stella_db_path)
    job = store.create(kind="call", to_number="+14155552671", brief="Book a table.")
    grok_hits: list[int] = []

    async def grok_connect(url, headers):
        grok_hits.append(1)
        raise AssertionError("locked gemini")

    async def gemini_connect(url, headers):
        return FakeProviderWS([json.dumps({"setupComplete": {}})])

    for _ in range(2):
        provider = await start_voice_bridge(
            settings=settings,
            job=job,
            telnyx_ws=FakeTelnyxWS(),
            grok_token_fn=lambda: "tok",
            grok_connect=grok_connect,
            hangup_cb=lambda: None,
            transcript_cb=lambda c: None,
            outcome_cb=lambda t: None,
            gemini_connect=gemini_connect,
            locked_provider="gemini",
        )
        assert provider == "gemini"
    assert grok_hits == []


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
            grok_token_fn=lambda: "tok",
            grok_connect=lambda u, h: None,
            hangup_cb=lambda: None,
            transcript_cb=lambda c: None,
            outcome_cb=lambda t: None,
            locked_provider="gemini",
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
    assert settings.xai_voice == "eve"


def test_dial_keeps_inbound_track_for_bidirectional(tmp_path):
    settings = make_settings(tmp_path)
    store = JobStore(settings.stella_db_path)
    transport = FakeTransport()
    telnyx = TelnyxClient(settings, transport=transport)
    xai = XAIAuth(settings)
    svc = JobService(settings, store, telnyx, xai, voice_chooser=make_chooser(settings, xai))
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
    assert gemini_model_name(settings).startswith("models/")


def test_chooser_probes_grok_once_then_caches_403(tmp_path):
    settings = make_settings(tmp_path, gemini_api_key="g-key")
    xai = XAIAuth(settings)
    probes: list[str] = []
    clock = {"t": 0.0}

    def probe(token: str) -> None:
        probes.append(token)
        raise ConnectionError("HTTP 403 quota exceeded")

    chooser = VoiceProviderChooser(
        settings,
        xai,
        grok_probe=probe,
        clock=lambda: clock["t"],
        down_ttl_seconds=300,
    )
    assert chooser.choose() == "gemini"
    assert chooser.choose() == "gemini"
    assert probes == ["test-xai-key"]
    assert chooser.grok_down_cached()
    clock["t"] = 299
    assert chooser.choose() == "gemini"
    assert probes == ["test-xai-key"]
    clock["t"] = 300
    assert chooser.choose() == "gemini"
    assert probes == ["test-xai-key", "test-xai-key"]


def test_place_call_uses_cached_gemini_without_second_probe(tmp_path):
    settings = make_settings(tmp_path, gemini_api_key="g-key")
    store = JobStore(settings.stella_db_path)
    transport = FakeTransport()
    telnyx = TelnyxClient(settings, transport=transport)
    xai = XAIAuth(settings)
    probes: list[int] = []

    def probe(token: str) -> None:
        probes.append(1)
        raise ConnectionError("HTTP 403")

    chooser = VoiceProviderChooser(settings, xai, grok_probe=probe)
    svc = JobService(settings, store, telnyx, xai, voice_chooser=chooser)
    a = svc.place_call(to="+14155552671", brief="first")
    b = svc.place_call(to="+14155552671", brief="second")
    assert a.voice_provider == "gemini"
    assert b.voice_provider == "gemini"
    assert probes == [1]


def test_chooser_uses_grok_when_probe_ok(tmp_path):
    settings = make_settings(tmp_path, gemini_api_key="g-key")
    xai = XAIAuth(settings)
    chooser = VoiceProviderChooser(settings, xai, grok_probe=lambda t: None)
    assert chooser.choose() == "grok"
