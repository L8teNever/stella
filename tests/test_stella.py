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
from stella.voice import build_instructions
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
    svc = JobService(settings, store, telnyx, xai)
    job = svc.place_call(
        to="+14155552671",
        brief="Reserve a table tomorrow 19:00 for 2, name Franz.",
        speak_to="the restaurant",
    )
    assert job.status == "dialing"
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
    svc = JobService(settings, store, TelnyxClient(settings, FakeTransport()), XAIAuth(settings))
    with pytest.raises(StellaError) as ei:
        svc.place_call(to="+14155552671", brief="hello")
    assert ei.value.code == "telnyx_config_missing"
    assert "TELNYX_API_KEY" in ei.value.message


def test_place_call_fails_without_xai(tmp_path):
    settings = make_settings(tmp_path, xai_api_key="")
    store = JobStore(settings.stella_db_path)
    svc = JobService(settings, store, TelnyxClient(settings, FakeTransport()), XAIAuth(settings))
    with pytest.raises(StellaError) as ei:
        svc.place_call(to="+14155552671", brief="hello")
    assert ei.value.code == "xai_auth_missing"


def test_webhook_updates_status(tmp_path):
    settings = make_settings(tmp_path)
    store = JobStore(settings.stella_db_path)
    svc = JobService(settings, store, TelnyxClient(settings, FakeTransport()), XAIAuth(settings))
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
    assert h.json()["xai_auth"] == "api_key"
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
