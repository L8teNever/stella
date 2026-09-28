from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from stella.config import Settings
from stella.errors import StellaError, normalize_e164
from stella.store import CallJob, JobStore
from stella.telnyx_client import TelnyxClient
from stella.voice_provider import VoiceProviderChooser
from stella.xai_auth import XAIAuth

logger = logging.getLogger(__name__)


class JobService:
    def __init__(
        self,
        settings: Settings,
        store: JobStore,
        telnyx: TelnyxClient,
        xai: XAIAuth,
        voice_chooser: VoiceProviderChooser | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.telnyx = telnyx
        self.xai = xai
        self.voice_chooser = voice_chooser or VoiceProviderChooser(settings, xai)

    def place_call(
        self,
        *,
        to: str,
        brief: str,
        context: str = "",
        speak_to: str = "",
        kind: str = "call",
    ) -> CallJob:
        to_e164 = normalize_e164(to)
        if not (brief or "").strip():
            raise StellaError("Missing `brief` (task / script context).", "invalid_brief")
        # One Grok realtime probe (or cache hit) for this job — sticky for the call.
        provider = self.voice_chooser.choose()
        job = self.store.create(
            kind=kind,
            to_number=to_e164,
            brief=brief.strip(),
            context=(context or "").strip(),
            speak_to=(speak_to or "").strip(),
        )
        self.store.update(job.id, voice_provider=provider)
        webhook = self.settings.public_http_url("/webhooks/telnyx")
        stream = self.settings.public_ws_url(f"/media/{job.id}")
        client_state = base64_json({"job_id": job.id})
        try:
            dialed = self.telnyx.dial(
                to=to_e164,
                webhook_url=webhook,
                stream_url=stream,
                client_state=client_state,
            )
        except StellaError as exc:
            self.store.update(job.id, status="failed", error=exc.message)
            raise
        self.store.update(
            job.id,
            status="dialing",
            voice_provider=provider,
            telnyx_call_control_id=dialed.get("call_control_id") or "",
            telnyx_call_leg_id=dialed.get("call_leg_id") or "",
        )
        return self.store.get(job.id)  # type: ignore[return-value]

    def status(self, call_id: str) -> CallJob:
        job = self.store.get(call_id)
        if not job:
            raise StellaError(f"Unknown call id: {call_id}", "not_found")
        return job

    def handle_telnyx_event(self, body: dict[str, Any]) -> None:
        data = body.get("data") or body
        event_type = data.get("event_type") or body.get("event_type") or ""
        payload = data.get("payload") or {}
        ccid = payload.get("call_control_id") or ""
        client_state = payload.get("client_state")
        job = None
        if ccid:
            job = self.store.get_by_call_control_id(ccid)
        if job is None and client_state:
            job_id = decode_client_state(client_state).get("job_id")
            if job_id:
                job = self.store.get(job_id)
        if job is None:
            logger.warning("Telnyx event %s for unknown call %s", event_type, ccid)
            return
        self.store.add_event(job.id, event_type, payload)
        status_map = {
            "call.initiated": "initiated",
            "call.answered": "in_progress",
            "call.hangup": "completed",
            "call.machine.detection.ended": job.status,
            "streaming.started": "in_progress",
            "streaming.stopped": job.status,
            "streaming.failed": "failed",
        }
        new_status = status_map.get(event_type)
        fields: dict[str, Any] = {}
        if new_status:
            fields["status"] = new_status
        if ccid and not job.telnyx_call_control_id:
            fields["telnyx_call_control_id"] = ccid
        if event_type == "call.hangup":
            hangup_cause = payload.get("hangup_cause") or payload.get("hangup_source") or ""
            if hangup_cause and not job.outcome:
                fields["outcome"] = fields.get("outcome") or f"Call ended ({hangup_cause})."
            fields["status"] = "completed"
        if event_type == "streaming.failed":
            fields["error"] = str(payload.get("reason") or "media streaming failed")
        if fields:
            updated = self.store.update(job.id, **fields)
            if updated and event_type == "call.hangup":
                self._maybe_callback(updated)

    def hangup(self, job: CallJob) -> None:
        if job.telnyx_call_control_id:
            try:
                self.telnyx.hangup(job.telnyx_call_control_id)
            except StellaError as exc:
                logger.warning("hangup failed for %s: %s", job.id, exc.message)

    def _maybe_callback(self, job: CallJob) -> None:
        url = (self.settings.stella_callback_url or "").strip()
        if not url:
            return
        try:
            httpx.post(url, json=job.to_public_dict(), timeout=10.0)
        except Exception:
            logger.exception("STELLA_CALLBACK_URL post failed")


def base64_json(obj: dict[str, Any]) -> str:
    import base64

    return base64.b64encode(json.dumps(obj).encode("utf-8")).decode("ascii")


def decode_client_state(raw: str) -> dict[str, Any]:
    import base64

    if not raw:
        return {}
    try:
        pad = "=" * (-len(raw) % 4)
        return json.loads(base64.b64decode(raw + pad).decode("utf-8"))
    except Exception:
        return {}
