from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from stella.config import Settings
from stella.ask_ida import ida_allowed_for
from stella.errors import StellaError, normalize_e164
from stella.store import CallJob, JobStore
from stella.telnyx_client import TelnyxClient

logger = logging.getLogger(__name__)

BERLIN = ZoneInfo("Europe/Berlin")
MAX_SCHEDULE_AHEAD = timedelta(days=14)
_HHMM = re.compile(r"^\s*(\d{1,2})[:.](\d{2})\s*(?:uhr)?\s*$", re.IGNORECASE)


def parse_run_at(
    *, uhrzeit: str = "", in_minuten: Any = None, datum: str = "", now: datetime | None = None
) -> datetime:
    """Resolve a spoken callback time (Europe/Berlin) to an aware datetime. Raises StellaError."""
    now = (now or datetime.now(timezone.utc)).astimezone(BERLIN)
    uhrzeit = (uhrzeit or "").strip()
    if in_minuten not in (None, ""):
        try:
            minutes = int(in_minuten)
        except (TypeError, ValueError):
            raise StellaError("`in_minuten` muss eine ganze Zahl sein.", "invalid_time") from None
        if minutes < 1:
            raise StellaError("`in_minuten` muss mindestens 1 sein.", "invalid_time")
        run_at = now + timedelta(minutes=minutes)
    elif uhrzeit:
        try:
            run_at = datetime.fromisoformat(uhrzeit)
            if run_at.tzinfo is None:
                run_at = run_at.replace(tzinfo=BERLIN)
        except ValueError:
            m = _HHMM.match(uhrzeit)
            if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
                raise StellaError(
                    f"Uhrzeit '{uhrzeit}' nicht verständlich, erwarte HH:MM (24h).", "invalid_time"
                ) from None
            day = now.date()
            if datum:
                try:
                    day = datetime.fromisoformat(datum.strip()).date()
                except ValueError:
                    raise StellaError(f"Datum '{datum}' ungültig (JJJJ-MM-TT).", "invalid_time") from None
            run_at = datetime(day.year, day.month, day.day, int(m.group(1)), int(m.group(2)), tzinfo=BERLIN)
    else:
        raise StellaError("Gib `uhrzeit` (HH:MM) oder `in_minuten` an.", "invalid_time")
    if run_at <= now:
        raise StellaError(
            f"Die Zeit {run_at:%H:%M} liegt in der Vergangenheit (jetzt {now:%H:%M} Uhr).",
            "time_in_past",
        )
    if run_at - now > MAX_SCHEDULE_AHEAD:
        raise StellaError("Rückrufe sind nur bis 14 Tage im Voraus möglich.", "time_too_far")
    return run_at


class JobService:
    def __init__(
        self,
        settings: Settings,
        store: JobStore,
        telnyx: TelnyxClient,
    ) -> None:
        self.settings = settings
        self.store = store
        self.telnyx = telnyx

    def place_call(
        self,
        *,
        to: str,
        brief: str,
        context: str = "",
        speak_to: str = "",
        kind: str = "call",
        allow_ida: bool | None = None,
    ) -> CallJob:
        to_e164 = normalize_e164(to)
        if not (brief or "").strip():
            raise StellaError("Missing `brief` (task / script context).", "invalid_brief")
        if not (self.settings.gemini_api_key or "").strip():
            raise StellaError(
                "GEMINI_API_KEY is not set; Stella needs Gemini Live to place calls.",
                "gemini_key_missing",
            )
        provider = "gemini"
        job = self.store.create(
            kind=kind,
            to_number=to_e164,
            brief=brief.strip(),
            context=(context or "").strip(),
            speak_to=(speak_to or "").strip(),
            allow_ida=ida_allowed_for(self.settings, to_e164, allow_ida),
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

    def plan_task(self, job: CallJob, args: dict[str, Any]) -> dict[str, Any]:
        """Validate a spoken 'do X later' request (raises StellaError); stores nothing."""
        run_at = parse_run_at(
            uhrzeit=str(args.get("uhrzeit") or ""),
            in_minuten=args.get("in_minuten"),
            datum=str(args.get("datum") or ""),
        )
        aufgabe = str(args.get("aufgabe") or args.get("grund") or "").strip()[:600]
        if not aufgabe:
            aufgabe = "Ruf mich an."
        return {
            "run_at": run_at, "aufgabe": aufgabe, "to": job.to_number,
            "speak_to": job.speak_to or "Simon",
        }

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
