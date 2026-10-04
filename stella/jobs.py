from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from stella.audio_pcmu import codec_from_telnyx_payload, preferred_telnyx_codec
from stella.config import Settings
from stella.ask_ida import ida_allowed_for
from stella.errors import StellaError, normalize_e164
from stella.store import CallJob, JobStore
from stella.telnyx_client import TelnyxClient

logger = logging.getLogger(__name__)

BERLIN = ZoneInfo("Europe/Berlin")
MAX_SCHEDULE_AHEAD = timedelta(days=14)
_HHMM = re.compile(r"^\s*(\d{1,2})[:.](\d{2})\s*(?:uhr)?\s*$", re.IGNORECASE)
INBOUND_UNKNOWN_MODES = frozenset({"hangup", "speak", "answer"})
DEFAULT_INBOUND_BRIEF = (
    "Du bist Stella. Simon ruft an — führe ein normales kurzes Gespräch auf Deutsch."
)


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
        media_codec = preferred_telnyx_codec(self.settings.telnyx_from_number, to_e164)
        job = self.store.create(
            kind=kind,
            to_number=to_e164,
            brief=brief.strip(),
            context=(context or "").strip(),
            speak_to=(speak_to or "").strip(),
            allow_ida=ida_allowed_for(self.settings, to_e164, allow_ida),
        )
        self.store.update(job.id, voice_provider=provider, media_codec=media_codec)
        webhook = self.settings.public_http_url("/webhooks/telnyx")
        stream = self.settings.public_ws_url(f"/media/{job.id}")
        client_state = base64_json({"job_id": job.id})
        try:
            dialed = self.telnyx.dial(
                to=to_e164,
                webhook_url=webhook,
                stream_url=stream,
                client_state=client_state,
                stream_bidirectional_codec=media_codec,
            )
        except StellaError as exc:
            self.store.update(job.id, status="failed", error=exc.message)
            raise
        self.store.update(
            job.id,
            status="dialing",
            voice_provider=provider,
            media_codec=media_codec,
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
            "kontext": str(args.get("_kontext") or "").strip()[-1500:],
        }

    def status(self, call_id: str) -> CallJob:
        job = self.store.get(call_id)
        if not job:
            raise StellaError(f"Unknown call id: {call_id}", "not_found")
        return job

    def handle_telnyx_event(self, body: dict[str, Any]) -> None:
        data = body.get("data") or body
        payload = data.get("payload") if isinstance(data.get("payload"), dict) else {}
        if not payload.get("call_control_id") and (
            data.get("call_control_id") or body.get("call_control_id")
        ):
            payload = {**{k: v for k, v in data.items() if k != "payload"}, **payload}
        event_type = str(
            data.get("event_type") or body.get("event_type") or payload.get("event_type") or ""
        )
        ccid = str(
            payload.get("call_control_id")
            or data.get("call_control_id")
            or body.get("call_control_id")
            or ""
        )
        client_state = payload.get("client_state")
        job = None
        if ccid:
            job = self.store.get_by_call_control_id(ccid)
        if job is None and client_state:
            job_id = decode_client_state(client_state).get("job_id")
            if job_id:
                job = self.store.get(job_id)
        # Any non-outbound initiated without a job is inbound (direction optional).
        if job is None and event_type == "call.initiated" and ccid and not self._looks_outbound(payload):
            job = self._accept_or_reject_inbound(payload)
            if job is None:
                return
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
        # Do not demote inbound after answer/reject (initiated would overwrite in_progress).
        inbound_locked = job.kind == "inbound" and job.status in {
            "in_progress",
            "rejected",
            "rejecting",
            "failed",
        }
        if new_status and not (event_type == "call.initiated" and inbound_locked):
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
        if event_type in {"call.answered", "streaming.started"}:
            detected = codec_from_telnyx_payload(payload)
            if detected:
                fields["media_codec"] = detected
        if event_type == "call.speak.ended" and job.kind == "inbound" and job.status == "rejecting":
            try:
                self.telnyx.hangup(job.telnyx_call_control_id or ccid)
            except StellaError as exc:
                logger.warning("inbound reject hangup failed: %s", exc.message)
            fields["status"] = "rejected"
        if fields:
            updated = self.store.update(job.id, **fields)
            if updated and event_type == "call.hangup":
                self._maybe_callback(updated)

    def _looks_outbound(self, payload: dict[str, Any]) -> bool:
        direction = str(
            payload.get("direction") or payload.get("call_direction") or ""
        ).strip().lower()
        return direction in {"outgoing", "outbound"}

    def _same_number(self, a: str, b: str) -> bool:
        left, right = party_e164(a), party_e164(b)
        return bool(left and right and left == right)

    def _inbound_unknown_mode(self) -> str:
        mode = (self.settings.stella_inbound_unknown or "hangup").strip().lower()
        return mode if mode in INBOUND_UNKNOWN_MODES else "hangup"

    def _accept_or_reject_inbound(self, payload: dict[str, Any]) -> CallJob | None:
        ccid = str(payload.get("call_control_id") or "")
        if not ccid:
            logger.warning("Inbound call.initiated without call_control_id")
            return None
        existing = self.store.get_by_call_control_id(ccid)
        if existing:
            return existing
        caller = (
            party_e164(payload.get("from"))
            or party_e164(payload.get("caller_id_number"))
            or party_e164(payload.get("from_number"))
            or "unknown"
        )
        dest = (
            party_e164(payload.get("to"))
            or party_e164(payload.get("to_number"))
            or party_e164(self.settings.telnyx_from_number)
        )
        owner_cfg = (self.settings.stella_owner_number or "").strip()
        owner = self._same_number(caller, owner_cfg)
        enabled = bool(self.settings.stella_inbound_enabled)
        mode = self._inbound_unknown_mode()
        dest_ours = self._same_number(dest, self.settings.telnyx_from_number) or not dest
        # Owner always; no owner configured → answer inbound to our DID; else unknown policy.
        accept = enabled and (owner or mode == "answer" or (not owner_cfg and dest_ours))
        logger.info(
            "inbound call %s from %s to %s owner=%s accept=%s",
            ccid,
            caller,
            dest,
            owner,
            accept,
        )
        media_codec = preferred_telnyx_codec(
            self.settings.telnyx_from_number, dest, caller if caller != "unknown" else None
        )
        brief = (self.settings.stella_inbound_brief or "").strip() or DEFAULT_INBOUND_BRIEF
        if accept and not owner:
            brief = (self.settings.stella_inbound_unknown_brief or "").strip() or brief
        speak_to = "Simon" if owner else "Anrufer"
        job = self.store.create(
            kind="inbound",
            to_number=caller,
            brief=brief,
            context=f"inbound to {dest or self.settings.telnyx_from_number}",
            speak_to=speak_to,
            allow_ida=ida_allowed_for(self.settings, caller),
        )
        self.store.update(
            job.id,
            voice_provider="gemini",
            media_codec=media_codec,
            telnyx_call_control_id=ccid,
            telnyx_call_leg_id=str(payload.get("call_leg_id") or ""),
            status="initiated",
        )
        job = self.store.get(job.id)
        assert job is not None
        if accept:
            self._answer_inbound(job, media_codec)
        elif enabled and mode == "speak":
            self._reject_inbound_speak(job)
        else:
            self._reject_inbound_hangup(job, "inbound disabled" if not enabled else "unknown caller")
        return self.store.get(job.id)

    def _answer_inbound(self, job: CallJob, media_codec: str) -> None:
        if not (self.settings.gemini_api_key or "").strip():
            self.store.update(job.id, status="failed", error="GEMINI_API_KEY is not set.")
            try:
                self.telnyx.hangup(job.telnyx_call_control_id)
            except StellaError as exc:
                logger.warning("inbound hangup after missing Gemini key: %s", exc.message)
            return
        webhook = self.settings.public_http_url("/webhooks/telnyx")
        stream = self.settings.public_ws_url(f"/media/{job.id}")
        client_state = base64_json({"job_id": job.id})
        try:
            # Embed the media stream on answer only — same as outbound dial.
            # A second streaming_start races the first WS and Telnyx returns
            # 422 90046 "Failed to connect to destination".
            self.telnyx.answer(
                job.telnyx_call_control_id,
                client_state=client_state,
                webhook_url=webhook,
                stream_url=stream,
                stream_bidirectional_codec=media_codec,
                party_number=job.to_number,
            )
        except StellaError as exc:
            self.store.update(job.id, status="failed", error=exc.message)
            logger.warning("inbound answer failed: %s", exc.message)
            return
        self.store.update(job.id, status="in_progress", voice_provider="gemini", media_codec=media_codec)

    def _reject_inbound_hangup(self, job: CallJob, reason: str) -> None:
        try:
            self.telnyx.hangup(job.telnyx_call_control_id)
        except StellaError as exc:
            logger.warning("inbound reject hangup failed: %s", exc.message)
        self.store.update(job.id, status="rejected", outcome=reason)

    def _reject_inbound_speak(self, job: CallJob) -> None:
        text = (self.settings.stella_inbound_reject_text or "").strip() or (
            "Diese Nummer ist nicht erreichbar."
        )
        try:
            self.telnyx.answer(job.telnyx_call_control_id)
            self.telnyx.speak(job.telnyx_call_control_id, text, language="de-DE")
        except StellaError as exc:
            logger.warning("inbound reject speak failed: %s", exc.message)
            self._reject_inbound_hangup(job, "unknown caller")
            return
        self.store.update(
            job.id,
            status="rejecting",
            outcome="unknown caller",
            allow_ida=False,
        )

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


def party_e164(raw: Any) -> str:
    if isinstance(raw, dict):
        raw = (
            raw.get("phone_number")
            or raw.get("number")
            or raw.get("from")
            or raw.get("to")
            or ""
        )
    text = str(raw or "").strip()
    if not text:
        return ""
    lower = text.lower()
    for prefix in ("sip:", "sips:", "tel:"):
        if lower.startswith(prefix):
            text = text[len(prefix) :]
            lower = text.lower()
            break
    if "@" in text:
        text = text.split("@", 1)[0]
    text = text.strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    if text.startswith("+"):
        candidate = "+" + digits
    elif digits:
        candidate = "+" + digits
    else:
        candidate = text
    try:
        return normalize_e164(candidate)
    except StellaError:
        return ""


def decode_client_state(raw: str) -> dict[str, Any]:
    import base64

    if not raw:
        return {}
    try:
        pad = "=" * (-len(raw) % 4)
        return json.loads(base64.b64decode(raw + pad).decode("utf-8"))
    except Exception:
        return {}
