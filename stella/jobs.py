from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from stella import call_sessions
from stella.audio_pcmu import codec_from_telnyx_payload, preferred_telnyx_codec
from stella.config import Settings
from stella.ask_ida import ida_allowed_for
from stella.errors import StellaError, normalize_e164
from stella.store import CallJob, ChannelBusy, JobStore
from stella.telnyx_client import TelnyxClient

logger = logging.getLogger(__name__)

BERLIN = ZoneInfo("Europe/Berlin")
MAX_SCHEDULE_AHEAD = timedelta(days=14)
_HHMM = re.compile(r"^\s*(\d{1,2})[:.](\d{2})\s*(?:uhr)?\s*$", re.IGNORECASE)
INBOUND_UNKNOWN_MODES = frozenset({"hangup", "speak", "answer"})
# The PSTN leg is up (or the dial HTTP is in flight). `completing` is only
# post-hangup bookkeeping and must not block the next call.
BUSY_STATUSES = frozenset({
    "queued",
    "dialing",
    "initiated",
    "in_progress",
    "rejecting",
})
TERMINAL_STATUSES = frozenset({"completed", "failed", "rejected"})
# Watched by the reaper. Includes post-hangup `completing`, which is not busy.
NON_TERMINAL_STATUSES = BUSY_STATUSES | frozenset({"completing"})
_ENDED_LIVENESS = frozenset({"ended", "ended_90015"})
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
        try:
            job = self.store.create(
                kind=kind,
                to_number=to_e164,
                brief=brief.strip(),
                context=(context or "").strip(),
                speak_to=(speak_to or "").strip(),
                allow_ida=ida_allowed_for(self.settings, to_e164, allow_ida),
                reject_if_status=BUSY_STATUSES,
            )
        except ChannelBusy as exc:
            raise StellaError(
                f"Stella is already on a call ({exc.call_id}).",
                "busy",
                call_id=exc.call_id,
            ) from exc
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

    def busy(self) -> dict[str, Any]:
        """Whether any call is on the line. One indexed read; no Telnyx, no waiting."""
        our = (self.settings.telnyx_from_number or "").strip()
        calls = []
        for row in self.store.list_by_status(BUSY_STATUSES):
            inbound = row["kind"] == "inbound"
            # Inbound stores the caller in to_number; outbound stores the destination.
            calls.append(
                {
                    "id": row["id"],
                    "direction": "inbound" if inbound else "outbound",
                    "status": row["status"],
                    "started_at": row["created_at"],
                    "to": our if inbound else row["to_number"],
                    "from": row["to_number"] if inbound else our,
                }
            )
        return {"busy": bool(calls), "count": len(calls), "calls": calls}

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
        # A late webhook must not pull a finished job back onto the line.
        # Status is applied with a conditional write so a snapshot taken before
        # our own hangup cannot demote `completed`. `failed` may still move to
        # `completed` when call.hangup follows streaming.failed.
        new_job_status = fields.pop("status", None)
        if new_job_status and not _webhook_status_allowed(job.status, new_job_status):
            new_job_status = None
        updated = job
        if fields:
            updated = self.store.update(job.id, **fields) or updated
        if new_job_status:
            updated, changed = self.store.update_call_status(job.id, new_job_status)
            if changed and updated and event_type == "call.hangup":
                self._maybe_callback(updated)
        # Hangup from either side, streaming failure, and later the media-session
        # close all schedule this. The claim inside the task makes it run once.
        if event_type in {"call.hangup", "streaming.failed"}:
            self._schedule_post_call_memory(job.id)

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

    def _schedule_post_call_memory(self, job_id: str) -> None:
        from stella import post_call_memory

        post_call_memory.start(self.settings, self.store, job_id, wait_for_media=True)

    def note_outcome(self, job_id: str, text: str) -> CallJob | None:
        """Spoken outcome. Never moves `completed` / `failed` / `rejected` back to completing."""
        return self.store.note_outcome(job_id, text, terminal_statuses=TERMINAL_STATUSES)

    def hangup(self, job: CallJob) -> bool:
        """Hang up the Telnyx leg. True when the leg is down, including error 90015."""
        if not job.telnyx_call_control_id:
            return False
        try:
            self.telnyx.hangup(job.telnyx_call_control_id)
        except StellaError as exc:
            if exc.code == "telnyx_call_ended":
                logger.info("hangup %s: telnyx call already ended", job.id)
                return True
            logger.warning("hangup failed for %s: %s", job.id, exc.message)
            return False
        return True

    def hangup_own_leg(self, job: CallJob) -> CallJob | None:
        """After our own hangup succeeds, mark the job completed. Do not wait for call.hangup."""
        if job.status in TERMINAL_STATUSES:
            return job
        if not self.hangup(job):
            return self.store.get(job.id)
        done = self.finish_call(job.id, status="completed")
        # Memory waits for the media socket to close. Status is already terminal.
        self._schedule_post_call_memory(job.id)
        return done

    def finish_call(
        self,
        job_id: str,
        *,
        status: str,
        error: str = "",
        outcome: str = "",
    ) -> CallJob | None:
        """Move a non-terminal job to a terminal status. A terminal job is left as it is."""
        job = self.store.get(job_id)
        if job is None:
            return None
        if job.status in TERMINAL_STATUSES:
            return job
        fields: dict[str, Any] = {"status": status}
        if error:
            fields["error"] = error[:2000]
        merged = _merge_outcome(job.outcome, outcome)
        if merged and merged != job.outcome:
            fields["outcome"] = merged
        updated, changed = self.store.update_unless(
            job_id,
            unless_status=TERMINAL_STATUSES,
            **fields,
        )
        if changed and updated is not None and updated.status in TERMINAL_STATUSES:
            logger.info("stella_call terminal job=%s status=%s", job_id, updated.status)
            self._maybe_callback(updated)
        return updated

    def finalize_session(self, job_id: str, *, failed: bool = False, error: str = "") -> None:
        """Media/WebSocket end: terminal status first, memory hook after, never the other way around."""
        try:
            current = self.store.get(job_id)
            status = "failed" if failed else "completed"
            # A reject-speak leg has no successful conversation. Telnyx 90015
            # still finishes as completed; that path does not come through here.
            if current is not None and status == "completed" and current.status == "rejecting":
                status = "rejected"
            self.finish_call(
                job_id,
                status=status,
                error=(error or "voice session failed") if failed else "",
            )
        except Exception:
            logger.exception("finish_call failed job=%s", job_id)
            try:
                self.store.update_unless(
                    job_id,
                    unless_status=TERMINAL_STATUSES,
                    status="failed",
                    error=(error or "voice session failed")[:2000],
                )
            except Exception:
                logger.exception("terminal status fallback failed job=%s", job_id)
        finally:
            self._schedule_post_call_memory(job_id)

    def plan_recovery(
        self,
        row: dict[str, Any],
        *,
        now: datetime,
        force: bool,
        liveness: str,
    ) -> tuple[str, str, bool] | None:
        """Decide whether a non-terminal row should be finished.

        Returns `(status, reason, hangup)` or None to leave the job alone.
        `hangup` is true when this process should still send Telnyx hangup.
        """
        status = str(row.get("status") or "")
        if status in TERMINAL_STATUSES or status not in NON_TERMINAL_STATUSES:
            return None
        ended = liveness in _ENDED_LIVENESS
        limit = (
            float(self.settings.stella_stale_completing_s)
            if status == "completing"
            else float(self.settings.stella_stale_call_s)
        )
        stale = _row_age_s(row, now) >= max(0.0, limit)
        if not force and not stale and not ended:
            return None
        if liveness == "ended_90015":
            reason = "auto-recovered: telnyx 90015 call already ended"
        elif ended:
            reason = "auto-recovered: telnyx call already ended"
        elif force:
            reason = "auto-recovered: manual reset"
        else:
            reason = f"auto-recovered: stale {status}"
        new_status = "completed" if ended else _end_status(status)
        ccid = str(row.get("telnyx_call_control_id") or "")
        return new_status, reason, bool(ccid) and not ended

    async def reconcile(self, *, force: bool = False, call_id: str | None = None) -> dict[str, Any]:
        """Finish non-terminal jobs that are stale or that Telnyx already ended.

        `force` ignores age (manual reset). Rows are updated, never deleted.
        Runs once at process start and on the reaper interval with `force` false.
        """
        if call_id:
            job = self.store.get(call_id)
            if job is None:
                raise StellaError(f"Unknown call id: {call_id}", "not_found")
            if job.status in TERMINAL_STATUSES:
                return {
                    "count": 0,
                    "reset": [],
                    "call": {"id": job.id, "status": job.status},
                }
        now = datetime.now(timezone.utc)
        reset: list[dict[str, Any]] = []
        for row in self.store.list_by_status(NON_TERMINAL_STATUSES):
            if call_id and row["id"] != call_id:
                continue
            liveness = "none"
            ccid = row.get("telnyx_call_control_id") or ""
            if ccid:
                liveness = await asyncio.to_thread(self.telnyx.call_liveness, ccid)
            plan = self.plan_recovery(row, now=now, force=force, liveness=liveness)
            if plan is None:
                continue
            new_status, reason, do_hangup = plan
            await call_sessions.close_session(row["id"])
            item = self.recover_job(
                row["id"],
                status=new_status,
                reason=reason,
                hangup=do_hangup,
            )
            if item:
                reset.append(item)
        return {"count": len(reset), "reset": reset}

    async def reset_stuck(self, call_id: str | None = None) -> dict[str, Any]:
        """Finish stuck calls now. `call_id` limits it to one job; empty means every open job."""
        return await self.reconcile(force=True, call_id=call_id)

    def recover_job(
        self,
        job_id: str,
        *,
        status: str,
        reason: str,
        hangup: bool,
    ) -> dict[str, Any] | None:
        job = self.store.get(job_id)
        if job is None or job.status in TERMINAL_STATUSES:
            return None
        if hangup and job.telnyx_call_control_id:
            self.hangup(job)
        before = job.status
        updated = self.finish_call(job_id, status=status, outcome=reason)
        if (
            updated is None
            or updated.status not in TERMINAL_STATUSES
            or reason not in (updated.outcome or "")
        ):
            return None
        logger.warning(
            "stella_reaper auto-recovered job=%s from=%s to=%s reason=%s",
            job_id,
            before,
            updated.status,
            reason,
        )
        try:
            self.store.add_event(
                job_id,
                "stella.auto_recovered",
                {"from_status": before, "status": updated.status, "reason": reason},
            )
        except Exception:
            logger.exception("stella_reaper event write failed job=%s", job_id)
        self._schedule_post_call_memory(job_id)
        return {
            "id": updated.id,
            "status": updated.status,
            "from_status": before,
            "outcome": updated.outcome,
        }

    def _maybe_callback(self, job: CallJob) -> None:
        url = (self.settings.stella_callback_url or "").strip()
        if not url:
            return
        try:
            httpx.post(url, json=job.to_public_dict(), timeout=10.0)
        except Exception:
            logger.exception("STELLA_CALLBACK_URL post failed")


def _webhook_status_allowed(current: str, new: str) -> bool:
    """Webhooks may finish a call. They may not reopen one that already ended."""
    if current not in TERMINAL_STATUSES:
        return True
    return current == "failed" and new == "completed"


def _end_status(status: str) -> str:
    """Terminal status for a leg we are closing ourselves (Telnyx did not already end it)."""
    if status in {"queued", "dialing", "initiated"}:
        return "failed"
    if status == "rejecting":
        return "rejected"
    return "completed"


def _merge_outcome(existing: str, extra: str) -> str:
    extra = (extra or "").strip()
    current = (existing or "").strip()
    if not extra:
        return current
    if not current:
        return extra[:2000]
    if extra in current:
        return current[:2000]
    return f"{current} ({extra})"[:2000]


def _row_age_s(row: dict[str, Any], now: datetime) -> float:
    raw = str(row.get("updated_at") or row.get("created_at") or "")
    try:
        text = raw.replace("Z", "+00:00")
        stamp = datetime.fromisoformat(text)
    except ValueError:
        return float("inf")
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return max(0.0, (now - stamp).total_seconds())


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
