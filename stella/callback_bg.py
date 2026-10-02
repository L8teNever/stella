"""Background callback booking: Claude + Ida Reminder, detached from the phone call.

The voice agent confirms a callback instantly (foreground); the real work runs here as a
detached task that keeps going after the call ended. Claude Code (with the Ida MCP servers)
creates an Ida Reminder so that Ida calls Simon back via Stella at the requested time. If that
fails or times out, Stella's own scheduler books the callback instead, so it never gets lost.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from stella import ask_ida as ask_ida_mod
from stella.config import Settings
from stella.errors import StellaError
from stella.jobs import BERLIN, JobService

logger = logging.getLogger(__name__)

BACKGROUND_TIMEOUT_S = 120.0
_FAILURES = {ask_ida_mod.MSG_FAILED, ask_ida_mod.MSG_TIMEOUT, ask_ida_mod.MSG_EMPTY, ask_ida_mod.MSG_BUSY}
_tasks: set[asyncio.Task] = set()


def build_request(plan: dict[str, Any]) -> str:
    run_at = plan["run_at"].astimezone(BERLIN)
    return (
        "Hintergrundauftrag, es wartet niemand auf eine Antwort. Lege mit dem Tool "
        "erinnerung_erstellen (Ida Reminder) eine Erinnerung an, die zum Zeitpunkt "
        f"{run_at:%Y-%m-%d %H:%M} (Europe/Berlin) Ida auslöst. Aufgabe der Erinnerung, wörtlich: "
        f"'Rückruf-Auftrag von Simon: Ruf ihn um {run_at:%H:%M} Uhr per Stella an "
        f"(stella_call, to={plan['to']}, speak_to={plan['speak_to']}). "
        f"Anlass: {plan['grund'] or 'kein besonderer Grund genannt'}. Danach diese Erinnerung leeren.' "
        "Sind alle Plätze belegt, nutze keinen belegten Platz und melde einen Fehler. "
        "Antworte nur mit dem einen Wort ERLEDIGT, wenn die Erinnerung angelegt ist, sonst mit "
        "FEHLER und einem kurzen Grund."
    )


async def run(settings: Settings, jobs: JobService, plan: dict[str, Any]) -> bool:
    """Book via Claude + Ida Reminder; fall back to Stella's scheduler. Returns True if booked."""
    answer = ""
    try:
        answer = await ask_ida_mod.ask_ida(
            settings, build_request(plan), timeout_s=BACKGROUND_TIMEOUT_S
        )
    except Exception:  # noqa: BLE001
        logger.exception("background callback via Claude crashed")
    ok = "ERLEDIGT" in answer.upper() and "FEHLER" not in answer.upper() and answer not in _FAILURES
    if ok:
        logger.info("callback booked via Ida Reminder for %s", plan["run_at"].isoformat())
        return True
    logger.warning("callback via Ida Reminder failed (%s); using Stella scheduler", answer[:200])
    try:
        jobs.schedule_call(
            to=plan["to"], brief=plan["brief"], run_at=plan["run_at"],
            speak_to=plan["speak_to"], allow_ida=plan["allow_ida"],
        )
        return True
    except StellaError as exc:
        logger.error("fallback scheduling failed: %s", exc.message)
        return False


def start(settings: Settings, jobs: JobService, plan: dict[str, Any]) -> asyncio.Task:
    """Spawn the detached task (survives the end of the phone call)."""
    task = asyncio.create_task(run(settings, jobs, plan))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return task
