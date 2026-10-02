"""Background callback booking: Claude + Ida Reminder, detached from the phone call.

The voice agent confirms a callback instantly (foreground); the real work runs here as a
detached task that keeps going after the call ended. Claude Code (with the Ida MCP servers)
creates an Ida Reminder so that Ida calls Simon back via Stella at the requested time. There is
no internal scheduler: Ida Reminder is the only place callbacks live.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from stella import ask_ida as ask_ida_mod
from stella.config import Settings
from stella.jobs import BERLIN

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


async def run(settings: Settings, plan: dict[str, Any]) -> bool:
    """Book via Claude + Ida Reminder (one retry). Returns True if the reminder was created."""
    for attempt in (1, 2):
        answer = ""
        try:
            answer = await ask_ida_mod.ask_ida(
                settings, build_request(plan), timeout_s=BACKGROUND_TIMEOUT_S
            )
        except Exception:  # noqa: BLE001
            logger.exception("background callback via Claude crashed")
        upper = answer.upper()
        if "ERLEDIGT" in upper and "FEHLER" not in upper and answer not in _FAILURES:
            logger.info("callback booked via Ida Reminder for %s", plan["run_at"].isoformat())
            return True
        logger.warning("callback booking attempt %d failed: %s", attempt, answer[:200])
    logger.error("callback for %s could NOT be booked", plan["run_at"].isoformat())
    return False


def start(settings: Settings, plan: dict[str, Any]) -> asyncio.Task:
    """Spawn the detached task (survives the end of the phone call)."""
    task = asyncio.create_task(run(settings, plan))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return task
