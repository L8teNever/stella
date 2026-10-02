"""Background booking of "do X later" requests: Claude + Ida Reminder, detached from the call.

The voice agent confirms the request instantly (foreground); the real work runs here as a
detached task that keeps going after the call ended. Claude Code (with the Ida MCP servers)
creates an Ida Reminder whose task text tells Ida what to do at that time: call Simon back via
Stella, remind him, send a Telegram message, do something else. There is no internal scheduler:
Ida Reminder is the only place these tasks live.
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
        f"'Auftrag von Simon, am Telefon gegeben, auszuführen um {run_at:%H:%M} Uhr: "
        f"{plan['aufgabe']} "
        + (
            f"Gesprächskontext (Ausschnitt aus dem Telefonat, in dem Simon das aufgetragen hat): "
            f"<<<{plan['kontext']}>>> "
            if plan.get("kontext")
            else ""
        )
        + f"Hinweise: Anrufe an Simon über stella_call (to={plan['to']}, speak_to={plan['speak_to']}, "
        "allow_ida=true, context = der Gesprächskontext oben vollständig, brief = was Stella ihm "
        "sagen soll, mit dem konkreten Anlass, plus der Satz: 'Danach frag, ob Simon noch "
        "etwas braucht; weitere Wünsche von ihm führst du mit frag_ida aus, bevor du auflegst.'); "
        "Nachrichten an Simon über Telegram (nachricht_senden); "
        "alles andere mit den passenden Tools. Danach diese Erinnerung leeren.' "
        "Prüfe vorher mit erinnerungen_liste, ob es für diesen Zeitpunkt schon eine Erinnerung mit "
        "derselben Aufgabe gibt; dann lege nichts neu an und antworte ERLEDIGT. "
        "Sind alle Plätze belegt, nutze keinen belegten Platz und melde einen Fehler. "
        "Antworte nur mit dem einen Wort ERLEDIGT, wenn die Erinnerung angelegt ist, sonst mit "
        "FEHLER und einem kurzen Grund."
    )


def classify(answer: str) -> str:
    """'ok', 'retry' (clear failure, safe to retry) or 'unknown' (do not retry: may be booked)."""
    upper = (answer or "").upper()
    if "FEHLER" in upper or answer in _FAILURES or not upper.strip():
        return "retry"
    if "ERLED" in upper:  # also tolerates truncated replies like "ERLED"
        return "ok"
    return "unknown"


async def run(settings: Settings, plan: dict[str, Any]) -> bool:
    """Book via Claude + Ida Reminder. Retries once, only on a clear failure."""
    for attempt in (1, 2):
        answer = ""
        try:
            answer = await ask_ida_mod.ask_ida(
                settings, build_request(plan), timeout_s=BACKGROUND_TIMEOUT_S
            )
        except Exception:  # noqa: BLE001
            logger.exception("background task booking via Claude crashed")
        verdict = classify(answer)
        if verdict == "ok":
            logger.info("task booked via Ida Reminder for %s", plan["run_at"].isoformat())
            return True
        if verdict == "unknown":
            logger.warning("task booking unclear (not retrying to avoid duplicates): %s", answer[:200])
            return False
        logger.warning("task booking attempt %d failed: %s", attempt, answer[:200])
    logger.error("task for %s could NOT be booked", plan["run_at"].isoformat())
    return False


def start(settings: Settings, plan: dict[str, Any]) -> asyncio.Task:
    """Spawn the detached task (survives the end of the phone call)."""
    task = asyncio.create_task(run(settings, plan))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return task
