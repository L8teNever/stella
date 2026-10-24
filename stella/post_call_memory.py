"""Write durable facts from an owner call into the Memory MCP server.

The job is already in a terminal status before this starts. The task is
detached and capped by STELLA_POST_CALL_MEMORY_TIMEOUT_S; a timeout or any
other error is written only to memory_summary. A job is claimed once in
SQLite, so repeated hangup events do not extract or write twice.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from stella import ask_ida as ask_ida_mod
from stella.config import Settings
from stella.live_mcp import LiveMcpPool, McpServerSpec, parse_mcp_servers, prefix_tool
from stella.store import JobStore
from stella.voice import is_owner_call

logger = logging.getLogger(__name__)

# Knowledge-graph tools. delete_* is intentionally absent.
WRITE_TOOLS = ("search_nodes", "create_entities", "add_observations", "create_relations")
MEDIA_WAIT_S = 20.0
_POLL_S = 0.05
# Phone lookups use ASK_IDA_TIMEOUT_S (often 25s). Extraction has no tools and
# runs after hangup, but a long transcript still needs a floor.
EXTRACT_TIMEOUT_FLOOR_S = 45.0
MEMORY_TOOL_TIMEOUT_S = 20.0

_FAILURES = {
    ask_ida_mod.MSG_FAILED,
    ask_ida_mod.MSG_TIMEOUT,
    ask_ida_mod.MSG_EMPTY,
    ask_ida_mod.MSG_BUSY,
}
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_media_open: set[str] = set()
_tasks: set[asyncio.Task] = set()

CallTool = Callable[[str, dict[str, Any]], Awaitable[dict[str, str]]]


class MemoryError(Exception):
    """Extraction or memory write failed. The message is safe to store on the job."""


@dataclass(frozen=True)
class Fact:
    entity: str
    entity_type: str
    observation: str
    relation_to: str = ""
    relation_to_type: str = ""
    relation_type: str = ""


EXTRACTION_SYSTEM = (
    "Du extrahierst dauerhaft relevante Fakten aus einem Telefongespräch zwischen Stella und Simon. "
    "Nur neue Informationen, die später noch stimmen sollen: neue Infos über Simon, andere Personen, "
    "Termine und Zusagen, Vorlieben, Entscheidungen, Aufgaben. "
    "Kein Smalltalk, keine Begrüßung, keine Tests, keine Vermutungen, nichts Doppeltes. "
    "Wenn nichts dauerhaft Relevant ist, antworte exakt mit {\"facts\":[]}. "
    "Sonst nur JSON, kein Markdown: "
    '{"facts":[{"entity":"Simon","entity_type":"person","observation":"ein kurzer deutscher Satz",'
    '"relation":null}]}. '
    "relation ist optional und sonst {\"to\":\"Name\",\"to_type\":\"person\",\"relation_type\":\"kennt\"}. "
    "entity trägt die Beobachtung. Erfinde nichts, was nicht im Transkript steht."
)


def note_media_open(job_id: str) -> None:
    _media_open.add(job_id)


def note_media_closed(job_id: str) -> None:
    _media_open.discard(job_id)


def memory_server(settings: Settings) -> McpServerSpec | None:
    wanted = (settings.stella_post_call_memory_server or "memory").strip() or "memory"
    for spec in parse_mcp_servers(settings.stella_mcp_servers):
        if spec.name == wanted:
            return spec
    return None


def writable_spec(spec: McpServerSpec) -> McpServerSpec | None:
    """Same server, but only the four write/search tools. Never delete_*."""
    if spec.allow is None:
        allow = WRITE_TOOLS
    else:
        allow = tuple(tool for tool in WRITE_TOOLS if tool in spec.allow)
    needed = {"search_nodes", "create_entities", "add_observations"}
    if not needed.issubset(allow):
        return None
    return McpServerSpec(name=spec.name, url=spec.url, token_env=spec.token_env, allow=allow)


def refuses_tool(name: str) -> bool:
    lowered = (name or "").strip().lower()
    if not lowered or lowered not in WRITE_TOOLS:
        return True
    return "delete" in lowered or "loeschen" in lowered


def extraction_argv(settings: Settings) -> list[str]:
    """Claude command for fact extraction. No allowed tools, delete tools denied."""
    return ask_ida_mod.claude_argv(
        settings,
        system_prompt=EXTRACTION_SYSTEM,
        max_turns=1,
        mcp_config=ask_ida_mod.empty_mcp_config(settings),
        allowed_tools=None,
    )


def extraction_prompt(transcript: str) -> str:
    text = (transcript or "").strip()[-12000:]
    return "Transkript:\n" + text


def parse_facts(text: str) -> list[Fact]:
    raw = (text or "").strip()
    if not raw or raw in _FAILURES:
        raise MemoryError(raw or "keine Antwort")
    match = _JSON_OBJECT.search(raw)
    if not match:
        lowered = raw.lower()
        if lowered in {"keine", "nichts", "none", "[]"} or "nichts relevant" in lowered:
            return []
        raise MemoryError("Antwort war kein JSON")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise MemoryError("Antwort war kein gültiges JSON") from exc
    items = data.get("facts") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    facts: list[Fact] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        entity = str(item.get("entity") or "").strip()[:80]
        observation = " ".join(str(item.get("observation") or "").split())[:400]
        if not entity or not observation:
            continue
        key = (entity.casefold(), observation.casefold())
        if key in seen:
            continue
        seen.add(key)
        relation = item.get("relation") if isinstance(item.get("relation"), dict) else {}
        facts.append(
            Fact(
                entity=entity,
                entity_type=str(item.get("entity_type") or "fact").strip()[:40] or "fact",
                observation=observation,
                relation_to=str(relation.get("to") or "").strip()[:80],
                relation_to_type=str(relation.get("to_type") or "thing").strip()[:40] or "thing",
                relation_type=str(relation.get("relation_type") or "").strip()[:40],
            )
        )
    return facts


def _entities_from_search(text: str) -> list[dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            return []
        try:
            data = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return []
    if isinstance(data, dict) and isinstance(data.get("entities"), list):
        return [item for item in data["entities"] if isinstance(item, dict)]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def _known_observations(entities: list[dict[str, Any]], name: str) -> set[str] | None:
    """Observations already stored, or None when that entity was not in the search hit."""
    found: set[str] | None = None
    for entity in entities:
        if str(entity.get("name") or "").strip().casefold() != name.casefold():
            continue
        found = found or set()
        for observation in entity.get("observations") or []:
            text = " ".join(str(observation).split())
            if text:
                found.add(text.casefold())
    return found


async def guarded_call(call_tool: CallTool, name: str, args: dict[str, Any]) -> dict[str, str]:
    if refuses_tool(name):
        logger.info("stella_memory refuse tool=%s", name)
        return {"error": "refused"}
    return await call_tool(name, args)


async def apply_facts(facts: list[Fact], call_tool: CallTool) -> list[str]:
    """Search first, then create or add observations. Never calls delete_*."""
    written: list[str] = []
    cache: dict[str, set[str] | None] = {}

    async def lookup(name: str) -> set[str] | None:
        if name.casefold() in cache:
            return cache[name.casefold()]
        result = await guarded_call(call_tool, "search_nodes", {"query": name})
        if result.get("error"):
            raise MemoryError(str(result["error"])[:200])
        known = _known_observations(_entities_from_search(result.get("result") or ""), name)
        cache[name.casefold()] = known
        return known

    for fact in facts:
        known = await lookup(fact.entity)
        if known is not None and fact.observation.casefold() in known:
            continue
        if known is None:
            created = await guarded_call(
                call_tool,
                "create_entities",
                {
                    "entities": [
                        {
                            "name": fact.entity,
                            "entityType": fact.entity_type or "fact",
                            "observations": [fact.observation],
                        }
                    ]
                },
            )
            if created.get("error"):
                raise MemoryError(str(created["error"])[:200])
            cache[fact.entity.casefold()] = {fact.observation.casefold()}
        else:
            added = await guarded_call(
                call_tool,
                "add_observations",
                {"observations": [{"entityName": fact.entity, "contents": [fact.observation]}]},
            )
            if added.get("error"):
                raise MemoryError(str(added["error"])[:200])
            known.add(fact.observation.casefold())
        line = f"{fact.entity}: {fact.observation}"
        if fact.relation_to and fact.relation_type:
            target = await lookup(fact.relation_to)
            if target is None:
                created = await guarded_call(
                    call_tool,
                    "create_entities",
                    {
                        "entities": [
                            {
                                "name": fact.relation_to,
                                "entityType": fact.relation_to_type or "thing",
                                "observations": [],
                            }
                        ]
                    },
                )
                if created.get("error"):
                    raise MemoryError(str(created["error"])[:200])
                cache[fact.relation_to.casefold()] = set()
            linked = await guarded_call(
                call_tool,
                "create_relations",
                {
                    "relations": [
                        {
                            "from": fact.entity,
                            "to": fact.relation_to,
                            "relationType": fact.relation_type,
                        }
                    ]
                },
            )
            if linked.get("error"):
                raise MemoryError(str(linked["error"])[:200])
            line += f" ({fact.relation_type} {fact.relation_to})"
        written.append(line)
    return written


async def extract_facts(
    settings: Settings,
    transcript: str,
    *,
    spawn: ask_ida_mod.Spawn | None = None,
) -> list[Fact]:
    timeout = max(float(settings.ask_ida_timeout_s or 0), EXTRACT_TIMEOUT_FLOOR_S)
    answer = await ask_ida_mod.run_claude(
        settings,
        extraction_prompt(transcript),
        system_prompt=EXTRACTION_SYSTEM,
        allowed_tools=None,
        max_turns=1,
        timeout_s=timeout,
        spawn=spawn,
        mcp_config=ask_ida_mod.empty_mcp_config(settings),
    )
    return parse_facts(answer)


async def write_facts(settings: Settings, spec: McpServerSpec, facts: list[Fact]) -> list[str]:
    safe = writable_spec(spec)
    if safe is None:
        raise MemoryError("Memory-Server erlaubt die nötigen Tools nicht")
    timeout = max(MEMORY_TOOL_TIMEOUT_S, float(settings.stella_mcp_tool_timeout_s or 0))
    pool = LiveMcpPool([safe], tool_timeout_s=timeout, connect_timeout_s=15.0)
    await pool.open()
    try:
        if prefix_tool(safe.name, "search_nodes") not in pool:
            raise MemoryError("Memory-Server nicht erreichbar")

        async def call_tool(name: str, args: dict[str, Any]) -> dict[str, str]:
            if refuses_tool(name):
                return {"error": "refused"}
            prefixed = prefix_tool(safe.name, name)
            result = await pool.call(prefixed, args)
            if "error" in result and "result" not in result:
                return result
            return result

        return await apply_facts(facts, call_tool)
    finally:
        await pool.close()


def _finish(store: JobStore, job_id: str, summary: str, written: list[str]) -> None:
    store.update(
        job_id,
        memory_summary=summary[:500],
        memory_written=json.dumps(written, ensure_ascii=False),
    )


async def _wait_for_media(job_id: str, wait_for_media: bool, max_wait_s: float) -> None:
    if not wait_for_media or job_id not in _media_open:
        return
    deadline = time.monotonic() + max(0.0, max_wait_s)
    while job_id in _media_open and time.monotonic() < deadline:
        await asyncio.sleep(_POLL_S)


async def run(
    settings: Settings,
    store: JobStore,
    job_id: str,
    *,
    wait_for_media: bool = False,
    max_wait_s: float = MEDIA_WAIT_S,
    extract: Callable[..., Awaitable[list[Fact]]] | None = None,
    write: Callable[..., Awaitable[list[str]]] | None = None,
) -> None:
    extract_fn = extract or extract_facts
    write_fn = write or write_facts
    try:
        await _wait_for_media(job_id, wait_for_media, max_wait_s)
        job = store.get(job_id)
        if job is None:
            return
        if not settings.stella_post_call_memory:
            if store.claim_memory(job_id):
                _finish(store, job_id, "deaktiviert", [])
                logger.info("stella_memory skip reason=disabled job=%s", job_id)
            return
        if not is_owner_call(settings, job.to_number):
            if store.claim_memory(job_id):
                _finish(store, job_id, "übersprungen: kein Owner-Anruf", [])
                logger.info("stella_memory skip reason=not_owner job=%s", job_id)
            return
        spec = memory_server(settings)
        if spec is None:
            if store.claim_memory(job_id):
                name = (settings.stella_post_call_memory_server or "memory").strip() or "memory"
                _finish(store, job_id, "kein Memory-Server konfiguriert", [])
                logger.info("stella_memory skip reason=no_server server=%s job=%s", name, job_id)
            return
        if not store.claim_memory(job_id):
            return
        transcript = (store.get(job_id).transcript or "").strip() if store.get(job_id) else ""
        if not transcript:
            _finish(store, job_id, "nichts Relevantes", [])
            logger.info("stella_memory skip reason=empty job=%s", job_id)
            return
        try:
            facts = await extract_fn(settings, transcript)
        except MemoryError as exc:
            _finish(store, job_id, f"Fehler: {exc}", [])
            logger.info("stella_memory skip reason=extract job=%s", job_id)
            return
        if not facts:
            _finish(store, job_id, "nichts Relevantes", [])
            logger.info("stella_memory skip reason=empty job=%s", job_id)
            return
        try:
            written = await write_fn(settings, spec, facts)
        except MemoryError as exc:
            _finish(store, job_id, f"Fehler: {exc}", [])
            logger.info("stella_memory skip reason=write job=%s", job_id)
            return
        if not written:
            _finish(store, job_id, "nichts Relevantes", [])
            logger.info("stella_memory skip reason=duplicate job=%s", job_id)
            return
        label = "1 Fakt geschrieben." if len(written) == 1 else f"{len(written)} Fakten geschrieben."
        _finish(store, job_id, label, written)
        logger.info("stella_memory job=%s facts=%d", job_id, len(written))
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("stella_memory failed job=%s", job_id)
        try:
            current = store.get(job_id)
            if current is not None and current.memory_done and not current.memory_summary:
                _finish(store, job_id, "Fehler: Memory-Lauf abgebrochen", [])
        except Exception:
            logger.exception("stella_memory status update failed job=%s", job_id)


def _note_memory_timeout(store: JobStore, job_id: str) -> None:
    """Record a hard timeout. Does not touch call status."""
    current = store.get(job_id)
    if current is None or current.memory_summary:
        return
    if not current.memory_done and not store.claim_memory(job_id):
        current = store.get(job_id)
        if current is None or current.memory_summary:
            return
    _finish(store, job_id, f"Fehler: {ask_ida_mod.MSG_TIMEOUT}", [])


async def _run_bounded(
    settings: Settings,
    store: JobStore,
    job_id: str,
    *,
    wait_for_media: bool,
    max_wait_s: float,
) -> None:
    timeout = float(settings.stella_post_call_memory_timeout_s or 180.0)
    if timeout <= 0:
        timeout = 180.0
    try:
        await asyncio.wait_for(
            run(
                settings,
                store,
                job_id,
                wait_for_media=wait_for_media,
                max_wait_s=max_wait_s,
            ),
            timeout=timeout,
        )
    except TimeoutError:
        logger.warning("stella_memory timeout job=%s", job_id)
        try:
            _note_memory_timeout(store, job_id)
        except Exception:
            logger.exception("stella_memory timeout update failed job=%s", job_id)
    except asyncio.CancelledError:
        raise


def start(
    settings: Settings,
    store: JobStore,
    job_id: str,
    *,
    wait_for_media: bool = False,
    max_wait_s: float = MEDIA_WAIT_S,
) -> asyncio.Task | None:
    """Schedule the memory pass. Returns None when no event loop is running.

    The task does not block the caller and cannot move the job out of a
    terminal status. A hard timeout writes the error into memory_summary only.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    task = loop.create_task(
        _run_bounded(
            settings,
            store,
            job_id,
            wait_for_media=wait_for_media,
            max_wait_s=max_wait_s,
        ),
        name=f"stella-memory-{job_id}",
    )
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return task
