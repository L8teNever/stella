"""Post-call memory: once per call, owner only, empty extraction writes nothing, never delete."""

from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest

from stella import post_call_memory as pcm
from stella.jobs import JobService
from stella.live_mcp import McpServerSpec
from stella.store import JobStore
from stella.telnyx_client import TelnyxClient
from tests.test_stella import OWNER, FakeTransport, make_settings

MEMORY_SERVERS = json.dumps(
    [
        {
            "name": "memory",
            "url": "https://memory.example/mcp",
            "token_env": "MEMORY_MCP_TOKEN",
            "allow": [
                "search_nodes",
                "create_entities",
                "add_observations",
                "create_relations",
                "delete_entities",
                "delete_observations",
                "delete_relations",
            ],
        }
    ]
)


def _service(tmp_path, **kwargs):
    defaults = dict(
        stella_owner_number=OWNER,
        stella_mcp_servers=MEMORY_SERVERS,
        stella_post_call_memory=True,
        ask_ida_model="claude-haiku-5-5",
    )
    defaults.update(kwargs)
    settings = make_settings(tmp_path, **defaults)
    store = JobStore(settings.stella_db_path)
    svc = JobService(settings, store, TelnyxClient(settings, FakeTransport()))
    return settings, store, svc


def _job(store, *, to_number=OWNER, transcript="Simon: Ich mag Tee am Abend.", ccid="cc-mem"):
    job = store.create(kind="call", to_number=to_number, brief="Rückruf")
    store.update(job.id, status="in_progress", telnyx_call_control_id=ccid)
    if transcript:
        store.append_transcript(job.id, transcript)
    return job


def _hangup(ccid="cc-mem", event="call.hangup"):
    return {
        "data": {
            "event_type": event,
            "payload": {"call_control_id": ccid, "hangup_cause": "normal_clearing", "reason": "timeout"},
        }
    }


async def _drain():
    tasks = [task for task in list(pcm._tasks) if not task.done()]
    if tasks:
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)


@pytest.fixture(autouse=True)
def _clear_media():
    pcm._media_open.clear()
    yield
    pcm._media_open.clear()


def test_extraction_command_uses_model_and_denies_delete(tmp_path):
    settings = make_settings(tmp_path, ask_ida_model="claude-haiku-5-5")
    argv = pcm.extraction_argv(settings)
    assert argv[argv.index("--model") + 1] == "claude-haiku-5-5"
    assert "--allowedTools" not in argv
    denied = argv[argv.index("--disallowedTools") + 1]
    assert "delete" in denied
    safe = pcm.writable_spec(
        McpServerSpec(
            name="memory",
            url="https://memory.example/mcp",
            allow=tuple(json.loads(MEMORY_SERVERS)[0]["allow"]),
        )
    )
    assert safe is not None
    assert "delete_entities" not in safe.allow
    assert "delete_observations" not in safe.allow
    assert "delete_relations" not in safe.allow
    assert set(safe.allow) == set(pcm.WRITE_TOOLS)


async def test_extract_spawn_has_no_delete_tool(tmp_path):
    settings = make_settings(tmp_path, ask_ida_model="claude-haiku-5-5")
    seen: dict = {}

    class Proc:
        returncode = 0

        async def communicate(self, data):
            return (b'{"facts":[]}', b"")

    async def spawn(*argv, **kwargs):
        seen["argv"] = argv
        return Proc()

    assert await pcm.extract_facts(settings, "Nur Hallo, wie geht's?", spawn=spawn) == []
    argv = seen["argv"]
    assert argv[argv.index("--model") + 1] == "claude-haiku-5-5"
    assert "--allowedTools" not in argv
    assert all(not str(part).startswith("delete_") for part in argv)


async def test_apply_facts_never_calls_delete():
    seen: list[str] = []

    async def call_tool(name, args):
        seen.append(name)
        if name == "search_nodes":
            return {"result": json.dumps({"entities": [], "relations": []})}
        return {"result": "ok"}

    for banned in ("delete_entities", "delete_observations", "delete_relations", "drop_graph"):
        refused = await pcm.guarded_call(call_tool, banned, {"entities": ["Simon"]})
        assert refused == {"error": "refused"}
    written = await pcm.apply_facts(
        [
            pcm.Fact(
                entity="Simon",
                entity_type="person",
                observation="Mag Tee am Abend.",
                relation_to="Tee",
                relation_to_type="thing",
                relation_type="trinkt",
            )
        ],
        call_tool,
    )
    assert written == ["Simon: Mag Tee am Abend. (trinkt Tee)"]
    assert seen
    assert all("delete" not in name and name in pcm.WRITE_TOOLS for name in seen)
    assert "create_entities" in seen and "search_nodes" in seen and "create_relations" in seen


async def test_apply_facts_updates_existing_entity():
    seen: list[str] = []

    async def call_tool(name, args):
        seen.append(name)
        if name == "search_nodes":
            return {
                "result": json.dumps(
                    {
                        "entities": [
                            {"name": "Simon", "entityType": "person", "observations": ["Wohnt in Berlin."]}
                        ]
                    }
                )
            }
        return {"result": "ok"}

    written = await pcm.apply_facts(
        [pcm.Fact(entity="Simon", entity_type="person", observation="Mag Tee am Abend.")],
        call_tool,
    )
    assert written == ["Simon: Mag Tee am Abend."]
    assert "add_observations" in seen
    assert "create_entities" not in seen
    assert all("delete" not in name for name in seen)


async def test_hook_runs_once(tmp_path, monkeypatch):
    settings, store, svc = _service(tmp_path)
    job = _job(store)
    extracts: list[str] = []
    writes: list[int] = []

    async def extract(_settings, transcript, **kwargs):
        extracts.append(transcript)
        await asyncio.sleep(0.02)
        return [pcm.Fact(entity="Simon", entity_type="person", observation="Mag Tee am Abend.")]

    async def write(_settings, spec, facts):
        writes.append(len(facts))
        assert spec.name == "memory"
        assert all("delete" not in fact.observation for fact in facts)
        return ["Simon: Mag Tee am Abend."]

    monkeypatch.setattr(pcm, "extract_facts", extract)
    monkeypatch.setattr(pcm, "write_facts", write)
    svc.handle_telnyx_event(_hangup(event="streaming.failed"))
    svc.handle_telnyx_event(_hangup())
    svc.handle_telnyx_event(_hangup())
    await _drain()
    assert extracts == ["Simon: Ich mag Tee am Abend."]
    assert writes == [1]
    public = store.get(job.id).to_public_dict()
    assert public["memory_summary"] == "1 Fakt geschrieben."
    assert public["memory_written"] == ["Simon: Mag Tee am Abend."]
    assert public["status"] == "completed"


async def test_hook_waits_until_media_closes(tmp_path, monkeypatch):
    settings, store, svc = _service(tmp_path)
    job = _job(store, ccid="cc-media")
    extracts: list[int] = []

    async def extract(_settings, transcript, **kwargs):
        extracts.append(1)
        return [pcm.Fact(entity="Simon", entity_type="person", observation="Mag Tee am Abend.")]

    async def write(_settings, spec, facts):
        return ["Simon: Mag Tee am Abend."]

    monkeypatch.setattr(pcm, "extract_facts", extract)
    monkeypatch.setattr(pcm, "write_facts", write)
    pcm.note_media_open(job.id)
    svc.handle_telnyx_event(_hangup("cc-media"))
    svc.handle_telnyx_event(_hangup("cc-media"))
    await asyncio.sleep(0.12)
    assert extracts == []
    pcm.note_media_closed(job.id)
    pcm.start(settings, store, job.id, wait_for_media=False)
    await _drain()
    assert extracts == [1]


async def test_skips_non_owner(tmp_path, monkeypatch):
    _settings, store, svc = _service(tmp_path)
    job = _job(store, to_number="+14155552671", ccid="cc-other")
    extracts: list[int] = []

    async def extract(_settings, transcript, **kwargs):
        extracts.append(1)
        return [pcm.Fact(entity="Simon", entity_type="person", observation="Geheim.")]

    async def write(_settings, spec, facts):
        raise AssertionError("write must not run for a foreign number")

    monkeypatch.setattr(pcm, "extract_facts", extract)
    monkeypatch.setattr(pcm, "write_facts", write)
    svc.handle_telnyx_event(_hangup("cc-other"))
    await _drain()
    assert extracts == []
    public = store.get(job.id).to_public_dict()
    assert public["memory_summary"] == "übersprungen: kein Owner-Anruf"
    assert public["memory_written"] == []


async def test_skips_empty_extraction(tmp_path, monkeypatch):
    _settings, store, svc = _service(tmp_path)
    job = _job(store, transcript="Stella: Ja, Simon?\nSimon: Nichts, nur kurz hallo.")
    writes: list[int] = []

    async def extract(_settings, transcript, **kwargs):
        return []

    async def write(_settings, spec, facts):
        writes.append(1)
        return ["should-not-write"]

    monkeypatch.setattr(pcm, "extract_facts", extract)
    monkeypatch.setattr(pcm, "write_facts", write)
    svc.handle_telnyx_event(_hangup())
    await _drain()
    assert writes == []
    public = store.get(job.id).to_public_dict()
    assert public["memory_summary"] == "nichts Relevantes"
    assert public["memory_written"] == []


async def test_skips_when_memory_server_missing(tmp_path, monkeypatch, caplog):
    _settings, store, svc = _service(tmp_path, stella_mcp_servers="")
    job = _job(store, ccid="cc-noserver")
    extracts: list[int] = []

    async def extract(_settings, transcript, **kwargs):
        extracts.append(1)
        return [pcm.Fact(entity="Simon", entity_type="person", observation="Mag Tee.")]

    monkeypatch.setattr(pcm, "extract_facts", extract)
    with caplog.at_level("INFO"):
        svc.handle_telnyx_event(_hangup("cc-noserver"))
        await _drain()
    assert extracts == []
    assert "reason=no_server" in caplog.text
    assert store.get(job.id).to_public_dict()["memory_summary"] == "kein Memory-Server konfiguriert"


def test_memory_columns_migrate(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE jobs (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            to_number TEXT NOT NULL,
            brief TEXT NOT NULL,
            context TEXT NOT NULL DEFAULT '',
            speak_to TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL,
            telnyx_call_control_id TEXT NOT NULL DEFAULT '',
            telnyx_call_leg_id TEXT NOT NULL DEFAULT '',
            outcome TEXT NOT NULL DEFAULT '',
            transcript TEXT NOT NULL DEFAULT '',
            error TEXT NOT NULL DEFAULT '',
            voice_provider TEXT NOT NULL DEFAULT '',
            allow_ida INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO jobs (id, kind, to_number, brief, status, created_at, updated_at) "
        "VALUES ('j', 'call', ?, 'b', 'completed', 't', 't')",
        (OWNER,),
    )
    conn.commit()
    conn.close()
    store = JobStore(str(path))
    job = store.get("j")
    assert job is not None and job.memory_done is False
    assert job.to_public_dict()["memory_written"] == []
    assert job.to_public_dict()["memory_summary"] is None
    assert store.claim_memory("j") is True
    assert store.claim_memory("j") is False
