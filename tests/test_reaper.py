"""Call completion does not wait for call.hangup, and stuck jobs recover themselves."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from stella import call_sessions, post_call_memory as pcm
from stella.ask_ida import MSG_TIMEOUT
from stella.jobs import JobService, TERMINAL_STATUSES
from stella.store import JobStore
from stella.telnyx_client import TelnyxClient
from tests.test_stella import FakeTransport, make_settings


def _backdate(store: JobStore, job_id: str, seconds: float) -> None:
    stamp = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()
    conn = sqlite3.connect(store._path)
    conn.execute("UPDATE jobs SET updated_at = ? WHERE id = ?", (stamp, job_id))
    conn.commit()
    conn.close()


class LivenessTransport:
    """GET /calls/{id} reports liveness; hangup is recorded and succeeds."""

    def __init__(self, liveness: str = "alive") -> None:
        self.liveness = liveness
        self.calls: list[tuple[str, str]] = []

    def request(self, method: str, url: str, **kwargs):
        self.calls.append((method, url))
        if method == "GET" and "/calls/" in url and "/actions/" not in url:
            if self.liveness == "ended_90015":
                return httpx.Response(
                    422,
                    json={
                        "errors": [
                            {
                                "code": "90015",
                                "title": "Call has already ended",
                                "detail": "Call has already ended",
                            }
                        ]
                    },
                )
            if self.liveness == "ended":
                return httpx.Response(
                    200,
                    json={"data": {"is_alive": False, "call_control_id": "cc-stale"}},
                )
            if self.liveness == "unknown":
                return httpx.Response(503, json={"errors": [{"code": "10000", "detail": "unavailable"}]})
            return httpx.Response(
                200,
                json={"data": {"is_alive": True, "call_control_id": "cc-stale"}},
            )
        if method == "POST" and url.endswith("/actions/hangup"):
            if self.liveness == "ended_90015":
                return httpx.Response(
                    422,
                    json={"errors": [{"code": 90015, "detail": "Call has already ended"}]},
                )
            return httpx.Response(200, json={"data": {"result": "ok"}})
        return httpx.Response(404, json={"errors": [{"code": "10005", "detail": "not mocked"}]})


def _service(tmp_path, transport=None, **kwargs):
    defaults = dict(stella_post_call_memory=False, stella_reaper_interval_s=60)
    defaults.update(kwargs)
    settings = make_settings(tmp_path, **defaults)
    store = JobStore(settings.stella_db_path)
    telnyx = TelnyxClient(settings, transport or FakeTransport())
    return settings, store, JobService(settings, store, telnyx), telnyx


def _open(store: JobStore, *, status: str = "completing", ccid: str = "", outcome: str = ""):
    job = store.create(kind="briefing", to_number="+14155552671", brief="Morning note.")
    fields = {"status": status}
    if ccid:
        fields["telnyx_call_control_id"] = ccid
    if outcome:
        fields["outcome"] = outcome
    store.update(job.id, **fields)
    return store.get(job.id)


@pytest.fixture(autouse=True)
def _clear_sessions():
    call_sessions.clear()
    pcm._media_open.clear()
    yield
    call_sessions.clear()
    pcm._media_open.clear()


async def _drain():
    tasks = [task for task in list(pcm._tasks) if not task.done()]
    if tasks:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=2)


def test_call_liveness_reads_90015_and_is_alive(tmp_path):
    settings = make_settings(tmp_path)
    ended = TelnyxClient(settings, LivenessTransport("ended_90015"))
    assert ended.call_liveness("cc-1") == "ended_90015"
    assert TelnyxClient(settings, LivenessTransport("ended")).call_liveness("cc-1") == "ended"
    assert TelnyxClient(settings, LivenessTransport("alive")).call_liveness("cc-1") == "alive"
    assert TelnyxClient(settings, LivenessTransport("unknown")).call_liveness("cc-1") == "unknown"


def test_own_hangup_completes_without_webhook_and_outcome_cannot_demote(tmp_path):
    _settings, store, svc, telnyx = _service(tmp_path, LivenessTransport("alive"))
    job = _open(store, status="in_progress", ccid="cc-own", outcome="")
    svc.note_outcome(job.id, "Briefing fertig.")
    mid = store.get(job.id)
    assert mid is not None and mid.status == "completing"
    assert mid.outcome == "Briefing fertig."
    assert svc.busy()["busy"] is False

    done = svc.hangup_own_leg(store.get(job.id))
    assert done is not None and done.status == "completed"
    assert done.outcome == "Briefing fertig."
    assert any(url.endswith("/actions/hangup") for _method, url in telnyx._http.calls)
    # No call.hangup webhook. A late outcome and a late answered event stay terminal.
    svc.note_outcome(job.id, "noch einmal")
    svc.handle_telnyx_event(
        {"data": {"event_type": "call.answered", "payload": {"call_control_id": "cc-own"}}}
    )
    final = store.get(job.id)
    assert final.status == "completed"
    assert final.outcome == "Briefing fertig."
    assert svc.busy() == {"busy": False, "count": 0, "calls": []}


def test_hangup_90015_counts_as_already_ended(tmp_path):
    _settings, store, svc, _telnyx = _service(tmp_path, LivenessTransport("ended_90015"))
    job = _open(store, status="completing", ccid="cc-gone", outcome="Briefing fertig.")
    done = svc.hangup_own_leg(job)
    assert done is not None and done.status == "completed"
    assert done.outcome == "Briefing fertig."
    assert "completed" in TERMINAL_STATUSES


def test_media_close_completes_when_hangup_webhook_never_arrives(tmp_path):
    _settings, store, svc, _telnyx = _service(tmp_path)
    job = _open(store, status="in_progress", ccid="cc-media")
    svc.note_outcome(job.id, "Briefing fertig.")
    assert store.get(job.id).status == "completing"
    svc.finalize_session(job.id)
    done = store.get(job.id)
    assert done.status == "completed"
    assert done.outcome == "Briefing fertig."
    assert svc.busy()["busy"] is False


def test_completing_does_not_block_the_next_call(tmp_path):
    _settings, store, svc, telnyx = _service(tmp_path)
    stuck = _open(store, status="completing", ccid="cc-old", outcome="fertig")
    placed = svc.place_call(to="+14155552671", brief="A new call.")
    assert placed.status == "dialing"
    assert placed.id != stuck.id
    assert any(call[0] == "POST" and str(call[1]).endswith("/calls") for call in telnyx._http.calls)


async def test_memory_timeout_still_leaves_the_call_completed(tmp_path, monkeypatch):
    settings, store, svc, _telnyx = _service(
        tmp_path,
        stella_post_call_memory=True,
        stella_post_call_memory_timeout_s=0.15,
    )
    job = _open(store, status="completing", outcome="Briefing fertig.")

    async def _hang(*_args, **_kwargs):
        await asyncio.sleep(30)

    monkeypatch.setattr(pcm, "run", _hang)
    svc.finalize_session(job.id)
    immediate = store.get(job.id)
    assert immediate.status == "completed"
    assert svc.busy()["busy"] is False
    await _drain()
    done = store.get(job.id)
    assert done.status == "completed"
    assert done.outcome == "Briefing fertig."
    assert done.memory_summary == f"Fehler: {MSG_TIMEOUT}"
    assert svc.busy() == {"busy": False, "count": 0, "calls": []}


async def test_reaper_closes_stale_completing_and_clears_busy(tmp_path, caplog):
    transport = LivenessTransport("alive")
    _settings, store, svc, _telnyx = _service(tmp_path, transport)
    job = _open(store, status="completing", ccid="cc-stale", outcome="Briefing fertig.")
    _backdate(store, job.id, 1000)
    with caplog.at_level(logging.WARNING, logger="stella.jobs"):
        result = await svc.reconcile()
    assert result["count"] == 1
    assert result["reset"][0]["id"] == job.id
    assert result["reset"][0]["status"] == "completed"
    done = store.get(job.id)
    assert done.status == "completed"
    assert "auto-recovered: stale completing" in done.outcome
    assert "Briefing fertig." in done.outcome
    assert any(url.endswith("/actions/hangup") for _method, url in transport.calls)
    events = store.events(job.id)
    assert events[-1]["event_type"] == "stella.auto_recovered"
    assert "stella_reaper auto-recovered" in caplog.text
    assert job.id in caplog.text
    assert svc.busy() == {"busy": False, "count": 0, "calls": []}
    # The row is still there.
    assert store.get(job.id) is not None


async def test_reaper_90015_completes_a_young_call_without_hangup(tmp_path, caplog):
    transport = LivenessTransport("ended_90015")
    _settings, store, svc, _telnyx = _service(tmp_path, transport)
    job = _open(store, status="in_progress", ccid="cc-90015", outcome="")
    with caplog.at_level(logging.WARNING, logger="stella.jobs"):
        result = await svc.reconcile()
    assert result["count"] == 1
    done = store.get(job.id)
    assert done.status == "completed"
    assert done.outcome == "auto-recovered: telnyx 90015 call already ended"
    assert not any(url.endswith("/actions/hangup") for _method, url in transport.calls)
    assert "90015" in caplog.text
    assert svc.busy()["busy"] is False


async def test_reaper_is_alive_false_completes(tmp_path):
    _settings, store, svc, _telnyx = _service(tmp_path, LivenessTransport("ended"))
    job = _open(store, status="in_progress", ccid="cc-dead")
    await svc.reconcile()
    done = store.get(job.id)
    assert done.status == "completed"
    assert done.outcome == "auto-recovered: telnyx call already ended"


async def test_reaper_leaves_a_live_young_call_and_fails_a_stale_dial(tmp_path):
    transport = LivenessTransport("alive")
    _settings, store, svc, _telnyx = _service(tmp_path, transport)
    live = _open(store, status="in_progress", ccid="cc-live")
    queued = _open(store, status="queued")
    _backdate(store, queued.id, 1000)
    await svc.reconcile()
    assert store.get(live.id).status == "in_progress"
    failed = store.get(queued.id)
    assert failed.status == "failed"
    assert failed.outcome == "auto-recovered: stale queued"
    assert svc.busy()["count"] == 1


async def test_reaper_closes_the_media_session(tmp_path):
    _settings, store, svc, _telnyx = _service(tmp_path)
    job = _open(store, status="completing", outcome="fertig")
    closed: list[str] = []

    async def _close() -> None:
        closed.append(job.id)

    call_sessions.register(job.id, _close)
    await svc.reset_stuck(job.id)
    assert closed == [job.id]
    assert store.get(job.id).status == "completed"
    assert "auto-recovered: manual reset" in store.get(job.id).outcome
    assert not call_sessions.has_session(job.id)


async def test_reset_stuck_unknown_and_already_terminal(tmp_path):
    _settings, store, svc, _telnyx = _service(tmp_path)
    done = _open(store, status="completed")
    quiet = await svc.reset_stuck(done.id)
    assert quiet["count"] == 0
    assert quiet["call"]["status"] == "completed"
    with pytest.raises(Exception) as ei:
        await svc.reset_stuck("missing-id")
    assert "missing-id" in str(ei.value)


def test_startup_reconcile_and_reset_endpoint(tmp_path, caplog):
    from fastapi.testclient import TestClient

    from stella.http_app import create_app

    settings = make_settings(
        tmp_path,
        stella_post_call_memory=False,
        stella_mcp_token="s3cret",
        stella_reaper_interval_s=3600,
    )
    store = JobStore(settings.stella_db_path)
    stale = _open(store, status="completing", ccid="", outcome="Briefing fertig.")
    _backdate(store, stale.id, 1000)
    young = _open(store, status="completing", outcome="noch dran")
    transport = LivenessTransport("alive")
    app = create_app(settings, telnyx=TelnyxClient(settings, transport))
    with caplog.at_level(logging.WARNING, logger="stella.jobs"):
        with TestClient(app) as client:
            assert client.post("/calls/reset-stuck").status_code == 401
            stale_body = client.get(
                f"/calls/{stale.id}",
                headers={"Authorization": "Bearer s3cret"},
            ).json()
            assert stale_body["status"] == "completed"
            assert "auto-recovered: stale completing" in stale_body["outcome"]
            assert "stella.auto_recovered" in {e["event_type"] for e in stale_body["events"]}
            young_body = client.get(
                f"/calls/{young.id}",
                headers={"Authorization": "Bearer s3cret"},
            ).json()
            assert young_body["status"] == "completing"
            busy = client.get("/status/busy", headers={"Authorization": "Bearer s3cret"}).json()
            assert busy["busy"] is False
            reset = client.post(
                "/calls/reset-stuck",
                json={"call_id": young.id},
                headers={"Authorization": "Bearer s3cret"},
            )
            assert reset.status_code == 200
            body = reset.json()
            assert body["count"] == 1
            assert body["reset"][0]["status"] == "completed"
            after = client.get(
                f"/calls/{young.id}",
                headers={"Authorization": "Bearer s3cret"},
            ).json()
            assert after["status"] == "completed"
            assert "auto-recovered: manual reset" in after["outcome"]
            missing = client.post(
                "/calls/reset-stuck",
                json={"call_id": "no-such"},
                headers={"Authorization": "Bearer s3cret"},
            )
            assert missing.status_code == 404
    assert "stella_reaper auto-recovered" in caplog.text


def test_media_websocket_close_sets_completed_without_webhook(tmp_path, monkeypatch):
    import threading
    import time

    from fastapi.testclient import TestClient

    from stella import http_app
    from stella.http_app import create_app

    release = threading.Event()

    async def _bridge(**kwargs):
        kwargs["outcome_cb"]("Briefing fertig.")
        # Hold the socket open until the client is attached, then return as a
        # media close with no Telnyx call.hangup webhook.
        await asyncio.to_thread(release.wait, 5)
        return "gemini"

    monkeypatch.setattr(http_app, "start_voice_bridge", _bridge)
    settings = make_settings(
        tmp_path,
        stella_post_call_memory=False,
        stella_reaper_interval_s=3600,
    )
    store = JobStore(settings.stella_db_path)
    job = _open(store, status="in_progress", ccid="cc-ws")
    transport = LivenessTransport("alive")
    app = create_app(settings, telnyx=TelnyxClient(settings, transport))
    with TestClient(app) as client:
        with client.websocket_connect(f"/media/{job.id}"):
            release.set()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and store.get(job.id).status != "completed":
                time.sleep(0.05)
        body = client.get(f"/calls/{job.id}").json()
    assert body["status"] == "completed"
    assert body["outcome"] == "Briefing fertig."
    assert not any(url.endswith("/actions/hangup") for _method, url in transport.calls)
    # The websocket path already finished the job; a late outcome must not reopen it.
    JobService(settings, store, TelnyxClient(settings, transport)).note_outcome(job.id, "spaeter")
    assert store.get(job.id).status == "completed"
    assert store.get(job.id).outcome == "Briefing fertig."
