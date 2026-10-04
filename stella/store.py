from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CallJob:
    id: str
    kind: str  # call | briefing | inbound
    to_number: str
    brief: str
    context: str
    speak_to: str
    status: str
    telnyx_call_control_id: str
    telnyx_call_leg_id: str
    outcome: str
    transcript: str
    error: str
    voice_provider: str
    allow_ida: bool
    media_codec: str
    created_at: str
    updated_at: str

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "to": self.to_number,
            "brief": self.brief,
            "context": self.context,
            "speak_to": self.speak_to,
            "status": self.status,
            "telnyx_call_control_id": self.telnyx_call_control_id or None,
            "outcome": self.outcome or None,
            "transcript": self.transcript or None,
            "error": self.error or None,
            "voice_provider": self.voice_provider or None,
            "allow_ida": self.allow_ida,
            "media_codec": self.media_codec or None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class JobStore:
    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._path = db_path
        self._lock = threading.Lock()
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
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
            try:
                conn.execute(
                    "ALTER TABLE jobs ADD COLUMN voice_provider TEXT NOT NULL DEFAULT ''"
                )
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE jobs ADD COLUMN allow_ida INTEGER NOT NULL DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE jobs ADD COLUMN media_codec TEXT NOT NULL DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def create(
        self,
        *,
        kind: str,
        to_number: str,
        brief: str,
        context: str = "",
        speak_to: str = "",
        allow_ida: bool = False,
    ) -> CallJob:
        job = CallJob(
            id=str(uuid.uuid4()),
            kind=kind,
            to_number=to_number,
            brief=brief,
            context=context or "",
            speak_to=speak_to or "",
            status="queued",
            telnyx_call_control_id="",
            telnyx_call_leg_id="",
            outcome="",
            transcript="",
            error="",
            voice_provider="",
            allow_ida=bool(allow_ida),
            media_codec="",
            created_at=_now(),
            updated_at=_now(),
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    id, kind, to_number, brief, context, speak_to, status,
                    telnyx_call_control_id, telnyx_call_leg_id, outcome,
                    transcript, error, voice_provider, allow_ida, media_codec,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    job.kind,
                    job.to_number,
                    job.brief,
                    job.context,
                    job.speak_to,
                    job.status,
                    job.telnyx_call_control_id,
                    job.telnyx_call_leg_id,
                    job.outcome,
                    job.transcript,
                    job.error,
                    job.voice_provider,
                    int(job.allow_ida),
                    job.media_codec,
                    job.created_at,
                    job.updated_at,
                ),
            )
            conn.commit()
        return job

    def get(self, job_id: str) -> CallJob | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def list_recent(self, limit: int = 10, kind: str | None = None) -> list[CallJob]:
        limit = max(1, min(int(limit), 50))
        sql = "SELECT * FROM jobs"
        args: list[Any] = []
        if kind:
            sql += " WHERE kind = ?"
            args.append(kind)
        sql += " ORDER BY datetime(created_at) DESC, created_at DESC LIMIT ?"
        args.append(limit)
        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [self._row_to_job(row) for row in rows]

    def get_by_call_control_id(self, call_control_id: str) -> CallJob | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE telnyx_call_control_id = ?",
                (call_control_id,),
            ).fetchone()
        return self._row_to_job(row) if row else None

    def update(self, job_id: str, **fields: Any) -> CallJob | None:
        if not fields:
            return self.get(job_id)
        fields["updated_at"] = _now()
        cols = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [job_id]
        with self._lock, self._connect() as conn:
            conn.execute(f"UPDATE jobs SET {cols} WHERE id = ?", values)
            conn.commit()
        return self.get(job_id)

    def append_transcript(self, job_id: str, chunk: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET transcript = transcript || ?, updated_at = ? WHERE id = ?",
                (chunk, _now(), job_id),
            )
            conn.commit()

    def add_event(self, job_id: str, event_type: str, payload: dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO events (job_id, event_type, payload, created_at) VALUES (?, ?, ?, ?)",
                (job_id, event_type, json.dumps(payload), _now()),
            )
            conn.commit()

    def events(self, job_id: str) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT event_type, payload, created_at FROM events WHERE job_id = ? ORDER BY id",
                (job_id,),
            ).fetchall()
        out = []
        for row in rows:
            out.append(
                {
                    "event_type": row["event_type"],
                    "payload": json.loads(row["payload"]),
                    "created_at": row["created_at"],
                }
            )
        return out

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> CallJob:
        return CallJob(
            id=row["id"],
            kind=row["kind"],
            to_number=row["to_number"],
            brief=row["brief"],
            context=row["context"],
            speak_to=row["speak_to"],
            status=row["status"],
            telnyx_call_control_id=row["telnyx_call_control_id"],
            telnyx_call_leg_id=row["telnyx_call_leg_id"],
            outcome=row["outcome"],
            transcript=row["transcript"],
            error=row["error"],
            voice_provider=row["voice_provider"] if "voice_provider" in row.keys() else "",
            allow_ida=bool(row["allow_ida"]) if "allow_ida" in row.keys() else False,
            media_codec=row["media_codec"] if "media_codec" in row.keys() else "",
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
