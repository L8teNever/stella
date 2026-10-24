from __future__ import annotations

import asyncio
import json
import logging
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from stella import __version__, call_sessions, callback_bg, master_prompt, post_call_memory
from stella.config import Settings, get_settings
from stella.errors import StellaError
from stella.jobs import BERLIN, JobService
from stella.mcp_app import build_mcp
from stella.store import JobStore
from stella.telnyx_client import TelnyxClient
from stella.voice import start_voice_bridge
from stella.webhooks import verify_telnyx_signature

logger = logging.getLogger("stella")


def _token_ok(request: Request, expected: str) -> bool:
    if not expected:
        return True
    auth = request.headers.get("authorization") or ""
    bearer = f"Bearer {expected}"
    q = request.query_params.get("Token") or request.query_params.get("token") or ""
    try:
        return secrets.compare_digest(auth, bearer) or secrets.compare_digest(q, expected)
    except ValueError:
        return False


def create_app(settings: Settings | None = None, telnyx: TelnyxClient | None = None) -> FastAPI:
    settings = settings or get_settings()
    settings.data_dir()
    store = JobStore(settings.stella_db_path)
    telnyx = telnyx or TelnyxClient(settings)
    jobs = JobService(settings, store, telnyx)
    mcp = build_mcp(jobs)
    mcp_asgi = mcp.streamable_http_app()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(level=logging.INFO)

        async def master_prompt_loop() -> None:
            while True:
                await master_prompt.refresh(settings)
                await asyncio.sleep(max(10.0, settings.master_prompt_refresh_s))

        async def reaper_loop() -> None:
            interval = max(5.0, float(settings.stella_reaper_interval_s or 60.0))
            while True:
                await asyncio.sleep(interval)
                try:
                    await jobs.reconcile()
                except Exception:
                    logger.exception("stella_reaper pass failed")

        mp_task = asyncio.create_task(master_prompt_loop())
        try:
            await jobs.reconcile()
        except Exception:
            logger.exception("stella_reaper startup reconcile failed")
        reaper_task = asyncio.create_task(reaper_loop())
        try:
            async with mcp.session_manager.run():
                yield
        finally:
            reaper_task.cancel()
            mp_task.cancel()

    app = FastAPI(title="Stella", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.store = store
    app.state.jobs = jobs
    app.state.mcp = mcp

    @app.get("/health")
    def health() -> dict:
        return {
            "ok": True,
            "service": "stella",
            "voice_provider": {
                "primary": "gemini",
                "gemini": "api_key" if (settings.gemini_api_key or "").strip() else "none",
            },
            "gemini_configured": bool((settings.gemini_api_key or "").strip()),
            "telnyx_configured": bool(
                settings.telnyx_api_key
                and settings.telnyx_connection_id
                and settings.telnyx_from_number
            ),
        }

    @app.post("/webhooks/telnyx")
    async def telnyx_webhook(
        request: Request,
        telnyx_signature_ed25519: str | None = Header(default=None),
        telnyx_timestamp: str | None = Header(default=None),
    ):
        raw = await request.body()
        if not settings.stella_skip_webhook_verify:
            if not settings.telnyx_public_key:
                raise HTTPException(
                    status_code=500,
                    detail="TELNYX_PUBLIC_KEY is not set; cannot verify webhooks. "
                    "Set STELLA_SKIP_WEBHOOK_VERIFY=true only for local tests.",
                )
            try:
                verify_telnyx_signature(
                    payload=raw,
                    timestamp=telnyx_timestamp or "",
                    signature_b64=telnyx_signature_ed25519 or "",
                    public_key_b64=settings.telnyx_public_key,
                )
            except StellaError as exc:
                raise HTTPException(status_code=401, detail=exc.message) from exc
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON")
        jobs.handle_telnyx_event(body)
        return {"received": True}

    @app.get("/status/busy")
    def status_busy():
        """Same snapshot as MCP `stella_busy`. Read-only; does not dial or wait."""
        return jobs.busy()

    @app.get("/calls/{call_id}")
    def get_call(call_id: str):
        try:
            job = jobs.status(call_id)
        except StellaError as exc:
            raise HTTPException(status_code=404, detail=exc.message) from exc
        data = job.to_public_dict()
        data["events"] = store.events(job.id)
        return data

    @app.post("/calls/reset-stuck")
    async def reset_stuck(request: Request):
        """Finish non-terminal calls now. Same token as `/calls` and `/mcp`."""
        call_id = ""
        raw = await request.body()
        if raw:
            try:
                body = json.loads(raw)
            except Exception as exc:
                raise HTTPException(status_code=400, detail="Invalid JSON") from exc
            if isinstance(body, dict):
                call_id = str(body.get("call_id") or "")
        try:
            return await jobs.reset_stuck(call_id.strip() or None)
        except StellaError as exc:
            status = 404 if exc.code == "not_found" else 400
            raise HTTPException(status_code=status, detail=exc.message) from exc

    @app.websocket("/media/{job_id}")
    async def media(websocket: WebSocket, job_id: str):
        await websocket.accept()
        job = store.get(job_id)
        if not job:
            await websocket.close(code=1008)
            return

        def transcript_cb(chunk: str) -> None:
            store.append_transcript(job_id, chunk)

        def outcome_cb(text: str) -> None:
            # `completing` only while the leg is still winding down. A terminal
            # status (our hangup, media close, webhook, reaper) is never demoted.
            jobs.note_outcome(job_id, text)

        async def hangup_cb() -> None:
            current = store.get(job_id)
            if current:
                # Successful hangup — including Telnyx 90015 — completes the job
                # here. call.hangup is not required.
                jobs.hangup_own_leg(current)

        async def close_media() -> None:
            try:
                await websocket.close(code=1000)
            except Exception:
                pass

        async def ws_connect(url: str, headers: dict):
            import websockets

            return await websockets.connect(url, additional_headers=headers or None)

        def schedule_cb(args: dict) -> dict:
            # Validate now (so a past time is corrected in the call), book in the background.
            plan = jobs.plan_task(job, args)
            callback_bg.start(settings, plan)
            run_at = plan["run_at"].astimezone(BERLIN)
            return {"result": f"Eingeplant für {run_at:%H:%M} Uhr."}

        failed = False
        err = ""
        call_sessions.register(job_id, close_media)
        try:
            post_call_memory.note_media_open(job_id)
            try:
                await start_voice_bridge(
                    settings=settings,
                    job=job,
                    telnyx_ws=websocket,
                    ws_connect=ws_connect,
                    hangup_cb=hangup_cb,
                    transcript_cb=transcript_cb,
                    outcome_cb=outcome_cb,
                    schedule_cb=schedule_cb,
                )
            except WebSocketDisconnect:
                pass
            except Exception as exc:
                failed = True
                err = str(exc)[:2000] or "voice session failed"
                logger.exception("media session failed for %s", job_id)
        finally:
            # Status first, then the detached memory hook. Every exit — hangup,
            # remote close, exception, Gemini failure — lands in a terminal status
            # even when the Telnyx call.hangup webhook never arrives.
            post_call_memory.note_media_closed(job_id)
            call_sessions.unregister(job_id)
            try:
                jobs.finalize_session(job_id, failed=failed, error=err)
            except Exception:
                logger.exception("finalize_session failed for %s", job_id)
                try:
                    current = store.get(job_id)
                    if current and current.status not in {"completed", "failed", "rejected"}:
                        store.update(
                            job_id,
                            status="failed",
                            error=err or "voice session failed",
                        )
                except Exception:
                    logger.exception("terminal status fallback failed for %s", job_id)

    @app.middleware("http")
    async def normalize_mcp_case(request: Request, call_next):
        path = request.scope.get("path") or ""
        if path.startswith("/MCP"):
            request.scope["path"] = "/mcp" + path[4:]
        return await call_next(request)

    @app.middleware("http")
    async def admin_auth_middleware(request: Request, call_next):
        path = (request.scope.get("path") or "").lower()
        needs = (
            path.startswith("/mcp")
            or path.startswith("/calls")
            or path.startswith("/status")
        )
        if needs and settings.stella_mcp_token and not _token_ok(request, settings.stella_mcp_token):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)

    app.mount("/mcp", mcp_asgi)
    return app
