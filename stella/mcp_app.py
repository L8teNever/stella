from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from stella.errors import StellaError
from stella.jobs import JobService


def build_mcp(service: JobService) -> FastMCP:
    mcp = FastMCP(
        "stella",
        instructions=(
            "Stella places outbound phone calls and automatically answers inbound "
            "PSTN to TELNYX_FROM_NUMBER (owner by default). Use stella_call for "
            "outbound; inbound needs no MCP. Pass everything the agent needs in "
            "brief/context. Stella has no direct access to Ida memory or other MCPs; only calls"
            " with the owner on the line may use the live `frag_ida` lookup."
        ),
        stateless_http=True,
    )
    mcp.settings.streamable_http_path = "/"
    mcp.settings.transport_security.enable_dns_rebinding_protection = False

    @mcp.tool()
    def stella_call(
        to: str,
        brief: str,
        context: str = "",
        speak_to: str = "",
        allow_ida: bool | None = None,
    ) -> dict[str, Any]:
        """Place an outbound call. `to` must be E.164. Stella only knows `brief` + `context`.

        `allow_ida`: hand the call access to Ida's tools (live `frag_ida` look-ups/actions and
        `aufgabe_planen`), so Simon can give follow-up tasks during the call. Only ever honoured
        when `to` is the configured owner number; unset means on for the owner. Pass true for
        reminder and callback calls.
        """
        try:
            job = service.place_call(
                to=to,
                brief=brief,
                context=context,
                speak_to=speak_to,
                kind="call",
                allow_ida=allow_ida,
            )
            return job.to_public_dict()
        except StellaError as exc:
            return exc.to_dict()

    @mcp.tool()
    def stella_call_status(call_id: str) -> dict[str, Any]:
        """Look up a Stella call job by id returned from stella_call / stella_briefing_call."""
        try:
            job = service.status(call_id)
            data = job.to_public_dict()
            data["events"] = service.store.events(job.id)
            return data
        except StellaError as exc:
            return exc.to_dict()

    @mcp.tool()
    def stella_recent_calls(limit: int = 10, kind: str = "") -> dict[str, Any]:
        """List recent Stella jobs (outbound and inbound). Inbound is automatic when someone dials Stella.

        `kind` optional: call | briefing | inbound. `stella_call_status` still returns one job + events.
        """
        try:
            limit_n = int(limit)
        except (TypeError, ValueError):
            limit_n = 10
        jobs = service.store.list_recent(limit=limit_n, kind=(kind or "").strip() or None)
        return {"calls": [job.to_public_dict() for job in jobs]}

    @mcp.tool()
    def stella_briefing_call(
        to: str, text: str, speak_to: str = "", allow_ida: bool | None = None
    ) -> dict[str, Any]:
        """Call a number and read `text` aloud (e.g. a morning briefing), then hang up."""
        try:
            brief = (
                "Read the following briefing aloud clearly, confirm they heard it, "
                "then call hang_up (do not keep the line open).\n\n"
                + text
            )
            job = service.place_call(
                to=to,
                brief=brief,
                context=text,
                speak_to=speak_to,
                kind="briefing",
                allow_ida=allow_ida,
            )
            return job.to_public_dict()
        except StellaError as exc:
            return exc.to_dict()

    return mcp
