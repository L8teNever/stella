from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from stella.errors import StellaError
from stella.jobs import JobService, parse_run_at


def build_mcp(service: JobService) -> FastMCP:
    mcp = FastMCP(
        "stella",
        instructions=(
            "Stella places outbound phone calls. Pass everything the agent needs "
            "in brief/context. Stella has no direct access to Ida memory or other MCPs; only calls"
            " to the owner number may use the live `frag_ida` lookup."
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

        `allow_ida`: offer the live `frag_ida` lookup tool. Only ever honoured when `to`
        is the configured owner number; leave unset for the default (on for the owner).
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

    @mcp.tool()
    def stella_schedule_call(
        to: str,
        brief: str,
        at: str = "",
        in_minutes: int | None = None,
        context: str = "",
        speak_to: str = "",
        allow_ida: bool | None = None,
    ) -> dict[str, Any]:
        """Schedule a call. `at` is HH:MM (24h, Europe/Berlin, today) or an ISO datetime;
        alternatively `in_minutes`. Runs automatically at that time (up to 14 days ahead)."""
        try:
            run_at = parse_run_at(uhrzeit=at, in_minuten=in_minutes)
            return service.schedule_call(
                to=to, brief=brief, run_at=run_at, context=context, speak_to=speak_to,
                allow_ida=allow_ida,
            )
        except StellaError as exc:
            return exc.to_dict()

    @mcp.tool()
    def stella_scheduled_calls() -> dict[str, Any]:
        """List pending scheduled calls."""
        return {"scheduled": service.store.list_scheduled("pending")}

    @mcp.tool()
    def stella_cancel_scheduled_call(schedule_id: str) -> dict[str, Any]:
        """Cancel a pending scheduled call by its id."""
        return {"cancelled": service.store.cancel_scheduled(schedule_id)}

    return mcp
