from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from stella.errors import StellaError
from stella.jobs import JobService


def build_mcp(service: JobService) -> FastMCP:
    mcp = FastMCP(
        "stella",
        instructions=(
            "Stella places outbound phone calls. Pass everything the agent needs "
            "in brief/context — Stella has no access to Ida memory or other MCPs."
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
    ) -> dict[str, Any]:
        """Place an outbound call. `to` must be E.164. Stella only knows `brief` + `context`."""
        try:
            job = service.place_call(
                to=to, brief=brief, context=context, speak_to=speak_to, kind="call"
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
    def stella_briefing_call(to: str, text: str, speak_to: str = "") -> dict[str, Any]:
        """Call a number and read `text` aloud (e.g. a morning briefing), then hang up."""
        try:
            brief = (
                "Read the following briefing aloud clearly, then confirm they heard it, then hang up.\n\n"
                + text
            )
            job = service.place_call(
                to=to,
                brief=brief,
                context=text,
                speak_to=speak_to,
                kind="briefing",
            )
            return job.to_public_dict()
        except StellaError as exc:
            return exc.to_dict()

    return mcp
