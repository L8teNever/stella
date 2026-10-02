from __future__ import annotations

import argparse
import sys

import uvicorn

from stella.config import get_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stella", description="Stella voice/phone stack")
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="Run HTTP API, Telnyx webhooks, MCP HTTP, media WS")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)

    sub.add_parser("mcp", help="Run MCP over stdio (for Cursor mcp.json)")

    args = parser.parse_args(argv)
    settings = get_settings()

    if args.cmd == "serve":
        host = args.host or settings.stella_host
        port = args.port or settings.stella_port
        uvicorn.run("stella.http_app:create_app", host=host, port=port, factory=True)
        return 0

    if args.cmd == "mcp":
        from stella.mcp_stdio import main as mcp_main

        mcp_main()
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
