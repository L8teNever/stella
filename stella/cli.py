from __future__ import annotations

import argparse
import json
import sys

import uvicorn

from stella.config import get_settings
from stella.xai_auth import XAIAuth


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stella", description="Stella voice/phone stack")
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="Run HTTP API, Telnyx webhooks, MCP HTTP, media WS")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)

    oauth = sub.add_parser("oauth", help="SuperGrok / xAI device-code login")
    oauth_sub = oauth.add_subparsers(dest="oauth_cmd", required=True)
    oauth_sub.add_parser("login", help="Start device login and poll until approved")
    oauth_sub.add_parser("status", help="Show whether OAuth tokens or XAI_API_KEY are available")

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

    if args.cmd == "oauth":
        auth = XAIAuth(settings)
        if args.oauth_cmd == "status":
            print(json.dumps({"mode": auth.auth_mode()}))
            return 0
        login = auth.start_device_login()
        print("Approve Stella on this xAI/Grok account:")
        print(f"  Visit: {login.verification_uri_complete or login.verification_uri}")
        print(f"  Code:  {login.user_code}")
        auth.poll_device_login(login)
        print("Saved OAuth tokens. Stella will use SuperGrok subscription quota.")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
