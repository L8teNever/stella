from __future__ import annotations

from stella.config import get_settings
from stella.jobs import JobService
from stella.mcp_app import build_mcp
from stella.store import JobStore
from stella.telnyx_client import TelnyxClient


def main() -> None:
    settings = get_settings()
    settings.data_dir()
    store = JobStore(settings.stella_db_path)
    jobs = JobService(settings, store, TelnyxClient(settings))
    mcp = build_mcp(jobs)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
