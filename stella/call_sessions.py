"""In-process media sessions the reaper can close.

The registry only holds a closer for the live WebSocket. It does not own the
SQLite job row and never deletes it.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

Closer = Callable[[], Awaitable[None]]
_closers: dict[str, Closer] = {}


def register(job_id: str, close: Closer) -> None:
    _closers[job_id] = close


def unregister(job_id: str) -> None:
    _closers.pop(job_id, None)


def has_session(job_id: str) -> bool:
    return job_id in _closers


async def close_session(job_id: str) -> bool:
    """Close the live media socket, if this process still has one."""
    close = _closers.pop(job_id, None)
    if close is None:
        return False
    try:
        await close()
    except Exception:
        logger.warning("session close failed for %s", job_id, exc_info=True)
        return False
    return True


def clear() -> None:
    """Drop registrations. Used by tests so closers do not leak across cases."""
    _closers.clear()
