"""μ-law (PCMU 8 kHz) <-> linear PCM helpers for Telnyx <-> Gemini Live."""

from __future__ import annotations

import audioop
import base64
import re


_RATE_RE = re.compile(r"rate=(\d+)", re.IGNORECASE)


def parse_pcm_rate(mime: str | None, default: int) -> int:
    if not mime:
        return default
    match = _RATE_RE.search(mime)
    if not match:
        return default
    return int(match.group(1))


class Pcmu8kToPcm16k:
    """Telnyx media payload (base64 PCMU 8 kHz) -> 16-bit PCM 16 kHz mono."""

    def __init__(self) -> None:
        self._state: tuple | None = None

    def convert_b64(self, pcmu_b64: str) -> bytes:
        ulaw = base64.b64decode(pcmu_b64)
        if not ulaw:
            return b""
        pcm8 = audioop.ulaw2lin(ulaw, 2)
        pcm16, self._state = audioop.ratecv(pcm8, 2, 1, 8000, 16000, self._state)
        return pcm16


class PcmToPcmu8k:
    """Gemini Live PCM (typically 24 kHz 16-bit LE) -> base64 PCMU 8 kHz for Telnyx."""

    def __init__(self, default_rate: int = 24000) -> None:
        self._default_rate = default_rate
        self._state: tuple | None = None
        self._rate = default_rate

    def convert_b64(self, pcm_b64: str, mime: str | None = None) -> str:
        rate = parse_pcm_rate(mime, self._default_rate)
        if rate != self._rate:
            self._rate = rate
            self._state = None
        pcm = base64.b64decode(pcm_b64)
        if not pcm:
            return ""
        if rate != 8000:
            pcm, self._state = audioop.ratecv(pcm, 2, 1, rate, 8000, self._state)
        ulaw = audioop.lin2ulaw(pcm, 2)
        return base64.b64encode(ulaw).decode("ascii")
