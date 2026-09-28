"""Local energy / RMS VAD for Telnyx inbound PCM (after 8 kHz μ-law → 16 kHz PCM).

Used to send Gemini Live ``activityStart`` / ``activityEnd`` faster than the
server-side end-of-speech detector.
"""

from __future__ import annotations

import audioop
import math
from dataclasses import dataclass

from stella.audio_pcmu import SAMPLE_WIDTH, TELNYX_RATE


@dataclass
class EnergyVadConfig:
    rms_threshold: int = 500
    min_speech_ms: int = 60
    silence_ms: int = 120
    frame_ms: int = 20


class EnergyVad:
    def __init__(self, config: EnergyVadConfig | None = None) -> None:
        self.config = config or EnergyVadConfig()
        self.speaking = False
        self._speech_ms = 0
        self._silence_ms = 0
        self._leftover = b""
        self.bytes_per_frame = (
            (TELNYX_RATE * 2)  # 16 kHz after upsample
            * SAMPLE_WIDTH
            * self.config.frame_ms
            // 1000
        )

    def reset(self) -> None:
        self.speaking = False
        self._speech_ms = 0
        self._silence_ms = 0
        self._leftover = b""

    def feed(self, pcm16le: bytes) -> list[str]:
        """Return activity events: ``start`` and/or ``end`` in order."""
        events: list[str] = []
        buf = self._leftover + pcm16le
        step = self.bytes_per_frame
        if step <= 0:
            return events
        offset = 0
        while offset + step <= len(buf):
            frame = buf[offset : offset + step]
            offset += step
            rms = audioop.rms(frame, SAMPLE_WIDTH) if frame else 0
            events.extend(self._step(rms))
        self._leftover = buf[offset:]
        return events

    def _step(self, rms: int) -> list[str]:
        cfg = self.config
        loud = rms >= cfg.rms_threshold
        out: list[str] = []
        if loud:
            self._speech_ms += cfg.frame_ms
            self._silence_ms = 0
            if not self.speaking and self._speech_ms >= cfg.min_speech_ms:
                self.speaking = True
                out.append("start")
        else:
            if self.speaking:
                self._silence_ms += cfg.frame_ms
                if self._silence_ms >= cfg.silence_ms:
                    self.speaking = False
                    self._speech_ms = 0
                    self._silence_ms = 0
                    out.append("end")
            else:
                self._speech_ms = 0
                self._silence_ms = 0
        return out


def pcm_rms(pcm16le: bytes) -> int:
    if not pcm16le:
        return 0
    return int(audioop.rms(pcm16le, SAMPLE_WIDTH))


def db_from_rms(rms: int) -> float:
    if rms <= 0:
        return -math.inf
    return 20.0 * math.log10(rms / 32768.0)
