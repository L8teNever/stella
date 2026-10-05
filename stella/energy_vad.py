"""Local energy / RMS VAD for Telnyx inbound PCM (after 8 kHz μ-law → 16 kHz PCM).

Used to send Gemini Live ``activityStart`` / ``activityEnd`` faster than the
server-side end-of-speech detector.

Steady cabin/engine noise is rejected with an adaptive noise floor (amplitude
hysteresis vs that floor). Silence hang time is unchanged: end-of-speech is
still ``silence_ms`` of frames below the end threshold — no extra padding.
"""

from __future__ import annotations

import audioop
import math
from collections import deque
from dataclasses import dataclass

from stella.audio_pcmu import SAMPLE_WIDTH, TELNYX_RATE


@dataclass
class EnergyVadConfig:
    rms_threshold: int = 500
    min_speech_ms: int = 60
    silence_ms: int = 120
    frame_ms: int = 20
    # Adaptive floor so continuous rumble above rms_threshold is not "speech".
    noise_adapt: bool = True
    start_over_noise: float = 2.0
    end_over_noise: float = 1.35
    # RMS at/above this is treated as speech even if unmodulated (vowels / tests).
    # Engine-like levels sit below this and need modulation + margin over noise.
    clear_rms: int = 3500
    onset_spread: float = 0.3
    onset_frames: int = 5
    noise_alpha_up: float = 0.1
    noise_alpha_down: float = 0.25


class EnergyVad:
    def __init__(self, config: EnergyVadConfig | None = None) -> None:
        self.config = config or EnergyVadConfig()
        self.speaking = False
        self._speech_ms = 0
        self._silence_ms = 0
        self._leftover = b""
        self._noise = 0.0
        n = max(1, int(self.config.onset_frames))
        self._recent_rms: deque[int] = deque(maxlen=n)
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
        self._noise = 0.0
        self._recent_rms.clear()

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

    def _spread(self) -> float:
        if len(self._recent_rms) < 2:
            return 0.0
        mean = sum(self._recent_rms) / len(self._recent_rms)
        peak = max(self._recent_rms)
        low = min(self._recent_rms)
        denom = max(mean, 1.0)
        return (peak - low) / denom

    def _start_threshold(self) -> int:
        cfg = self.config
        if not cfg.noise_adapt:
            return cfg.rms_threshold
        return max(cfg.rms_threshold, int(self._noise * cfg.start_over_noise))

    def _end_threshold(self) -> int:
        cfg = self.config
        if not cfg.noise_adapt:
            return cfg.rms_threshold
        return max(cfg.rms_threshold, int(self._noise * cfg.end_over_noise))

    def _onset_loud(self, rms: int) -> bool:
        cfg = self.config
        if rms < self._start_threshold():
            return False
        if not cfg.noise_adapt:
            return True
        if rms >= cfg.clear_rms:
            return True
        return self._spread() >= cfg.onset_spread

    def _update_noise(self, rms: int, candidate_speech: bool) -> None:
        cfg = self.config
        if not cfg.noise_adapt or candidate_speech:
            return
        n = self._noise
        if rms < n:
            self._noise = n + cfg.noise_alpha_down * (rms - n)
        else:
            self._noise = n + cfg.noise_alpha_up * (rms - n)

    def _step(self, rms: int) -> list[str]:
        cfg = self.config
        self._recent_rms.append(int(rms))
        if self.speaking:
            loud = rms >= self._end_threshold()
            # Unmodulated mid-level energy (engine lock) may raise the floor so
            # end-threshold can release. Real speech is >= clear_rms and is not trained.
            train_noise = rms < cfg.clear_rms and self._spread() < cfg.onset_spread
        else:
            loud = self._onset_loud(rms)
            train_noise = not loud
        self._update_noise(rms, candidate_speech=not train_noise)
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


def energy_vad_config_from_settings(settings) -> EnergyVadConfig:
    return EnergyVadConfig(
        rms_threshold=int(settings.stella_client_vad_rms),
        min_speech_ms=int(settings.stella_client_vad_min_speech_ms),
        silence_ms=int(settings.stella_client_vad_silence_ms),
        noise_adapt=bool(settings.stella_client_vad_noise_adapt),
        start_over_noise=float(settings.stella_client_vad_start_over_noise),
        end_over_noise=float(settings.stella_client_vad_end_over_noise),
        clear_rms=int(settings.stella_client_vad_clear_rms),
        onset_spread=float(settings.stella_client_vad_onset_spread),
    )
