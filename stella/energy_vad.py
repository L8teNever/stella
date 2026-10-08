"""Local energy VAD for Telnyx inbound PCM (8 kHz G.711 → 16 kHz PCM).

Sends Gemini Live ``activityStart`` / ``activityEnd``. Cabin rumble is
rejected with (1) high-pass energy so low-frequency motor noise is not
"speech", (2) an adaptive floor trained on stationary frames, (3) amplitude
hysteresis vs that floor. Hang times are unchanged: ``min_speech_ms`` to
start, ``silence_ms`` of non-speech-like frames to end — no extra padding.

Loud unmodulated energy is never treated as speech just because it is loud
(``clear_rms`` is not a bypass).
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
    noise_adapt: bool = True
    start_over_noise: float = 1.8
    end_over_noise: float = 1.15
    onset_spread: float = 0.25
    onset_frames: int = 5
    # First-difference RMS; rumble has a low hp/full ratio.
    highpass: bool = True
    rumble_hp_ratio: float = 0.10
    noise_alpha_up: float = 0.25
    noise_alpha_down: float = 0.4
    # Kept for env compatibility; not a speech bypass (that locked VAD in cars).
    clear_rms: int = 0


class EnergyVad:
    def __init__(self, config: EnergyVadConfig | None = None) -> None:
        self.config = config or EnergyVadConfig()
        self.speaking = False
        self._speech_ms = 0
        self._silence_ms = 0
        self._leftover = b""
        self._noise = 0.0
        self._hp_prev = 0
        n = max(1, int(self.config.onset_frames))
        self._recent: deque[int] = deque(maxlen=n)
        self.bytes_per_frame = (
            (TELNYX_RATE * 2)
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
        self._hp_prev = 0
        self._recent.clear()

    def feed(self, pcm16le: bytes) -> list[str]:
        events: list[str] = []
        buf = self._leftover + pcm16le
        step = self.bytes_per_frame
        if step <= 0:
            return events
        offset = 0
        while offset + step <= len(buf):
            frame = buf[offset : offset + step]
            offset += step
            full, hp = self._frame_levels(frame)
            events.extend(self._step(full, hp))
        self._leftover = buf[offset:]
        return events

    def _frame_levels(self, frame: bytes) -> tuple[int, int]:
        n = len(frame) // SAMPLE_WIDTH
        if n <= 0:
            return 0, 0
        full = audioop.rms(frame, SAMPLE_WIDTH) if frame else 0
        prev = self._hp_prev
        acc = 0
        for i in range(n):
            s = int.from_bytes(frame[2 * i : 2 * i + 2], "little", signed=True)
            d = s - prev
            acc += d * d
            prev = s
        self._hp_prev = prev
        hp = int(math.sqrt(acc / n)) if n else 0
        return int(full), hp

    def _spread(self) -> float:
        if len(self._recent) < 2:
            return 0.0
        mean = sum(self._recent) / len(self._recent)
        return (max(self._recent) - min(self._recent)) / max(mean, 1.0)

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

    def _update_noise(self, feat: int, fast: bool) -> None:
        cfg = self.config
        if not cfg.noise_adapt:
            return
        n = self._noise
        if feat < n:
            self._noise = n + cfg.noise_alpha_down * (feat - n)
            return
        alpha = cfg.noise_alpha_up if fast else cfg.noise_alpha_up * 0.08
        self._noise = n + alpha * (feat - n)

    def _step(self, full_rms: int, hp_rms: int) -> list[str]:
        cfg = self.config
        feat = hp_rms if cfg.highpass else full_rms
        hp_ratio = hp_rms / max(full_rms, 1)
        self._recent.append(int(feat))
        # Too few frames: do not treat as stationary cabin (would eat quiet onsets).
        modulated = len(self._recent) < 2 or self._spread() >= cfg.onset_spread
        start_thr = self._start_threshold()
        # LF-dominated fullband energy is motor/road, even when HP RMS exceeds 500.
        rumble = (
            bool(cfg.highpass)
            and full_rms >= cfg.rms_threshold
            and hp_ratio < cfg.rumble_hp_ratio
        )

        if cfg.noise_adapt:
            # Stationary / rumble frames *are* the cabin floor — train even if loud.
            # Do not train on frames that already look like speech (keeps quiet onsets).
            fast = rumble or not modulated
            if rumble or (fast and feat < start_thr) or (not self.speaking and feat < start_thr):
                self._update_noise(feat, fast=fast)

        start_thr = self._start_threshold()
        end_thr = self._end_threshold()
        if rumble:
            loud = False
        elif not cfg.noise_adapt:
            loud = feat >= cfg.rms_threshold
        elif not self.speaking:
            loud = feat >= start_thr
        else:
            # Vowels in the quiet: unmodulated but well above the floor.
            # Cabin after a turn: near the floor and/or unmodulated → not speech.
            loud = feat >= end_thr and (modulated or feat >= start_thr)

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
        onset_spread=float(settings.stella_client_vad_onset_spread),
        highpass=bool(settings.stella_client_vad_highpass),
        rumble_hp_ratio=float(settings.stella_client_vad_rumble_hp_ratio),
        noise_alpha_up=float(settings.stella_client_vad_noise_alpha_up),
        clear_rms=int(getattr(settings, "stella_client_vad_clear_rms", 0) or 0),
    )
