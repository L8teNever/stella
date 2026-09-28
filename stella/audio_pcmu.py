"""μ-law (PCMU 8 kHz) <-> linear PCM helpers for Telnyx <-> Gemini Live.

Telnyx bidirectional RTP expects PCMU frames whose payload length is a
multiple of 160 bytes (20 ms at 8 kHz). Gemini Live uses 16-bit PCM
little-endian (typically 16 kHz in, 24 kHz out).

Downsampling 24 kHz -> 8 kHz with audioop.ratecv (linear interpolation)
aliases badly and often emits a length that is not 20 ms-aligned. Sending
that blob as one Telnyx media event mis-frames RTP and sounds swapped,
choppy, or distorted. Integer-ratio averaging + 20 ms framing fixes that.
"""

from __future__ import annotations

import audioop
import base64
import re
import struct

TELNYX_RATE = 8000
TELNYX_FRAME_SAMPLES = 160  # 20 ms
TELNYX_FRAME_BYTES = TELNYX_FRAME_SAMPLES  # μ-law is 1 byte/sample
SAMPLE_WIDTH = 2
GEMINI_IN_RATE = 16000
GEMINI_OUT_RATE = 24000

_RATE_RE = re.compile(r"rate=(\d+)", re.IGNORECASE)
_INT16_MIN = -32768
_INT16_MAX = 32767


def parse_pcm_rate(mime: str | None, default: int) -> int:
    if not mime:
        return default
    match = _RATE_RE.search(mime)
    if not match:
        return default
    rate = int(match.group(1))
    return rate if rate > 0 else default


def _clamp_i16(value: int) -> int:
    if value < _INT16_MIN:
        return _INT16_MIN
    if value > _INT16_MAX:
        return _INT16_MAX
    return value


def unpack_pcm16le(pcm: bytes) -> list[int]:
    n = len(pcm) // SAMPLE_WIDTH
    if n == 0:
        return []
    return list(struct.unpack_from("<" + "h" * n, pcm, 0))


def pack_pcm16le(samples: list[int]) -> bytes:
    if not samples:
        return b""
    return struct.pack("<" + "h" * len(samples), *(_clamp_i16(s) for s in samples))


def downsample_integer(samples: list[int], factor: int) -> tuple[list[int], list[int]]:
    """Box-filter downsample (anti-alias) by an integer factor."""
    if factor < 1:
        raise ValueError("factor must be >= 1")
    if factor == 1:
        return list(samples), []
    n = (len(samples) // factor) * factor
    out: list[int] = []
    for i in range(0, n, factor):
        out.append(sum(samples[i : i + factor]) // factor)
    return out, samples[n:]


def upsample_linear(samples: list[int], factor: int) -> list[int]:
    """Linear upsample by an integer factor (8 kHz -> 16 kHz uses factor=2)."""
    if factor < 1:
        raise ValueError("factor must be >= 1")
    if factor == 1 or not samples:
        return list(samples)
    out: list[int] = []
    last = len(samples) - 1
    for i, s0 in enumerate(samples):
        s1 = samples[i + 1] if i < last else s0
        out.append(_clamp_i16(s0))
        for k in range(1, factor):
            # (factor-k)/factor * s0 + k/factor * s1
            interp = ((factor - k) * s0 + k * s1) // factor
            out.append(_clamp_i16(interp))
    return out


def _split_pcmu_frames(ulaw: bytes) -> tuple[list[bytes], bytes]:
    frames: list[bytes] = []
    offset = 0
    n = len(ulaw)
    while offset + TELNYX_FRAME_BYTES <= n:
        frames.append(ulaw[offset : offset + TELNYX_FRAME_BYTES])
        offset += TELNYX_FRAME_BYTES
    return frames, ulaw[offset:]


class Pcmu8kToPcm16k:
    """Telnyx media payload (base64 PCMU 8 kHz) -> 16-bit LE PCM 16 kHz mono."""

    def __init__(self) -> None:
        self._odd = b""

    def convert_b64(self, pcmu_b64: str) -> bytes:
        try:
            ulaw = base64.b64decode(pcmu_b64)
        except Exception:
            return b""
        if not ulaw:
            return b""
        # ulaw2lin emits native-endian 16-bit; pack back as LE for Gemini.
        native = audioop.ulaw2lin(ulaw, SAMPLE_WIDTH)
        samples = list(struct.unpack_from("h" * (len(native) // SAMPLE_WIDTH), native, 0))
        pcm16 = upsample_linear(samples, GEMINI_IN_RATE // TELNYX_RATE)
        return pack_pcm16le(pcm16)


class PcmToPcmu8k:
    """Gemini Live PCM (typically 24 kHz 16-bit LE) -> 20 ms base64 PCMU frames."""

    def __init__(self, default_rate: int = GEMINI_OUT_RATE) -> None:
        self._default_rate = default_rate
        self._ratecv_state: tuple | None = None
        self._rate = default_rate
        self._byte_leftover = b""
        self._sample_leftover: list[int] = []
        self._ulaw_leftover = b""

    def convert_frames_b64(self, pcm_b64: str, mime: str | None = None) -> list[str]:
        rate = parse_pcm_rate(mime, self._default_rate)
        if rate != self._rate:
            self._rate = rate
            self._ratecv_state = None
            self._sample_leftover = []
        try:
            raw = base64.b64decode(pcm_b64)
        except Exception:
            return []
        if not raw:
            return []

        buf = self._byte_leftover + raw
        even = len(buf) - (len(buf) % SAMPLE_WIDTH)
        self._byte_leftover = buf[even:]
        samples = self._sample_leftover + unpack_pcm16le(buf[:even])
        self._sample_leftover = []

        if rate == TELNYX_RATE:
            pcm8 = samples
        elif rate > TELNYX_RATE and rate % TELNYX_RATE == 0:
            pcm8, self._sample_leftover = downsample_integer(samples, rate // TELNYX_RATE)
        else:
            packed = pack_pcm16le(samples)
            converted, self._ratecv_state = audioop.ratecv(
                packed, SAMPLE_WIDTH, 1, rate, TELNYX_RATE, self._ratecv_state
            )
            pcm8 = unpack_pcm16le(converted)

        ulaw = self._ulaw_leftover + audioop.lin2ulaw(pack_pcm16le(pcm8), SAMPLE_WIDTH)
        frames, self._ulaw_leftover = _split_pcmu_frames(ulaw)
        return [base64.b64encode(f).decode("ascii") for f in frames]

    def convert_b64(self, pcm_b64: str, mime: str | None = None) -> str:
        frames = self.convert_frames_b64(pcm_b64, mime)
        if not frames:
            return ""
        joined = b"".join(base64.b64decode(f) for f in frames)
        return base64.b64encode(joined).decode("ascii")

    def flush_frames_b64(self) -> list[str]:
        """Pad any leftover μ-law to one last 20 ms frame (μ-law 0xFF ≈ silence)."""
        if not self._ulaw_leftover:
            return []
        pad = TELNYX_FRAME_BYTES - len(self._ulaw_leftover)
        frame = self._ulaw_leftover + b"\xff" * pad
        self._ulaw_leftover = b""
        return [base64.b64encode(frame).decode("ascii")]
