"""G.711 (PCMU μ-law / PCMA A-law, 8 kHz) <-> linear PCM for Telnyx <-> Gemini.

Telnyx bidirectional RTP expects G.711 frames whose payload length is a
multiple of 160 bytes (20 ms at 8 kHz). German PSTN typically answers PCMA;
US numbers typically use PCMU. Gemini Live uses 16-bit PCM little-endian
(typically 16 kHz in, 24 kHz out).

Downsampling 24 kHz -> 8 kHz with audioop.ratecv (linear interpolation)
aliases badly and often emits a length that is not 20 ms-aligned. Sending
that blob as one Telnyx media event mis-frames RTP and sounds swapped,
choppy, or distorted. Integer-ratio decimation with a proper windowed-sinc
low-pass (telephone band, ~3.6 kHz) + 20 ms framing fixes that and keeps
sibilants from folding back as harshness.
"""

from __future__ import annotations

import audioop
import base64
import re
import struct
from functools import lru_cache

import numpy as np

TELNYX_RATE = 8000
TELNYX_FRAME_SAMPLES = 160  # 20 ms
TELNYX_FRAME_BYTES = TELNYX_FRAME_SAMPLES  # G.711 is 1 byte/sample
SAMPLE_WIDTH = 2
GEMINI_IN_RATE = 16000
GEMINI_OUT_RATE = 24000
CODEC_PCMU = "PCMU"
CODEC_PCMA = "PCMA"
# μ-law 0xFF and A-law 0xD5 are the encodings of linear 0 (idle).
G711_SILENCE = {CODEC_PCMU: b"\xff", CODEC_PCMA: b"\xd5"}

_RATE_RE = re.compile(r"rate=(\d+)", re.IGNORECASE)
_INT16_MIN = -32768
_INT16_MAX = 32767


def normalize_g711_codec(raw: object | None) -> str:
    """Map Telnyx / SIP labels to PCMU or PCMA. Unknown values default to PCMU."""
    text = str(raw or "").strip().upper()
    compact = (
        text.replace("-", "")
        .replace("_", "")
        .replace(" ", "")
        .replace(".", "")
        .replace("/", "")
    )
    if any(token in compact for token in ("PCMA", "ALAW", "G711A")):
        return CODEC_PCMA
    return CODEC_PCMU


def preferred_telnyx_codec(*e164: str | None) -> str:
    """PCMA for German (+49) PSTN legs; PCMU otherwise (US μ-law)."""
    for number in e164:
        digits = "".join(ch for ch in str(number or "") if ch.isdigit())
        if digits.startswith("49"):
            return CODEC_PCMA
    return CODEC_PCMU


def codec_from_telnyx_payload(payload: dict | None) -> str | None:
    """Read codec from call.answered / streaming / media start payloads."""
    if not isinstance(payload, dict):
        return None
    candidates: list[object] = [
        payload.get("codec"),
        payload.get("encoding"),
        payload.get("stream_bidirectional_codec"),
    ]
    media_format = payload.get("media_format") or payload.get("mediaFormat")
    if isinstance(media_format, dict):
        candidates.append(media_format.get("encoding"))
        candidates.append(media_format.get("codec"))
    start = payload.get("start")
    if isinstance(start, dict):
        nested = codec_from_telnyx_payload(start)
        if nested:
            return nested
    for raw in candidates:
        if raw is None or str(raw).strip() == "":
            continue
        return normalize_g711_codec(raw)
    return None


def g711_to_lin(companded: bytes, codec: str) -> bytes:
    if normalize_g711_codec(codec) == CODEC_PCMA:
        return audioop.alaw2lin(companded, SAMPLE_WIDTH)
    return audioop.ulaw2lin(companded, SAMPLE_WIDTH)


def lin_to_g711(pcm_native: bytes, codec: str) -> bytes:
    if normalize_g711_codec(codec) == CODEC_PCMA:
        return audioop.lin2alaw(pcm_native, SAMPLE_WIDTH)
    return audioop.lin2ulaw(pcm_native, SAMPLE_WIDTH)


def g711_silence_byte(codec: str) -> bytes:
    return G711_SILENCE[normalize_g711_codec(codec)]


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


@lru_cache(maxsize=8)
def _lowpass_taps(rate: int, cutoff_hz: float, taps: int) -> np.ndarray:
    """Kaiser-windowed sinc low-pass, unity DC gain."""
    n = np.arange(taps) - (taps - 1) / 2
    fc = cutoff_hz / rate
    h = 2 * fc * np.sinc(2 * fc * n) * np.kaiser(taps, 8.0)
    return (h / h.sum()).astype(np.float64)


class FirDecimator:
    """Streaming anti-aliased integer decimation (e.g. 24 kHz -> 8 kHz), chunk-safe."""

    def __init__(self, rate: int, factor: int, cutoff_hz: float = 3600.0, taps: int = 63) -> None:
        self._factor = factor
        self._h = _lowpass_taps(rate, cutoff_hz, taps)
        # Zero-primed history so the output length equals len(input) // factor.
        self._tail = np.zeros(taps - 1)
        self._phase = 0

    def process(self, samples: list[int]) -> list[int]:
        if not samples:
            return []
        buf = np.concatenate([self._tail, np.asarray(samples, dtype=np.float64)])
        filtered = np.convolve(buf, self._h, mode="valid")  # len == len(samples)
        self._tail = buf[-(len(self._h) - 1):]
        out = filtered[self._phase :: self._factor]
        self._phase = (self._phase - len(filtered)) % self._factor
        return np.clip(np.rint(out), _INT16_MIN, _INT16_MAX).astype(int).tolist()


class FirUpsampler2:
    """Streaming 2x interpolation (8 -> 16 kHz) with an image-rejection low-pass."""

    def __init__(self, rate_out: int = GEMINI_IN_RATE, cutoff_hz: float = 3700.0, taps: int = 31) -> None:
        self._h = _lowpass_taps(rate_out, cutoff_hz, taps) * 2  # compensate zero stuffing
        self._tail = np.zeros(taps - 1)

    def process(self, samples: list[int]) -> list[int]:
        if not samples:
            return []
        stuffed = np.zeros(len(samples) * 2)
        stuffed[::2] = samples
        buf = np.concatenate([self._tail, stuffed])
        out = np.convolve(buf, self._h, mode="valid")
        self._tail = buf[-(len(self._h) - 1):]
        return np.clip(np.rint(out), _INT16_MIN, _INT16_MAX).astype(int).tolist()


def _split_g711_frames(companded: bytes) -> tuple[list[bytes], bytes]:
    frames: list[bytes] = []
    offset = 0
    n = len(companded)
    while offset + TELNYX_FRAME_BYTES <= n:
        frames.append(companded[offset : offset + TELNYX_FRAME_BYTES])
        offset += TELNYX_FRAME_BYTES
    return frames, companded[offset:]


_split_pcmu_frames = _split_g711_frames


class Pcmu8kToPcm16k:
    """Telnyx media payload (base64 G.711 8 kHz) -> 16-bit LE PCM 16 kHz mono."""

    def __init__(self, codec: str = CODEC_PCMU) -> None:
        self.codec = normalize_g711_codec(codec)
        self._odd = b""
        self._up = FirUpsampler2()

    def convert_b64(self, pcmu_b64: str) -> bytes:
        try:
            companded = base64.b64decode(pcmu_b64)
        except Exception:
            return b""
        if not companded:
            return b""
        # audioop *law2lin emits native-endian 16-bit; pack back as LE for Gemini.
        native = g711_to_lin(companded, self.codec)
        samples = list(struct.unpack_from("h" * (len(native) // SAMPLE_WIDTH), native, 0))
        pcm16 = self._up.process(samples)
        return pack_pcm16le(pcm16)


class PcmToPcmu8k:
    """Gemini Live PCM (typically 24 kHz 16-bit LE) -> 20 ms base64 G.711 frames."""

    def __init__(self, default_rate: int = GEMINI_OUT_RATE, codec: str = CODEC_PCMU) -> None:
        self.codec = normalize_g711_codec(codec)
        self._default_rate = default_rate
        self._ratecv_state: tuple | None = None
        self._rate = default_rate
        self._byte_leftover = b""
        self._sample_leftover: list[int] = []
        self._ulaw_leftover = b""
        self._decimator: FirDecimator | None = None

    def convert_frames_b64(self, pcm_b64: str, mime: str | None = None) -> list[str]:
        rate = parse_pcm_rate(mime, self._default_rate)
        if rate != self._rate:
            self._rate = rate
            self._ratecv_state = None
            self._sample_leftover = []
            self._decimator = None
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
            if self._decimator is None:
                self._decimator = FirDecimator(rate, rate // TELNYX_RATE)
            keep = (len(samples) // (rate // TELNYX_RATE)) * (rate // TELNYX_RATE)
            pcm8 = self._decimator.process(samples[:keep])
            self._sample_leftover = samples[keep:]
        else:
            packed = pack_pcm16le(samples)
            converted, self._ratecv_state = audioop.ratecv(
                packed, SAMPLE_WIDTH, 1, rate, TELNYX_RATE, self._ratecv_state
            )
            pcm8 = unpack_pcm16le(converted)

        ulaw = self._ulaw_leftover + lin_to_g711(pack_pcm16le(pcm8), self.codec)
        frames, self._ulaw_leftover = _split_g711_frames(ulaw)
        return [base64.b64encode(f).decode("ascii") for f in frames]

    def convert_b64(self, pcm_b64: str, mime: str | None = None) -> str:
        frames = self.convert_frames_b64(pcm_b64, mime)
        if not frames:
            return ""
        joined = b"".join(base64.b64decode(f) for f in frames)
        return base64.b64encode(joined).decode("ascii")

    def flush_frames_b64(self) -> list[str]:
        """Pad leftover G.711 to one last 20 ms frame (codec idle byte ≈ silence)."""
        if not self._ulaw_leftover:
            return []
        pad = TELNYX_FRAME_BYTES - len(self._ulaw_leftover)
        frame = self._ulaw_leftover + g711_silence_byte(self.codec) * pad
        self._ulaw_leftover = b""
        return [base64.b64encode(frame).decode("ascii")]
