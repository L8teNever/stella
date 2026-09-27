from __future__ import annotations

import hashlib
import hmac
from datetime import datetime, timezone

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_der_public_key
import base64

from stella.errors import StellaError


def verify_telnyx_signature(
    *,
    payload: bytes,
    timestamp: str,
    signature_b64: str,
    public_key_b64: str,
) -> None:
    """Verify Telnyx Ed25519 webhook signature (timestamp + '|' + raw body)."""
    if not timestamp or not signature_b64:
        raise StellaError("Missing Telnyx signature headers.", "webhook_unverified")
    try:
        ts = int(timestamp)
    except ValueError as exc:
        raise StellaError("Invalid Telnyx timestamp header.", "webhook_unverified") from exc
    now = int(datetime.now(timezone.utc).timestamp())
    if abs(now - ts) > 300:
        raise StellaError("Telnyx webhook timestamp is too old.", "webhook_unverified")

    signed = f"{timestamp}|".encode("utf-8") + payload
    try:
        key_bytes = base64.b64decode(public_key_b64)
        sig = base64.b64decode(signature_b64)
    except Exception as exc:
        raise StellaError("Could not decode Telnyx signature or public key.", "webhook_unverified") from exc

    public_key = _load_ed25519(key_bytes)
    try:
        public_key.verify(sig, signed)
    except InvalidSignature as exc:
        raise StellaError("Telnyx webhook signature mismatch.", "webhook_unverified") from exc


def _load_ed25519(key_bytes: bytes) -> Ed25519PublicKey:
    if len(key_bytes) == 32:
        return Ed25519PublicKey.from_public_bytes(key_bytes)
    loaded = load_der_public_key(key_bytes)
    if not isinstance(loaded, Ed25519PublicKey):
        raise StellaError("TELNYX_PUBLIC_KEY is not an Ed25519 key.", "webhook_unverified")
    return loaded


def timing_safe_equal(a: str, b: str) -> bool:
    return hmac.compare_digest(
        hashlib.sha256(a.encode()).digest(),
        hashlib.sha256(b.encode()).digest(),
    )
