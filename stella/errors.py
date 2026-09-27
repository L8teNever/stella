from __future__ import annotations

import phonenumbers


class StellaError(Exception):
    def __init__(self, message: str, code: str = "stella_error") -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def to_dict(self) -> dict:
        return {"error": self.code, "message": self.message}


def normalize_e164(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        raise StellaError("Missing destination number (`to`).", "invalid_number")
    try:
        parsed = phonenumbers.parse(text, None)
    except phonenumbers.NumberParseException as exc:
        raise StellaError(f"Invalid E.164 number: {text}", "invalid_number") from exc
    if not phonenumbers.is_valid_number(parsed):
        raise StellaError(f"Invalid E.164 number: {text}", "invalid_number")
    return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
