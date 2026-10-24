"""Telnyx Call Control HTTP client. Failures are explicit errors, not silent skips."""

from __future__ import annotations

import logging
from typing import Any, Protocol

import httpx

from stella.audio_pcmu import CODEC_PCMU, preferred_telnyx_codec
from stella.config import Settings
from stella.errors import StellaError

logger = logging.getLogger(__name__)

# Telnyx: the leg is already gone. Hangup and retrieve both surface this.
CALL_ALREADY_ENDED = "90015"


class TelnyxTransport(Protocol):
    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response: ...


class TelnyxClient:
    def __init__(self, settings: Settings, transport: TelnyxTransport | None = None) -> None:
        self.settings = settings
        self._http = transport or httpx.Client(timeout=30.0)

    def _require_config(self) -> None:
        missing = []
        if not self.settings.telnyx_api_key:
            missing.append("TELNYX_API_KEY")
        if not self.settings.telnyx_connection_id:
            missing.append("TELNYX_CONNECTION_ID")
        if not self.settings.telnyx_from_number:
            missing.append("TELNYX_FROM_NUMBER")
        if missing:
            raise StellaError(
                "Telnyx is not configured. Set: " + ", ".join(missing),
                "telnyx_config_missing",
            )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.telnyx_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _url(self, path: str) -> str:
        return self.settings.telnyx_api_base.rstrip("/") + path

    def _parse(self, resp: httpx.Response, action: str) -> dict[str, Any]:
        if resp.status_code >= 400:
            raise StellaError(
                f"Telnyx {action} failed ({resp.status_code}): {resp.text}",
                "telnyx_api_error",
            )
        try:
            return resp.json()
        except ValueError as exc:
            raise StellaError(f"Telnyx {action} returned non-JSON: {resp.text}", "telnyx_api_error") from exc

    def dial(
        self,
        *,
        to: str,
        webhook_url: str,
        stream_url: str,
        client_state: str | None = None,
        stream_bidirectional_codec: str | None = None,
    ) -> dict[str, Any]:
        self._require_config()
        codec = self._normalize_codec(stream_bidirectional_codec, to)
        payload: dict[str, Any] = {
            "to": to,
            "from": self.settings.telnyx_from_number,
            "connection_id": self.settings.telnyx_connection_id,
            "webhook_url": webhook_url,
            "webhook_url_method": "POST",
            **self._stream_fields(stream_url, codec),
            "answering_machine_detection": "detect",
        }
        if client_state:
            payload["client_state"] = client_state
        resp = self._http.request("POST", self._url("/calls"), headers=self._headers(), json=payload)
        data = self._parse(resp, "dial")
        inner = data.get("data") or data
        call_control_id = inner.get("call_control_id")
        if not call_control_id:
            raise StellaError(
                f"Telnyx dial succeeded but response had no call_control_id: {data}",
                "telnyx_api_error",
            )
        return inner

    def _normalize_codec(self, codec: str | None, *numbers: str | None) -> str:
        chosen = codec or preferred_telnyx_codec(self.settings.telnyx_from_number, *numbers)
        return chosen if chosen in {"PCMU", "PCMA"} else CODEC_PCMU

    def _stream_fields(self, stream_url: str, codec: str) -> dict[str, Any]:
        # inbound_track = remote-party audio (caller on inbound, callee on
        # outbound). both_tracks echoes our outbound RTP into the model.
        return {
            "stream_url": stream_url,
            "stream_track": "inbound_track",
            "stream_bidirectional_mode": "rtp",
            "stream_bidirectional_codec": codec,
            "stream_bidirectional_sampling_rate": 8000,
        }

    def _action(self, call_control_id: str, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_config()
        resp = self._http.request(
            "POST",
            self._url(f"/calls/{call_control_id}/actions/{action}"),
            headers=self._headers(),
            json=payload,
        )
        return self._parse(resp, action)

    def answer(
        self,
        call_control_id: str,
        *,
        client_state: str | None = None,
        webhook_url: str | None = None,
        stream_url: str | None = None,
        stream_bidirectional_codec: str | None = None,
        party_number: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if client_state:
            payload["client_state"] = client_state
        if webhook_url:
            payload["webhook_url"] = webhook_url
            payload["webhook_url_method"] = "POST"
        if stream_url:
            codec = self._normalize_codec(stream_bidirectional_codec, party_number)
            payload.update(self._stream_fields(stream_url, codec))
        return self._action(call_control_id, "answer", payload)

    def streaming_start(
        self,
        call_control_id: str,
        *,
        stream_url: str,
        stream_bidirectional_codec: str | None = None,
        party_number: str | None = None,
        client_state: str | None = None,
    ) -> dict[str, Any]:
        codec = self._normalize_codec(stream_bidirectional_codec, party_number)
        payload: dict[str, Any] = self._stream_fields(stream_url, codec)
        if client_state:
            payload["client_state"] = client_state
        return self._action(call_control_id, "streaming_start", payload)

    def hangup(self, call_control_id: str) -> dict[str, Any]:
        self._require_config()
        resp = self._http.request(
            "POST",
            self._url(f"/calls/{call_control_id}/actions/hangup"),
            headers=self._headers(),
            json={},
        )
        if CALL_ALREADY_ENDED in response_error_codes(resp):
            raise StellaError(
                "Telnyx hangup: call already ended (90015)",
                "telnyx_call_ended",
            )
        return self._parse(resp, "hangup")

    def speak(self, call_control_id: str, text: str, *, language: str = "en-US") -> dict[str, Any]:
        self._require_config()
        resp = self._http.request(
            "POST",
            self._url(f"/calls/{call_control_id}/actions/speak"),
            headers=self._headers(),
            json={"payload": text, "voice": "female", "language": language},
        )
        return self._parse(resp, "speak")

    def retrieve(self, call_control_id: str) -> dict[str, Any]:
        self._require_config()
        resp = self._http.request(
            "GET",
            self._url(f"/calls/{call_control_id}"),
            headers=self._headers(),
        )
        return self._parse(resp, "retrieve")

    def call_liveness(self, call_control_id: str) -> str:
        """Whether Telnyx still has this leg.

        Returns `ended_90015` (error 90015), `ended` (`is_alive` false or a
        terminal call state), `alive`, or `unknown` when the API cannot be read.
        Never raises: the reaper treats `unknown` as "decide from age only".
        """
        if not (call_control_id or "").strip():
            return "unknown"
        try:
            self._require_config()
        except StellaError:
            return "unknown"
        try:
            resp = self._http.request(
                "GET",
                self._url(f"/calls/{call_control_id}"),
                headers=self._headers(),
                timeout=5.0,
            )
        except Exception:
            logger.warning("telnyx retrieve failed for %s", call_control_id, exc_info=True)
            return "unknown"
        if CALL_ALREADY_ENDED in response_error_codes(resp):
            return "ended_90015"
        if resp.status_code >= 400:
            return "unknown"
        try:
            data = resp.json()
        except ValueError:
            return "unknown"
        inner = data.get("data") if isinstance(data, dict) else None
        if not isinstance(inner, dict):
            inner = data if isinstance(data, dict) else {}
        if inner.get("is_alive") is False:
            return "ended"
        state = str(inner.get("state") or inner.get("status") or "").strip().lower()
        if state in {"hangup", "ended", "destroyed", "completed"}:
            return "ended"
        if inner.get("is_alive") is True or inner.get("call_control_id"):
            return "alive"
        return "unknown"


def response_error_codes(resp: httpx.Response) -> set[str]:
    """Telnyx `errors[].code` values. 90015 is also accepted from a non-JSON body."""
    codes: set[str] = set()
    body: Any = None
    try:
        body = resp.json()
    except Exception:
        body = None
    if isinstance(body, dict):
        errors = body.get("errors")
        if isinstance(errors, list):
            for err in errors:
                if isinstance(err, dict) and err.get("code") is not None:
                    codes.add(str(err["code"]))
    if not codes and CALL_ALREADY_ENDED in (getattr(resp, "text", None) or ""):
        codes.add(CALL_ALREADY_ENDED)
    return codes
