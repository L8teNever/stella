"""Telnyx Call Control HTTP client. Failures are explicit errors, not silent skips."""

from __future__ import annotations

from typing import Any, Protocol

import httpx

from stella.config import Settings
from stella.errors import StellaError


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
    ) -> dict[str, Any]:
        self._require_config()
        payload: dict[str, Any] = {
            "to": to,
            "from": self.settings.telnyx_from_number,
            "connection_id": self.settings.telnyx_connection_id,
            "webhook_url": webhook_url,
            "webhook_url_method": "POST",
            "stream_url": stream_url,
            "stream_track": "inbound_track",
            "stream_bidirectional_mode": "rtp",
            "stream_bidirectional_codec": "PCMU",
            "stream_bidirectional_sampling_rate": 8000,
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

    def hangup(self, call_control_id: str) -> dict[str, Any]:
        self._require_config()
        resp = self._http.request(
            "POST",
            self._url(f"/calls/{call_control_id}/actions/hangup"),
            headers=self._headers(),
            json={},
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
