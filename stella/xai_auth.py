"""xAI / SuperGrok auth: OAuth device-code (subscription quota) + API-key fallback."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from stella.config import Settings
from stella.errors import StellaError

DEVICE_CODE_URL = "https://auth.x.ai/oauth2/device/code"
TOKEN_URL = "https://auth.x.ai/oauth2/token"
DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
DEFAULT_SCOPE = (
    "openid profile email offline_access grok-cli:access api:access "
    "conversations:read conversations:write"
)


@dataclass
class DeviceLogin:
    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str
    interval: int
    expires_in: int


class XAIAuth:
    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self.settings = settings
        self._client = client or httpx.Client(timeout=30.0)

    def bearer_token(self) -> str:
        """Prefer live SuperGrok OAuth access token; fall back to XAI_API_KEY."""
        oauth = self._load_tokens()
        if oauth:
            token = self._ensure_fresh(oauth)
            return token["access_token"]
        if self.settings.xai_api_key:
            return self.settings.xai_api_key
        raise StellaError(
            "No xAI credentials. Set XAI_API_KEY or run `stella oauth login` "
            "(SuperGrok / X Premium+ device flow).",
            "xai_auth_missing",
        )

    def auth_mode(self) -> str:
        if self._load_tokens():
            return "oauth"
        if self.settings.xai_api_key:
            return "api_key"
        return "none"

    def start_device_login(self) -> DeviceLogin:
        resp = self._client.post(
            DEVICE_CODE_URL,
            data={
                "client_id": self.settings.xai_oauth_client_id,
                "scope": DEFAULT_SCOPE,
            },
            headers={"Accept": "application/json"},
        )
        if resp.status_code >= 400:
            raise StellaError(
                f"xAI device-code request failed ({resp.status_code}): {resp.text}",
                "xai_oauth_error",
            )
        body = resp.json()
        return DeviceLogin(
            device_code=body["device_code"],
            user_code=body["user_code"],
            verification_uri=body.get("verification_uri") or body.get("verification_url", ""),
            verification_uri_complete=body.get("verification_uri_complete", ""),
            interval=max(int(body.get("interval") or 5), 1),
            expires_in=int(body.get("expires_in") or 600),
        )

    def poll_device_login(self, login: DeviceLogin, *, max_wait: int | None = None) -> dict[str, Any]:
        deadline = time.time() + (max_wait or login.expires_in)
        interval = login.interval
        while time.time() < deadline:
            resp = self._client.post(
                TOKEN_URL,
                data={
                    "grant_type": DEVICE_GRANT,
                    "client_id": self.settings.xai_oauth_client_id,
                    "device_code": login.device_code,
                },
                headers={"Accept": "application/json"},
            )
            body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            if resp.status_code < 400 and body.get("access_token"):
                self._save_tokens(body)
                return body
            error = body.get("error") or ""
            if error == "authorization_pending":
                time.sleep(interval)
                continue
            if error == "slow_down":
                interval = min(interval + 5, 30)
                time.sleep(interval)
                continue
            if error in {"access_denied", "authorization_denied", "expired_token"}:
                raise StellaError(f"xAI OAuth failed: {error}", "xai_oauth_denied")
            if resp.status_code >= 400:
                time.sleep(interval)
                continue
            time.sleep(interval)
        raise StellaError("xAI OAuth timed out waiting for approval.", "xai_oauth_timeout")

    def _ensure_fresh(self, tokens: dict[str, Any]) -> dict[str, Any]:
        expires_at = float(tokens.get("expires_at") or 0)
        if expires_at and expires_at > time.time() + 60:
            return tokens
        refresh = tokens.get("refresh_token")
        if not refresh:
            raise StellaError(
                "Stored xAI OAuth token expired and has no refresh_token. Run `stella oauth login`.",
                "xai_oauth_expired",
            )
        resp = self._client.post(
            TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "client_id": self.settings.xai_oauth_client_id,
                "refresh_token": refresh,
            },
            headers={"Accept": "application/json"},
        )
        if resp.status_code >= 400:
            raise StellaError(
                f"xAI token refresh failed ({resp.status_code}): {resp.text}. Run `stella oauth login`.",
                "xai_oauth_refresh_failed",
            )
        body = resp.json()
        if not body.get("refresh_token"):
            body["refresh_token"] = refresh
        self._save_tokens(body)
        return self._load_tokens() or body

    def _token_path(self) -> Path:
        return Path(self.settings.xai_oauth_token_path)

    def _load_tokens(self) -> dict[str, Any] | None:
        path = self._token_path()
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if not data.get("access_token"):
            return None
        return data

    def _save_tokens(self, body: dict[str, Any]) -> None:
        expires_in = int(body.get("expires_in") or 3600)
        stored = {
            "access_token": body["access_token"],
            "refresh_token": body.get("refresh_token", ""),
            "expires_in": expires_in,
            "expires_at": time.time() + expires_in,
            "token_type": body.get("token_type", "Bearer"),
            "scope": body.get("scope", ""),
        }
        path = self._token_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stored, indent=2), encoding="utf-8")
        path.chmod(0o600)
