from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    stella_host: str = "0.0.0.0"
    stella_port: int = 8080
    stella_public_base_url: str = "http://localhost:8080"
    stella_callback_url: str = ""
    stella_db_path: str = "/data/stella.db"
    stella_mcp_token: str = ""
    stella_skip_webhook_verify: bool = False

    telnyx_api_key: str = ""
    telnyx_connection_id: str = ""
    telnyx_from_number: str = ""
    telnyx_public_key: str = ""
    telnyx_api_base: str = "https://api.telnyx.com/v2"

    xai_api_key: str = ""
    xai_voice_model: str = "grok-voice-latest"
    xai_voice: str = "eve"
    xai_oauth_token_path: str = "/data/xai_oauth.json"
    xai_oauth_client_id: str = "b1a00492-073a-47ea-816f-4c329264a828"
    xai_realtime_url: str = "wss://api.x.ai/v1/realtime"

    def data_dir(self) -> Path:
        p = Path(self.stella_db_path).parent
        p.mkdir(parents=True, exist_ok=True)
        token_parent = Path(self.xai_oauth_token_path).parent
        token_parent.mkdir(parents=True, exist_ok=True)
        return p

    def public_http_url(self, path: str) -> str:
        base = self.stella_public_base_url.rstrip("/")
        if not path.startswith("/"):
            path = "/" + path
        return base + path

    def public_ws_url(self, path: str) -> str:
        http = self.public_http_url(path)
        if http.startswith("https://"):
            return "wss://" + http[len("https://") :]
        if http.startswith("http://"):
            return "ws://" + http[len("http://") :]
        return http


@lru_cache
def get_settings() -> Settings:
    return Settings()
