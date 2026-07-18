from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the thin gateway in front of vLLM."""

    vllm_base_url: str = "http://localhost:8000"
    vllm_api_key: str = "local-token"
    model_name: str = "Qwen/Qwen3-1.7B"
    request_timeout_s: float = 180.0

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
