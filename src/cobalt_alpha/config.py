from functools import lru_cache
from importlib.resources import files
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["ollama", "openrouter"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: LLMProvider = Field(
        default="ollama",
        validation_alias="LLM_PROVIDER",
    )
    llm_base_url: str = Field(
        default="",
        validation_alias="LLM_BASE_URL",
        min_length=1,
    )
    llm_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="LLM_API_KEY",
        min_length=1,
    )
    ollama_models_dir: str = Field(
        default="",
        validation_alias="OLLAMA_MODELS_DIR",
    )
    phoenix_enabled: bool = Field(default=True, validation_alias="PHOENIX_ENABLED")
    phoenix_project_name: str = Field(
        default="cobalt_alpha",
        validation_alias="PHOENIX_PROJECT_NAME",
    )
    phoenix_collector_endpoint: str = Field(
        default="",
        validation_alias="PHOENIX_COLLECTOR_ENDPOINT",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def load_prompt(name: str) -> str:
    return (files("cobalt_alpha.prompts") / name).read_text(encoding="utf-8")


def load_eval_prompt(name: str) -> str:
    return (files("cobalt_alpha.eval_prompts") / name).read_text(encoding="utf-8")
