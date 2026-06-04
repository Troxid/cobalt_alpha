from typing import Literal

from langchain_openai.chat_models import ChatOpenAI

from cobalt_alpha.config import Settings, get_settings

LLMSize = Literal["lite", "max"]


def _chat_openai(
    settings: Settings,
    model: str,
    *,
    timeout: float,
    max_completion_tokens: int,
) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=model,
        timeout=timeout,
        max_completion_tokens=max_completion_tokens,
    )


def _model_name(settings: Settings, size: LLMSize) -> str:
    match settings.llm_provider, size:
        case "openrouter", "max":
            # return "google/gemma-4-26b-a4b-it"
            return "google/gemma-4-31b-it"
        case "openrouter", "lite":
            return "google/gemma-3-4b-it"
        case "ollama", "max":
            return "gemma4:26b"
        case "ollama", "lite":
            return "gemma4:e4b"
        case _:
            raise ValueError(
                f"Unsupported LLM provider/size: {settings.llm_provider}/{size}"
            )


def _llm(
    size: LLMSize,
    *,
    timeout: float,
    max_completion_tokens: int,
) -> ChatOpenAI:
    settings = get_settings()
    return _chat_openai(
        settings,
        model=_model_name(settings, size),
        timeout=timeout,
        max_completion_tokens=max_completion_tokens,
    )


def llm_lite() -> ChatOpenAI:
    return _llm("lite", timeout=30, max_completion_tokens=2000)


def llm_max() -> ChatOpenAI:
    return _llm("max", timeout=60, max_completion_tokens=2000)


def llm_router() -> ChatOpenAI:
    return _llm("lite", timeout=20, max_completion_tokens=100)


def llm_direct() -> ChatOpenAI:
    return _llm("lite", timeout=30, max_completion_tokens=1000)


def llm_planner() -> ChatOpenAI:
    return _llm("max", timeout=60, max_completion_tokens=2000)


def llm_verifier() -> ChatOpenAI:
    return _llm("max", timeout=30, max_completion_tokens=500)


def llm_codegen() -> ChatOpenAI:
    return _llm("lite", timeout=60, max_completion_tokens=10000)
