from deepeval.models import OpenRouterModel

from cobalt_alpha.config import get_settings


def openrouter_eval_model(model: str) -> OpenRouterModel:
    settings = get_settings()
    if settings.llm_provider != "openrouter":
        raise ValueError("OpenRouter eval model requires LLM_PROVIDER=openrouter")

    return OpenRouterModel(
        model=model,
        api_key=settings.llm_api_key.get_secret_value(),
        base_url=settings.llm_base_url,
    )
