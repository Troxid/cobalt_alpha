from functools import lru_cache
from typing import Any

from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register

from cobalt_alpha.config import get_settings


@lru_cache(maxsize=1)
def setup_phoenix() -> None:
    settings = get_settings()
    if not settings.phoenix_enabled:
        return

    register_kwargs: dict[str, Any] = {
        "project_name": settings.phoenix_project_name,
        "endpoint": settings.phoenix_collector_endpoint,
    }

    tracer_provider = register(**register_kwargs)
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
