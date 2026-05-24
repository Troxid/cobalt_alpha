from langchain_openai.chat_models import ChatOpenAI

from cobalt_alpha.config import get_settings


def llm_lite():
    settings = get_settings()
    llm = ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model="gemma4:e4b",
        timeout=60,
    )
    return llm


def llm_max():
    settings = get_settings()
    llm = ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model="gemma4:26b",
        timeout=60,
    )
    return llm
