from langchain_openai import ChatOpenAI

from cobalt_alpha.config import get_settings


def test_llm_invoke():
    settings = get_settings()

    llm = ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model="gemma:latest",
        timeout=60,
    )

    response = llm.invoke("привет")
    print(response)

    assert response is not None
    assert getattr(response, "content", None) is not None
