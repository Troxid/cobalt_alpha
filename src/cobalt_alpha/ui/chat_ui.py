from functools import lru_cache
from typing import Any

import gradio as gr
from langgraph.checkpoint.memory import MemorySaver

from cobalt_alpha.graph import build_graph
from cobalt_alpha.setup_phoenix import setup_phoenix
from cobalt_alpha.state import GraphState


@lru_cache(maxsize=1)
def get_graph() -> Any:
    setup_phoenix()
    return build_graph(checkpointer=MemorySaver())


def respond(message: str, history: list[dict[str, str]], request: gr.Request) -> str:
    del history

    user_input = message.strip()
    if not user_input:
        return "Введите запрос."

    graph = get_graph()
    thread_id = request.session_hash or "default"
    result = graph.invoke(
        {"current_user_input": user_input},
        config={"configurable": {"thread_id": thread_id}},
    )
    state = GraphState.model_validate(result)

    if state.execution_error:
        return f"{state.model_response}\n\nОшибка выполнения: {state.execution_error}"

    return state.model_response


def create_demo() -> gr.ChatInterface:
    return gr.ChatInterface(
        fn=respond,
        title="Cobalt|Alpha",
        description="AI-агент для научных, инженерных и математических расчётов.",
        analytics_enabled=False,
        flagging_mode="never",
    )


def main() -> None:
    create_demo().launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
