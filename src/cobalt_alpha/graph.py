from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph, RetryPolicy
from langgraph.types import Command
from pydantic import BaseModel
from pydantic.fields import Field

from cobalt_alpha.config import load_prompt
from cobalt_alpha.llm_models import llm_lite, llm_max
from cobalt_alpha.setup_phoenix import setup_phoenix


class RouterDecision(BaseModel):
    score: float = Field(
        ge=0.0, le=1.0, description="Оцена сложности запроса пользователя"
    )


class GraphState(BaseModel):
    raw_user_input: str = Field("", description="Необработанный запрос пользователя")
    complexity_estimation: float = 0.0
    selected_model: str = ""
    model_response: str = ""


def node_router(state: GraphState) -> GraphState:
    "Оценивает сложность запроса пользователя"
    llm = llm_lite().with_structured_output(RouterDecision)
    router_prompt = load_prompt("node_router.md")

    response = llm.invoke(
        [
            ("system", router_prompt),
            ("human", state.raw_user_input),
        ]
    )
    decision = RouterDecision.model_validate(response)

    return state.model_copy(update={"complexity_estimation": decision.score})


def node_lite(state: GraphState) -> GraphState:
    "Дешевая модель"
    llm = llm_lite()
    response = llm.invoke(state.raw_user_input)

    return state.model_copy(
        update={
            "model_response": str(response.content),
        }
    )


def node_max(state: GraphState) -> GraphState:
    "Дорогая модель"
    llm = llm_max()
    response = llm.invoke(state.raw_user_input)

    return state.model_copy(
        update={
            "model_response": str(response.content),
        }
    )


def route_after_router(state: GraphState) -> Command[Literal["node_lite", "node_max"]]:
    if state.complexity_estimation > 0.5:
        return Command(goto="node_max", update={"selected_model": "max"})
    return Command(goto="node_lite", update={"selected_model": "lite"})


def build_graph() -> CompiledStateGraph[GraphState, None, GraphState, GraphState]:
    setup_phoenix()  # TODO seprikov: move from build_graph to web_server

    graph = StateGraph(GraphState)

    graph.add_node("node_router", node_router, retry_policy=RetryPolicy(max_attempts=3))
    graph.add_node("route_after_router", route_after_router)
    graph.add_node("node_lite", node_lite)
    graph.add_node("node_max", node_max)

    graph.add_edge(START, "node_router")
    graph.add_edge("node_router", "route_after_router")
    graph.add_edge("node_lite", END)
    graph.add_edge("node_max", END)

    return graph.compile()
