from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph, RetryPolicy

from cobalt_alpha.nodes import (
    NODE_CODEGEN,
    NODE_DIRECT_ANSWER,
    NODE_EVAL_PYTHON,
    NODE_INGEST_USER_INPUT,
    NODE_PLAN_FAILED,
    NODE_PLANNER,
    NODE_PREPARE_PLANNER,
    NODE_ROUTER,
    NODE_VERIFIER,
    node_codegen,
    node_direct_answer,
    node_eval_python,
    node_ingest_user_input,
    node_plan_failed,
    node_planner,
    node_prepare_planner,
    node_router,
    node_verifier,
)
from cobalt_alpha.state import GraphState


def route_after_router(
    state: GraphState,
) -> Literal[
    "node_direct_answer",
    "node_prepare_planner",
]:
    if state.router_intent == "direct_answer":
        return NODE_DIRECT_ANSWER

    return NODE_PREPARE_PLANNER


def route_after_verifier(
    state: GraphState,
) -> Literal["node_planner", "node_codegen", "node_plan_failed"]:
    if state.plan_status == "verified" and state.steps_status == "verified":
        return NODE_CODEGEN

    if state.verification_attempts < state.max_verification_attempts:
        return NODE_PLANNER

    return NODE_PLAN_FAILED


def route_after_eval(
    state: GraphState,
) -> str:
    if (
        state.execution_ok is False
        and state.codegen_attempts < state.max_codegen_attempts
    ):
        return NODE_CODEGEN

    return END


def build_graph(
    checkpointer=None,
) -> CompiledStateGraph[GraphState, None, GraphState, GraphState]:
    graph = StateGraph(GraphState)

    graph.add_node(NODE_INGEST_USER_INPUT, node_ingest_user_input)
    graph.add_node(NODE_ROUTER, node_router, retry_policy=RetryPolicy(max_attempts=3))
    graph.add_node(
        NODE_DIRECT_ANSWER, node_direct_answer, retry_policy=RetryPolicy(max_attempts=3)
    )
    graph.add_node(
        NODE_PREPARE_PLANNER,
        node_prepare_planner,
        retry_policy=RetryPolicy(max_attempts=2),
    )
    graph.add_node(NODE_PLANNER, node_planner, retry_policy=RetryPolicy(max_attempts=2))
    graph.add_node(
        NODE_VERIFIER, node_verifier, retry_policy=RetryPolicy(max_attempts=2)
    )
    graph.add_node(NODE_CODEGEN, node_codegen, retry_policy=RetryPolicy(max_attempts=2))
    graph.add_node(NODE_EVAL_PYTHON, node_eval_python)
    graph.add_node(NODE_PLAN_FAILED, node_plan_failed)

    graph.add_edge(START, NODE_INGEST_USER_INPUT)
    graph.add_edge(NODE_INGEST_USER_INPUT, NODE_ROUTER)
    graph.add_conditional_edges(
        NODE_ROUTER,
        route_after_router,
        [
            NODE_DIRECT_ANSWER,
            NODE_PREPARE_PLANNER,
        ],
    )
    graph.add_edge(NODE_PREPARE_PLANNER, NODE_PLANNER)
    graph.add_edge(NODE_PLANNER, NODE_VERIFIER)
    graph.add_conditional_edges(
        NODE_VERIFIER,
        route_after_verifier,
        [NODE_PLANNER, NODE_CODEGEN, NODE_PLAN_FAILED],
    )
    graph.add_edge(NODE_CODEGEN, NODE_EVAL_PYTHON)
    graph.add_conditional_edges(NODE_EVAL_PYTHON, route_after_eval, [NODE_CODEGEN, END])
    graph.add_edge(NODE_PLAN_FAILED, END)
    graph.add_edge(NODE_DIRECT_ANSWER, END)

    return graph.compile(checkpointer=checkpointer)
