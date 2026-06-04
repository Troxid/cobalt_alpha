from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import cast

import uvicorn
from fastapi import FastAPI, Request
from langgraph.graph.state import CompiledStateGraph

from cobalt_alpha.graph import build_graph
from cobalt_alpha.http_server.dto import (
    GraphStateResponseDTO,
    InvokeGraphRequestDTO,
    graph_state_to_dto,
)
from cobalt_alpha.setup_phoenix import setup_phoenix
from cobalt_alpha.state import GraphState

GraphCompiled = CompiledStateGraph[GraphState, None, GraphState, GraphState]


def get_graph(request: Request) -> GraphCompiled:
    return cast(GraphCompiled, request.app.state.graph)


@asynccontextmanager
async def app_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    setup_phoenix()
    app.state.graph = build_graph()
    yield


def create_app() -> FastAPI:
    app = FastAPI(lifespan=app_lifespan)

    @app.post("/invoke", response_model=GraphStateResponseDTO)
    def invoke_graph(
        request_dto: InvokeGraphRequestDTO, request: Request
    ) -> GraphStateResponseDTO:
        graph = get_graph(request)
        result = graph.invoke(GraphState(current_user_input=request_dto.user_input))
        graph_state = GraphState.model_validate(result)
        return graph_state_to_dto(graph_state)

    return app


def run_http_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    uvicorn.run(create_app(), host=host, port=port)
