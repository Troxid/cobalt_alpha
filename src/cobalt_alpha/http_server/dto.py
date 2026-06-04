from pydantic import BaseModel, Field

from cobalt_alpha.state import GraphState


class InvokeGraphRequestDTO(BaseModel):
    user_input: str = Field(min_length=1, description="Входной запрос пользователя")


class GraphStateResponseDTO(BaseModel):
    current_user_input: str
    complexity_estimation: float
    selected_model: str
    model_response: str


def graph_state_to_dto(state: GraphState) -> GraphStateResponseDTO:
    return GraphStateResponseDTO(
        current_user_input=state.current_user_input,
        complexity_estimation=state.complexity_estimation,
        selected_model=state.selected_model,
        model_response=state.model_response,
    )
