from typing import Literal

from pydantic import BaseModel
from pydantic.fields import Field

from cobalt_alpha.structured_output import RouterIntent

ArtifactStatus = Literal["missing", "draft", "verified", "stale"]


class GraphState(BaseModel):
    current_user_input: str = Field("", description="Текущее сообщение пользователя")
    messages: list[dict[str, str]] = Field(default_factory=list)
    router_intent: RouterIntent = "direct_answer"
    complexity_estimation: float = 0.0
    selected_model: Literal["lite", "max"] = "lite"
    model_response: str = ""
    answer_artifact: str = ""
    task_artifact: str = ""

    problem_spec: str = ""

    plan: str = ""
    plan_status: ArtifactStatus = "missing"

    steps: list[str] = Field(default_factory=list)
    steps_status: ArtifactStatus = "missing"

    plan_verification_feedback: str = ""
    verification_attempts: int = 0
    max_verification_attempts: int = 2

    code: str = ""
    code_status: ArtifactStatus = "missing"
    codegen_attempts: int = 0
    max_codegen_attempts: int = 3
    codegen_feedback: str = ""

    result: str = ""
    result_status: ArtifactStatus = "missing"

    execution_ok: bool | None = None
    execution_error: str = ""
    eval_stdout: str = ""
    eval_result_repr: str = ""


def _to_stale_or_missing(status: ArtifactStatus) -> ArtifactStatus:
    if status == "missing":
        return "missing"
    return "stale"


def _append_message(
    messages: list[dict[str, str]], role: str, content: str
) -> list[dict[str, str]]:
    if messages and messages[-1] == {"role": role, "content": content}:
        return messages
    return [*messages, {"role": role, "content": content}]


def ingest_user_input(state: GraphState) -> GraphState:
    user_input = state.current_user_input.strip()
    if not user_input:
        return state

    input_changed = bool(state.messages) and state.messages[-1] != {
        "role": "user",
        "content": user_input,
    }
    update: dict[str, object] = {
        "current_user_input": user_input,
        "messages": _append_message(state.messages, "user", user_input),
    }

    if input_changed:
        update |= {
            "plan_status": _to_stale_or_missing(state.plan_status),
            "steps_status": _to_stale_or_missing(state.steps_status),
            "code_status": _to_stale_or_missing(state.code_status),
            "result_status": _to_stale_or_missing(state.result_status),
            "verification_attempts": 0,
            "plan_verification_feedback": "",
            "codegen_attempts": 0,
            "codegen_feedback": "",
            "execution_error": "",
        }

    return state.model_copy(update=update)


def apply_plan(state: GraphState, plan: str) -> GraphState:
    if plan == state.plan and state.plan_status == "verified":
        return state

    if plan != state.plan:
        return state.model_copy(
            update={
                "plan": plan,
                "plan_status": "verified",
                "steps_status": _to_stale_or_missing(state.steps_status),
                "code_status": _to_stale_or_missing(state.code_status),
                "result_status": _to_stale_or_missing(state.result_status),
            }
        )

    return state.model_copy(
        update={
            "plan": plan,
            "plan_status": "verified",
        }
    )


def apply_steps(state: GraphState, steps: list[str]) -> GraphState:
    if steps == state.steps and state.steps_status == "verified":
        return state

    if steps != state.steps:
        return state.model_copy(
            update={
                "steps": steps,
                "steps_status": "verified",
                "code_status": _to_stale_or_missing(state.code_status),
                "result_status": _to_stale_or_missing(state.result_status),
            }
        )

    return state.model_copy(
        update={
            "steps": steps,
            "steps_status": "verified",
        }
    )


def apply_code(state: GraphState, code: str) -> GraphState:
    if code == state.code and state.code_status == "verified":
        return state

    if code != state.code:
        return state.model_copy(
            update={
                "code": code,
                "code_status": "verified",
                "result_status": _to_stale_or_missing(state.result_status),
                "codegen_attempts": state.codegen_attempts + 1,
            }
        )

    return state.model_copy(
        update={
            "code": code,
            "code_status": "verified",
            "codegen_attempts": state.codegen_attempts + 1,
        }
    )


def apply_result(state: GraphState, result: str) -> GraphState:
    return state.model_copy(
        update={
            "result": result,
            "result_status": "verified",
            "model_response": result,
            "answer_artifact": result,
            "messages": _append_message(state.messages, "assistant", result),
        }
    )
