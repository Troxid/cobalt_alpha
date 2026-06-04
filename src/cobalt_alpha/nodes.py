from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage

from cobalt_alpha.config import load_prompt
from cobalt_alpha.llm_models import (
    llm_codegen,
    llm_direct,
    llm_planner,
    llm_router,
    llm_verifier,
)
from cobalt_alpha.state import (
    GraphState,
    _to_stale_or_missing,
    apply_code,
    apply_result,
    ingest_user_input,
)
from cobalt_alpha.structured_output import PlanDraft, PlanVerification, RouterDecision
from cobalt_alpha.tools.eval_python import run_python


NODE_INGEST_USER_INPUT = "node_ingest_user_input"


def node_ingest_user_input(state: GraphState) -> GraphState:
    return ingest_user_input(state)


NODE_ROUTER = "node_router"


def node_router(state: GraphState) -> GraphState:
    "Определяет intent запроса и выбирает ветку обработки"
    llm = llm_router().with_structured_output(RouterDecision)
    router_prompt = load_prompt("node_router.md")
    router_input = state.current_user_input
    previous_context = state.task_artifact or (
        f"problem_spec: {state.problem_spec}\n"
        f"answer_artifact: {state.answer_artifact}"
    ).strip()
    if previous_context:
        router_input = (
            f"Previous computation context:\n"
            f"{previous_context}\n\n"
            f"Current request:\n{state.current_user_input}"
        )

    response = llm.invoke(
        [
            SystemMessage(content=router_prompt),
            HumanMessage(content=router_input),
        ]
    )
    decision = RouterDecision.model_validate(response)
    selected_model: Literal["lite", "max"] = (
        "lite" if decision.intent == "direct_answer" else "max"
    )

    return state.model_copy(
        update={
            "complexity_estimation": decision.score,
            "router_intent": decision.intent,
            "selected_model": selected_model,
        }
    )


NODE_DIRECT_ANSWER = "node_direct_answer"


def node_direct_answer(state: GraphState) -> GraphState:
    llm = llm_direct()
    direct_answer_prompt = load_prompt("direct_answer.md")
    direct_answer_input = state.current_user_input
    if state.answer_artifact:
        direct_answer_input = (
            f"Previous answer artifact:\n{state.answer_artifact}\n\n"
            f"Current request:\n{direct_answer_input}"
        )

    response = llm.invoke(
        [
            SystemMessage(content=direct_answer_prompt),
            HumanMessage(content=direct_answer_input),
        ]
    )
    answer = str(response.content)
    return state.model_copy(
        update={
            "result": answer,
            "result_status": "verified",
            "model_response": answer,
            "messages": [*state.messages, {"role": "assistant", "content": answer}],
        }
    )


NODE_PREPARE_PLANNER = "node_prepare_planner"


def node_prepare_planner(state: GraphState) -> GraphState:
    return state.model_copy(
        update={
            "verification_attempts": 0,
            "plan_verification_feedback": "",
            "codegen_attempts": 0,
            "codegen_feedback": "",
        }
    )


NODE_PLANNER = "node_planner"


def node_planner(state: GraphState) -> GraphState:
    "Генерирует черновик problem spec + plan + steps"
    llm = llm_planner().with_structured_output(PlanDraft)
    planner_prompt = load_prompt("planner.md")

    planner_input = state.current_user_input
    if state.router_intent == "refine_computation":
        planner_input = (
            f"Previous task artifacts:\n"
            f"{state.task_artifact or _format_task_artifact(state)}\n\n"
            f"Refinement request:\n{planner_input}"
        )

    if state.plan_verification_feedback:
        planner_input = (
            f"{planner_input}\n\n"
            f"Feedback from verifier (must be fixed): {state.plan_verification_feedback}"
        )

    response = llm.invoke(
        [
            SystemMessage(content=planner_prompt),
            HumanMessage(content=planner_input),
        ]
    )
    draft = PlanDraft.model_validate(response)

    code_status = state.code_status
    result_status = state.result_status
    if draft.plan != state.plan or draft.steps != state.steps:
        code_status = _to_stale_or_missing(state.code_status)
        result_status = _to_stale_or_missing(state.result_status)

    return state.model_copy(
        update={
            "problem_spec": draft.problem_spec,
            "plan": draft.plan,
            "plan_status": "draft",
            "steps": draft.steps,
            "steps_status": "draft",
            "plan_verification_feedback": "",
            "code_status": code_status,
            "result_status": result_status,
        }
    )


NODE_VERIFIER = "node_verifier"


def node_verifier(state: GraphState) -> GraphState:
    "Проверяет корректность плана и шагов"
    llm = llm_verifier().with_structured_output(PlanVerification)
    verifier_prompt = load_prompt("verifier.md")

    verification_input = (
        f"problem_spec:\n{state.problem_spec}\n\n"
        f"plan:\n{state.plan}\n\n"
        f"steps:\n" + "\n".join(f"- {step}" for step in state.steps)
    )

    response = llm.invoke(
        [
            SystemMessage(content=verifier_prompt),
            HumanMessage(content=verification_input),
        ]
    )
    verdict = PlanVerification.model_validate(response)

    if verdict.approved:
        return state.model_copy(
            update={
                "plan_status": "verified",
                "steps_status": "verified",
                "plan_verification_feedback": verdict.feedback,
                "verification_attempts": state.verification_attempts + 1,
            }
        )

    return state.model_copy(
        update={
            "plan_status": _to_stale_or_missing(state.plan_status),
            "steps_status": _to_stale_or_missing(state.steps_status),
            "plan_verification_feedback": verdict.feedback,
            "verification_attempts": state.verification_attempts + 1,
        }
    )


NODE_CODEGEN = "node_codegen"


def node_codegen(state: GraphState) -> GraphState:
    "Генерирует Python-код для вычислений"
    llm = llm_codegen()
    codegen_prompt = load_prompt("codegen.md")

    codegen_input = (
        f"problem_spec:\n{state.problem_spec}\n\n"
        f"plan:\n{state.plan}\n\n"
        f"steps:\n" + "\n".join(f"- {step}" for step in state.steps)
    )
    if state.codegen_feedback:
        codegen_input = (
            f"{codegen_input}\n\n"
            f"Previous code failed. Fix the code using this feedback:\n"
            f"{state.codegen_feedback}"
        )

    response = llm.invoke(
        [
            SystemMessage(content=codegen_prompt),
            HumanMessage(content=codegen_input),
        ]
    )

    generated_code = _sanitize_generated_code(str(response.content))
    return apply_code(state, generated_code)


NODE_EVAL_PYTHON = "node_eval_python"


def node_eval_python(state: GraphState) -> GraphState:
    "Выполняет сгенерированный Python-код"
    eval_result = run_python(state.code, timeout_sec=8.0)

    if eval_result.ok:
        if eval_result.result_repr:
            result_text = eval_result.result_repr
        elif eval_result.stdout.strip():
            result_text = eval_result.stdout.strip()
        else:
            result_text = "Execution completed"

        updated = apply_result(state, result_text)
        return updated.model_copy(
            update={
                "execution_ok": True,
                "execution_error": "",
                "eval_stdout": eval_result.stdout,
                "eval_result_repr": eval_result.result_repr or "",
                "codegen_feedback": "",
                "task_artifact": _format_task_artifact(updated),
            }
        )

    error_text = f"{eval_result.error_type}: {eval_result.error_message}"
    return state.model_copy(
        update={
            "result": error_text,
            "model_response": error_text,
            "code_status": "stale",
            "result_status": "stale",
            "execution_ok": False,
            "execution_error": error_text,
            "eval_stdout": eval_result.stdout,
            "eval_result_repr": eval_result.result_repr or "",
            "codegen_feedback": (
                f"{error_text}\n\n"
                f"stdout:\n{eval_result.stdout}\n\n"
                f"traceback:\n{eval_result.traceback_text or ''}"
            ),
        }
    )


NODE_PLAN_FAILED = "node_plan_failed"


def node_plan_failed(state: GraphState) -> GraphState:
    "Финальный узел, если план не прошел верификацию"
    message = (
        "Plan verification failed after max attempts. "
        f"Last feedback: {state.plan_verification_feedback}"
    )
    return state.model_copy(
        update={
            "result": message,
            "model_response": message,
            "result_status": "verified",
            "execution_ok": False,
            "execution_error": message,
        }
    )


def _sanitize_generated_code(raw_code: str) -> str:
    code = raw_code.strip()

    if "```" in code:
        chunks = code.split("```")
        if len(chunks) >= 3:
            code = chunks[1].strip()
            if code.startswith("python\n"):
                code = code[len("python\n") :]

    if "<unused" in code:
        code = code.split("<unused", maxsplit=1)[0].rstrip()

    return code


def _format_task_artifact(state: GraphState) -> str:
    return (
        f"problem_spec:\n{state.problem_spec}\n\n"
        f"plan:\n{state.plan}\n\n"
        f"steps:\n" + "\n".join(f"- {step}" for step in state.steps) + "\n\n"
        f"last_result:\n{state.result or state.answer_artifact}"
    )
