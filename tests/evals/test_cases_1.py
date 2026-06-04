import pytest

pytest.importorskip("deepeval")

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from cobalt_alpha.config import load_eval_prompt
from cobalt_alpha.graph import build_graph
from cobalt_alpha.llm_eval_models import openrouter_eval_model
from cobalt_alpha.state import GraphState

RC_CIRCUIT_PROMPT = load_eval_prompt("rc_circuit_input.md")
EXPECTED_RC_CIRCUIT_OUTPUT = load_eval_prompt("rc_circuit_expected.md")
OHM_LAW_PROMPT = load_eval_prompt("ohm_law_input.md")
EXPECTED_OHM_LAW_OUTPUT = load_eval_prompt("ohm_law_expected.md")

EVAL_MODEL = openrouter_eval_model("qwen/qwen3.6-plus")


def run_agent(user_input: str) -> str:
    graph = build_graph()
    result = graph.invoke(GraphState(current_user_input=user_input))
    state = GraphState.model_validate(result)
    return state.model_response


def test_ohm_law_current_calculation_deepeval() -> None:
    actual_output = run_agent(OHM_LAW_PROMPT)

    correctness_metric = GEval(
        name="Ohm law current calculation correctness",
        criteria=(
            "Determine whether the actual output correctly computes current using "
            "Ohm's law. It must identify I = U / R and return 0.05 A or an "
            "equivalent value such as 50 mA."
        ),
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=0.8,
        model=EVAL_MODEL,
    )

    test_case = LLMTestCase(
        input=OHM_LAW_PROMPT,
        actual_output=actual_output,
        expected_output=EXPECTED_OHM_LAW_OUTPUT,
    )

    assert_test(test_case, [correctness_metric])


def test_rc_circuit_calculation_deepeval() -> None:
    actual_output = run_agent(RC_CIRCUIT_PROMPT)

    correctness_metric = GEval(
        name="RC circuit calculation correctness",
        criteria=(
            "Determine whether the actual output correctly solves the RC charging "
            "problem. It must use the RC charging equation, convert units correctly, "
            "and return tau, capacitor voltages at 1, 3, and 5 seconds, and time to "
            "90% charge with values close to the expected output."
        ),
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=0.75,
        model=EVAL_MODEL,
    )

    test_case = LLMTestCase(
        input=RC_CIRCUIT_PROMPT,
        actual_output=actual_output,
        expected_output=EXPECTED_RC_CIRCUIT_OUTPUT,
    )

    assert_test(test_case, [correctness_metric])
