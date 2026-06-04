import ast
import asyncio
import io
import multiprocessing as mp
import traceback
from dataclasses import dataclass
from contextlib import redirect_stdout
from time import perf_counter

from pydantic import BaseModel


class EvalPythonResult(BaseModel):
    ok: bool
    stdout: str = ""
    result_repr: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    traceback_text: str | None = None
    execution_time_ms: int
    timed_out: bool = False


@dataclass
class ExecuteUserCodeResult:
    local_ctx: dict[str, object]
    last_value: object | None


def _execute_user_code(code: str) -> ExecuteUserCodeResult:
    local_ctx: dict[str, object] = {}

    parsed = ast.parse(code, mode="exec")
    body = parsed.body

    if body and isinstance(body[-1], ast.Expr):
        exec_ast = ast.Module(body=body[:-1], type_ignores=[])
        last_expr_ast = ast.Expression(body=body[-1].value)

        exec(compile(exec_ast, "<eval_python>", "exec"), local_ctx, local_ctx)
        last_value = eval(
            compile(last_expr_ast, "<eval_python>", "eval"),
            local_ctx,
            local_ctx,
        )
        return ExecuteUserCodeResult(local_ctx=local_ctx, last_value=last_value)

    exec(compile(parsed, "<eval_python>", "exec"), local_ctx, local_ctx)
    return ExecuteUserCodeResult(local_ctx=local_ctx, last_value=None)


def _worker(code: str, result_queue: mp.Queue) -> None:
    started = perf_counter()
    stdout_buffer = io.StringIO()

    try:
        with redirect_stdout(stdout_buffer):
            exec_result = _execute_user_code(code)

        result_queue.put(
            {
                "ok": True,
                "stdout": stdout_buffer.getvalue(),
                "result_repr": None if exec_result.last_value is None else repr(exec_result.last_value),
                "execution_time_ms": int((perf_counter() - started) * 1000),
                "timed_out": False,
            }
        )
    except Exception as exc:
        result_queue.put(
            {
                "ok": False,
                "stdout": stdout_buffer.getvalue(),
                "result_repr": None,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback_text": traceback.format_exc(),
                "execution_time_ms": int((perf_counter() - started) * 1000),
                "timed_out": False,
            }
        )


def run_python(code: str, timeout_sec: float = 5.0) -> EvalPythonResult:
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    process = ctx.Process(target=_worker, args=(code, queue))

    started = perf_counter()
    process.start()
    process.join(timeout_sec)

    if process.is_alive():
        process.terminate()
        process.join(1)
        return EvalPythonResult(
            ok=False,
            stdout="",
            result_repr=None,
            error_type="ExecutionTimeout",
            error_message=f"Execution exceeded timeout: {timeout_sec} sec",
            execution_time_ms=int((perf_counter() - started) * 1000),
            timed_out=True,
        )

    if queue.empty():
        return EvalPythonResult(
            ok=False,
            stdout="",
            result_repr=None,
            error_type="ExecutionError",
            error_message="Worker exited without returning a result",
            execution_time_ms=int((perf_counter() - started) * 1000),
            timed_out=False,
        )

    payload = queue.get()
    return EvalPythonResult.model_validate(payload)


async def run_python_async(code: str, timeout_sec: float = 5.0) -> EvalPythonResult:
    return await asyncio.to_thread(run_python, code, timeout_sec)
