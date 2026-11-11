"""
Quick Query Engine (Fast Path).
Executes single-shot analytical and visualization queries in ~1-2 seconds
with security sandboxing (restricted builtins and execution timeout).
"""
import concurrent.futures
import contextlib
import io
import os
import re
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter

load_dotenv(override=True)
if os.getenv("OPENROUTER") and not os.getenv("OPENROUTER_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER")

MODEL = os.getenv("FAST_MODEL", "nvidia/nemotron-3.5-lightning:free")
if MODEL.startswith("openrouter:"):
    MODEL = MODEL[len("openrouter:"):]

EXEC_TIMEOUT_SECONDS = 15

# Safe built-in functions allowed inside exec() sandbox
SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "filter": filter, "float": float, "int": int,
    "isinstance": isinstance, "issubclass": issubclass, "iter": iter,
    "len": len, "list": list, "map": map, "max": max, "min": min,
    "next": next, "print": print, "range": range, "reversed": reversed,
    "round": round, "set": set, "slice": slice, "sorted": sorted,
    "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    "True": True, "False": False, "None": None,
}

# Forbidden keywords for AST / text sanity check
FORBIDDEN_PATTERNS = [
    r"\bimport\s+(?:os|sys|subprocess|shutil|socket|urllib|requests|pathlib)\b",
    r"\bfrom\s+(?:os|sys|subprocess|shutil|socket|urllib|requests|pathlib)\b",
    r"\bopen\s*\(",
    r"\b__import__\b",
    r"\beval\s*\(",
    r"\bexec\s*\(",
]

llm = ChatOpenRouter(
    model=MODEL,
    api_key=os.getenv("OPENROUTER") or os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
    max_retries=3,
)


def strip_fences(text: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


def validate_code_safety(code: str) -> tuple[bool, str]:
    """Scans code for forbidden OS / subprocess / filesystem calls."""
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, code):
            return False, f"Security Violation: Forbidden construct detected ({pat})"
    return True, ""


QUICK_CODER_SYSTEM = """You are an expert Python data analyst. Write concise code to answer the user's query.

ENVIRONMENT (already available — do NOT re-import or re-read):
- `df`: pandas DataFrame with the dataset
- `pd`, `np`, `plt`, `sns`: already imported
- No disk I/O. No `plt.show()` (figure is captured automatically).
- Charts: use `plt.figure(figsize=(7, 4))` with a descriptive title and axis labels.

OUTPUT FORMAT — choose one:
1. **Table result** → assign to `result_df` (rendered as an interactive table).
2. **Single metric / text answer** → `print()` a concise markdown sentence with bold key numbers.
3. **Chart** → build with matplotlib/seaborn; no `plt.show()`.
Never print raw DataFrame text; use `result_df` or formatted markdown instead.

Return ONLY executable Python code.
"""


def generate_quick_code(query: str, schema_summary: str, columns: list[str], chat_history: list = None) -> str:
    history_context = ""
    if chat_history:
        recent = chat_history[-3:]
        history_lines = [f"{m.get('role', 'user')}: {m.get('text', '')}" for m in recent if m.get('text')]
        if history_lines:
            history_context = f"Recent Conversation Context:\n" + "\n".join(history_lines) + "\n\n"

    prompt = (
        f"Dataset Columns: {', '.join(columns)}\n\n"
        f"Dataset Summary:\n{schema_summary[:1000]}\n\n"
        f"{history_context}"
        f"User Question: {query}\n\n"
        "Write concise Python code to analyze the data and generate the requested chart/statistics."
    )
    res = llm.invoke([
        ("system", QUICK_CODER_SYSTEM),
        ("user", prompt)
    ]).content
    if isinstance(res, list):
        res = "".join(b if isinstance(b, str) else b.get("text", "") for b in res)
    return strip_fences(res)


def _exec_worker(code: str, local_env: dict, stdout_capture: io.StringIO):
    with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stdout_capture):
        exec(code, local_env)


def execute_quick_code(code: str, df: pd.DataFrame) -> dict:
    """Executes the Python snippet in a sandboxed environment with restricted builtins and execution timeout."""
    plt.close("all")  # reset any prior figure

    # 1. Security Check
    is_safe, sec_reason = validate_code_safety(code)
    if not is_safe:
        return {
            "text": "",
            "table": None,
            "figure": None,
            "code": code,
            "error": sec_reason
        }

    stdout_capture = io.StringIO()
    local_env = {
        "__builtins__": SAFE_BUILTINS,
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns
    }
    error_msg = None
    fig = None

    # 2. Execution with Timeout
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_exec_worker, code, local_env, stdout_capture)
        try:
            future.result(timeout=EXEC_TIMEOUT_SECONDS)
            if plt.get_fignums():
                fig = plt.gcf()
        except concurrent.futures.TimeoutError:
            error_msg = f"Execution timed out after {EXEC_TIMEOUT_SECONDS}s (possible infinite loop)."
        except Exception as e:
            error_msg = str(e)

    output_text = stdout_capture.getvalue().strip()

    # Capture interactive result table if generated
    result_table = local_env.get("result_df")
    if result_table is not None and not isinstance(result_table, pd.DataFrame):
        try:
            result_table = pd.DataFrame(result_table)
        except Exception:
            result_table = None

    return {
        "text": output_text,
        "table": result_table,
        "figure": fig,
        "code": code,
        "error": error_msg
    }


def run_quick_query(query: str, df: pd.DataFrame, schema_summary: str, chat_history: list = None) -> dict:
    """End-to-end quick query helper: generates code and executes it with sandboxing."""
    try:
        code = generate_quick_code(query, schema_summary, [str(c) for c in df.columns], chat_history=chat_history)
    except Exception as e:
        return {
            "text": "",
            "table": None,
            "figure": None,
            "code": "",
            "error": f"AI model request failed: {e}"
        }

    result = execute_quick_code(code, df)
    
    # If the first attempt errored, attempt one automatic fix
    if result.get("error") and "Security Violation" not in result["error"]:
        fix_prompt = (
            f"Question: {query}\n"
            f"Previous code that failed:\n```python\n{code}\n```\n"
            f"Error message:\n{result['error']}\n"
            "Fix the Python code."
        )
        try:
            fix_code = strip_fences(llm.invoke([
                ("system", QUICK_CODER_SYSTEM),
                ("user", fix_prompt)
            ]).content)
            result = execute_quick_code(fix_code, df)
            result["code"] = fix_code
        except Exception:
            pass

    return result