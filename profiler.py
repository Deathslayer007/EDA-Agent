"""
Autonomous EDA Engine (Deep Path).
Question-Directed Hierarchical Investigation via LangGraph:
Decomposes complex or investigative questions into targeted sub-hypotheses,
executes code & visual evidence for each across stateful nodes in LangGraph,
and synthesizes a direct-answer executive briefing.
"""
from concurrent.futures import ThreadPoolExecutor
import contextlib
import io
import os
import re
import threading
import warnings
from typing import Optional, TypedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import START, END, StateGraph

load_dotenv(override=True)
if os.getenv("OPENROUTER") and not os.getenv("OPENROUTER_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER")
warnings.filterwarnings("ignore", message=".*fixed sampling defaults.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------- Models & Config ----------
FAST_MODEL = os.getenv("FAST_MODEL", "nvidia/nemotron-3.5-lightning:free")
if FAST_MODEL.startswith("openrouter:"):
    FAST_MODEL = FAST_MODEL[len("openrouter:"):]

REASONING_MODEL = os.getenv("REASONING_MODEL", os.getenv("EDA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"))
if REASONING_MODEL.startswith("openrouter:"):
    REASONING_MODEL = REASONING_MODEL[len("openrouter:"):]

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Lock for thread-safe Matplotlib rendering
matplotlib_lock = threading.Lock()

api_key = os.getenv("OPENROUTER") or os.getenv("OPENROUTER_API_KEY")

fast_llm = ChatOpenRouter(
    model=FAST_MODEL,
    api_key=api_key,
    temperature=0.1,
    max_retries=1,
    request_timeout=35000,
)

reasoning_llm = ChatOpenRouter(
    model=REASONING_MODEL,
    api_key=api_key,
    temperature=0.1,
    max_retries=1,
    request_timeout=45000,
)

# Safe built-in functions allowed inside in-memory exec()
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


# ---------- Helpers & Text Sanitization ----------
def strip_fences(text: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


def fix_numbers(text: str) -> str:
    text = re.sub(r"\b[Nn]\s*=\s*(\d[\d,]*(?:\.\d+)?)", r"n = \1", text)
    return re.sub(r"(\bn = \d[\d,]*)\.0+(?!\d)", r"\1", text)


def fix_pct(text: str) -> str:
    return re.sub(r"(\d+(?:\.\d+)?)\s*%", lambda m: f"{float(m.group(1)):.1f}%", text)


def sanitize_text(text: str) -> str:
    return fix_numbers(fix_pct(text.strip()))


# ---------- Prompts (Dynamic & User-Directed) ----------
DECOMPOSER_SYSTEM = """Decompose the user's analytical question into 3–4 sequential, empirical sub-questions.

RULES:
- Each sub-question must directly advance answering the user's specific inquiry.
- Reference actual column names from the dataset summary.
- Each must be answerable with one pandas calculation + one matplotlib/seaborn chart.
- Always require group sample sizes (n = …) alongside rates or averages.
- Tailor the decomposition to the user's intent:
  • Specific hypothesis → progressive breakdown: base rates, covariate interactions, thresholds, edge cases.
  • Exploratory question → top predictors, interaction effects, notable subgroups.
  • Distribution/anomaly question → shape, extreme-value profiling, underlying drivers.
"""

CODER_SYSTEM = """Write concise Python code to answer the given sub-question.

ENVIRONMENT (already available):
- `df`: pandas DataFrame | `pd`, `np`, `plt`, `sns`: imported | `CHART_PATH`: save path for the chart.
- Save exactly ONE chart: `plt.savefig(CHART_PATH, bbox_inches='tight')`. No `plt.show()`.

REQUIREMENTS:
- Always `print()` group sample sizes (n = …) alongside any rates, averages, or percentages.
- Use ONLY existing columns from the dataset summary. Map domain values to their column (e.g. Cherbourg/Southampton -> 'Embarked' values 'C'/'S').
- Print human-readable summary tables or key numbers.
- Return ONLY executable Python code.
"""

REPORTER_SYSTEM = """Write a Markdown executive briefing that directly answers the user's question using the empirical evidence provided.

STRUCTURE:
1. **Lead with the answer.** First 1–2 sentences: definitive answer + primary numbers and sample sizes (n = …).
2. **Evidence sections.** Use descriptive `###` headings tailored to the findings (not generic labels). Quote exact numbers, percentages, and sample sizes. Cite step charts where relevant (e.g. `[Step 1 Chart]`).
3. **Nuances.** Note surprising subgroups, confounders, or edge cases that qualify the answer.
4. **Next steps.** 2–3 concrete, actionable recommendations stemming from this analysis.

Keep headings at `###` level. No filler, no boilerplate.
"""


def _find_best_column(columns: list[str], text: str, target: str) -> str:
    """Intelligently matches an analytical question to the most relevant DataFrame column using domain synonyms and fuzzy stemming."""
    text_lower = text.lower()
    candidates = [c for c in columns if c.lower() != target.lower()]
    if not candidates:
        return columns[0] if columns else target

    # 1. Exact column name match (case-insensitive)
    for c in candidates:
        if c.lower() in text_lower:
            return c

    # 2. Known domain synonym mappings
    SYNONYM_MAP = {
        ("embark", "port", "cherbourg", "southampton", "queenstown", "dock", "board"): ["Embarked", "embarked", "Port", "port"],
        ("class", "pclass", "ticket class", "cabin class", "tier", "luxury"): ["Pclass", "pclass", "Class", "class"],
        ("sex", "gender", "male", "female", "man", "men", "woman", "women", "mother", "father", "boy", "girl"): ["Sex", "sex", "Gender", "gender"],
        ("age", "young", "old", "child", "children", "elderly", "adult", "infant", "baby", "kid"): ["Age", "age"],
        ("fare", "price", "ticket price", "paid", "cost", "expensive", "cheap"): ["Fare", "fare"],
        ("family", "relative", "relatives", "sibsp", "parch", "parent", "parents", "sibling", "siblings", "solo", "alone"): ["SibSp", "Parch", "FamilySize", "family_size"],
    }
    for keywords, target_cols in SYNONYM_MAP.items():
        if any(k in text_lower for k in keywords):
            for tc in target_cols:
                for c in candidates:
                    if c.lower() == tc.lower():
                        return c

    # 3. Stem / prefix match (first 4+ characters of column name)
    for c in candidates:
        stem = c.lower()[:max(4, len(c) - 1)]
        if len(stem) >= 3 and stem in text_lower:
            return c

    return candidates[0]


def decompose_agenda(goal: str, target: str, summary_str: str, max_steps: int = 4) -> tuple[list[str], Optional[str]]:
    """Generates a targeted 3-4 step investigation agenda dynamically tailored to the user's inquiry."""
    cols = [line.split(":")[0].replace("-", "").strip() for line in summary_str.splitlines() if ":" in line and "-" in line]
    best_col = _find_best_column(cols, goal, target)

    fallback_agenda = [
        f"What is the baseline {target} distribution across categories of {best_col}? Show rates and group sizes (n = ...).",
        f"How do other key confounding variables interact with {best_col} in affecting {target}?",
        f"Are there specific high-risk or high-survival subgroups within {best_col}?",
        f"What are the combined effects and key patterns involving {best_col} and {target}?",
    ][:max_steps]

    user_prompt = (
        f"User Inquiry / Goal: {goal}\n"
        f"Target Column: {target}\n\n"
        f"Dataset Summary:\n{summary_str[:800]}\n\n"
        f"Generate {max_steps} specific, complementary sub-questions that systematically answer the user's inquiry.\n"
        "Output ONLY numbered lines (1., 2., 3., 4.). No markdown fences or conversational filler."
    )
    def _call_decomposer(m):
        return m.invoke([
            ("system", DECOMPOSER_SYSTEM),
            ("user", user_prompt)
        ]).content

    last_error = None
    for model in [fast_llm, reasoning_llm]:
        ex = ThreadPoolExecutor(max_workers=1)
        try:
            fut = ex.submit(_call_decomposer, model)
            res = fut.result(timeout=35.0)
            if isinstance(res, list):
                res = "".join(b if isinstance(b, str) else b.get("text", "") for b in res)

            agenda = [
                re.sub(r"^\d+[\.\)]\s*", "", line).strip()
                for line in res.splitlines()
                if re.match(r"^\d+[\.\)]", line.strip()) and len(line.strip()) > 8
            ][:max_steps]
            if len(agenda) >= 2:
                return agenda, None
        except Exception as e:
            last_error = str(e)
            continue
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

    return fallback_agenda, last_error


def execute_step_in_memory(code: str, df: pd.DataFrame, chart_path: str) -> tuple[str, Optional[str]]:
    """Executes step code in RAM under a thread lock to ensure safe Matplotlib rendering."""
    with matplotlib_lock:
        plt.close("all")
        stdout_capture = io.StringIO()
        local_env = {
            "__builtins__": SAFE_BUILTINS,
            "df": df.copy(),
            "pd": pd,
            "np": np,
            "plt": plt,
            "sns": sns,
            "CHART_PATH": chart_path,
        }
        err = None
        try:
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stdout_capture):
                exec(code, local_env)
        except Exception as e:
            err = str(e)
        finally:
            plt.close("all")

        output = stdout_capture.getvalue().strip()
        return output, err


# ---------- LangGraph State & Node Definitions ----------
class EDAState(TypedDict):
    goal: str
    target: str
    summary: str
    df: pd.DataFrame
    max_steps: int
    agenda: list[str]
    current_idx: int
    question: str
    code: str
    output: str
    error: Optional[str]
    retries: int
    step_charts: list[str]
    findings: list[dict]
    report: str


def profiler_node(state: EDAState) -> dict:
    """LangGraph Node 1: Decomposes the user's question into an empirical agenda."""
    steps = state.get("max_steps", 4)
    agenda, _ = decompose_agenda(state["goal"], state["target"], state["summary"], max_steps=steps)
    return {
        "agenda": agenda,
        "current_idx": 0,
        "step_charts": [],
        "findings": [],
    }


def planner_node(state: EDAState) -> dict:
    """LangGraph Node 2: Selects the next empirical sub-question to investigate."""
    idx = state["current_idx"]
    q = state["agenda"][idx]
    return {
        "current_idx": idx,
        "question": q,
        "retries": 0,
        "error": None,
    }


def _call_coder_llm(prompt: str) -> str:
    res = fast_llm.invoke([
        ("system", CODER_SYSTEM),
        ("user", prompt)
    ]).content
    if isinstance(res, list):
        return "".join(b if isinstance(b, str) else b.get("text", "") for b in res)
    return str(res)


def coder_node(state: EDAState) -> dict:
    """LangGraph Node 3: Generates Python analysis and plotting code."""
    q = state["question"]
    target = state["target"]
    summary = state["summary"]

    if state.get("error"):
        prompt = (
            f"Sub-question: {q}\n"
            f"Previous code that failed:\n```python\n{state['code']}\n```\n"
            f"Error message:\n{state['error']}\n"
            "Fix the Python code. Return ONLY executable Python code."
        )
    else:
        prompt = (
            f"Sub-Question to investigate: {q}\n\n"
            f"Primary Target: {target}\n"
            f"Dataset Summary:\n{summary[:1000]}\n\n"
            "Write concise Python code to analyze the data, print numerical results with group sizes (n = ...), and save a relevant chart to CHART_PATH."
        )

    ex = ThreadPoolExecutor(max_workers=1)
    try:
        fut = ex.submit(_call_coder_llm, prompt)
        res = fut.result(timeout=30.0)
        code = strip_fences(res)
    except Exception:
        # Robust question-aware fallback code using domain column matching
        eval_col = _find_best_column(list(state["df"].columns), q, target)
        code = (
            f"counts = df.groupby('{eval_col}')['{target}'].count()\n"
            f"rates = df.groupby('{eval_col}')['{target}'].value_counts(normalize=True).unstack()\n"
            "for idx_val in rates.index:\n"
            "    n_val = counts.get(idx_val, 0)\n"
            "    row_rates = {k: f'{v:.1%}' for k, v in rates.loc[idx_val].items()}\n"
            "    print(f'Group [{idx_val}] (n = {n_val}): {row_rates}')\n"
            "plt.figure(figsize=(7, 4))\n"
            f"sns.countplot(data=df, x='{eval_col}', hue='{target}')\n"
            f"plt.title({repr(q[:45])})\n"
            "plt.savefig(CHART_PATH, bbox_inches='tight')\n"
        )
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    new_retries = state.get("retries", 0) + (1 if state.get("error") else 0)
    return {"code": code, "retries": new_retries}


def executor_node(state: EDAState) -> dict:
    """LangGraph Node 4: Executes Python analysis script in RAM."""
    step_num = state["current_idx"] + 1
    chart_path = os.path.join(OUT_DIR, f"step_{step_num}.png")
    output, err = execute_step_in_memory(state["code"], state["df"], chart_path)

    # If retry also failed, execute safe fallback so step has empirical output
    if err and state.get("retries", 0) >= 1:
        q = state["question"]
        target = state["target"]
        df = state["df"]
        eval_col = _find_best_column(list(df.columns), q, target)
        fallback_code = (
            f"counts = df.groupby('{eval_col}')['{target}'].count()\n"
            f"rates = df.groupby('{eval_col}')['{target}'].value_counts(normalize=True).unstack()\n"
            "for idx_val in rates.index:\n"
            "    n_val = counts.get(idx_val, 0)\n"
            "    row_rates = {k: f'{v:.1%}' for k, v in rates.loc[idx_val].items()}\n"
            "    print(f'Group [{idx_val}] (n = {n_val}): {row_rates}')\n"
            "plt.figure(figsize=(7, 4))\n"
            f"sns.countplot(data=df, x='{eval_col}', hue='{target}')\n"
            f"plt.title({repr(q[:45])})\n"
            "plt.savefig(CHART_PATH, bbox_inches='tight')\n"
        )
        fb_out, fb_err = execute_step_in_memory(fallback_code, df, chart_path)
        if not fb_err:
            output, err = fb_out, None

    return {
        "output": output,
        "error": err,
    }


def recorder_node(state: EDAState) -> dict:
    """LangGraph Node 5: Records empirical findings, chart paths, and advances index."""
    idx = state["current_idx"]
    step_num = idx + 1
    chart_path = os.path.join(OUT_DIR, f"step_{step_num}.png")
    actual_chart = chart_path if os.path.exists(chart_path) else ""

    first_finding_line = ""
    if state.get("output"):
        for line in state["output"].splitlines():
            line_str = line.strip()
            if line_str.startswith("Group ") or (any(k in line_str.lower() for k in ["%", "rate", "diff"]) and not line_str.lower().startswith("sample size:")):
                first_finding_line = line_str
                break
        if not first_finding_line:
            first_finding_line = state["output"].splitlines()[0].strip()

    new_finding = {
        "step": step_num,
        "question": state["question"],
        "finding": first_finding_line or "Observed empirical distribution across groups.",
        "output": state.get("output", ""),
        "chart": actual_chart
    }

    new_charts = list(state.get("step_charts", []))
    if actual_chart and actual_chart not in new_charts:
        new_charts.append(actual_chart)

    return {
        "findings": state.get("findings", []) + [new_finding],
        "step_charts": new_charts,
        "current_idx": idx + 1,
    }


def reporter_node(state: EDAState) -> dict:
    """LangGraph Node 6: Synthesizes direct-answer executive briefing."""
    evidence_blocks = []
    for r in state.get("findings", []):
        chart_tag = f"Chart: {os.path.basename(r['chart'])}\n" if r.get("chart") else ""
        evidence_blocks.append(
            f"Step {r['step']} Investigation: *{r['question']}*\n"
            f"{chart_tag}"
            f"Empirical Output:\n{r.get('output', r.get('finding', ''))}\n"
        )
    all_evidence = "\n---\n".join(evidence_blocks)

    user_prompt = (
        f"USER INQUIRY / GOAL: {state['goal']}\n"
        f"TARGET COLUMN: {state['target']}\n\n"
        f"EMPIRICAL RESULTS FROM INVESTIGATION STEPS:\n{all_evidence}\n\n"
        "Directly answer the user's specific inquiry using these empirical findings. "
        "State the bottom-line answer up front and structure the briefing logically using clean, content-specific ### headings."
    )

    def _call_reporter(m):
        return m.invoke([
            ("system", REPORTER_SYSTEM),
            ("user", user_prompt)
        ]).content

    report = ""
    last_reporter_err = None
    for model in [fast_llm, reasoning_llm]:
        ex = ThreadPoolExecutor(max_workers=1)
        try:
            fut = ex.submit(_call_reporter, model)
            content = fut.result(timeout=35.0)
            if isinstance(content, list):
                content = "".join(b if isinstance(b, str) else b.get("text", "") for b in content)
            if content and len(content.strip()) > 100:
                report = content
                break
        except Exception as e:
            last_reporter_err = str(e)
            continue
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

    if not report:
        err_detail = f" (`{last_reporter_err}`)" if last_reporter_err else ""
        report = (
            f"### Investigation Findings: {state['goal']}\n\n"
            f"> ⚠️ **Notice**: The final executive briefing could not be generated by the AI model{err_detail}. "
            f"Below are the empirical findings and step charts computed directly from the dataset.\n\n"
            f"### Empirical Evidence from Investigation Steps\n\n{all_evidence}\n\n"
            f"### Recommended Actions\n"
            f"- Review the step charts below to inspect the empirical distributions directly\n"
            f"- Verify your OpenRouter API key quota or model status if this notice persists"
        )

    report = sanitize_text(report)
    report = re.sub(r"^#{1,2}\s+", "### ", report, flags=re.MULTILINE)

    report_file = os.path.join(OUT_DIR, "report.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    return {"report": report}


# ---------- Conditional Routing Functions ----------
def route_after_executor(state: EDAState) -> str:
    """If execution failed and retries remain, retry code generation; else record."""
    if state.get("error") and state.get("retries", 0) < 1:
        return "coder"
    return "recorder"


def route_after_recorder(state: EDAState) -> str:
    """If more sub-questions remain in the agenda, continue to planner; else synthesize."""
    if state["current_idx"] < len(state.get("agenda", [])):
        return "planner"
    return "reporter"


# ---------- Graph Construction ----------
def build_eda_graph():
    """Builds the stateful LangGraph Multi-Agent EDA Graph."""
    workflow = StateGraph(EDAState)
    workflow.add_node("profiler", profiler_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("recorder", recorder_node)
    workflow.add_node("reporter", reporter_node)

    workflow.add_edge(START, "profiler")
    workflow.add_edge("profiler", "planner")
    workflow.add_edge("planner", "coder")
    workflow.add_edge("coder", "executor")

    workflow.add_conditional_edges(
        "executor",
        route_after_executor,
        {"coder": "coder", "recorder": "recorder"}
    )
    workflow.add_conditional_edges(
        "recorder",
        route_after_recorder,
        {"planner": "planner", "reporter": "reporter"}
    )
    workflow.add_edge("reporter", END)

    return workflow.compile()


# ---------- Public Streaming Generator for Streamlit ----------
def run_autonomous_eda_stream(
    df: pd.DataFrame,
    goal: str = "Explore key patterns and drivers",
    target: Optional[str] = None,
    max_steps: int = 4
):
    """
    Generator yielding live execution events from the LangGraph workflow to Streamlit.
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    # Determine target column
    if not target or target not in df.columns:
        candidates = ["survived", "churn", "target", "label", "outcome", "y", "fraud"]
        found = next((c for c in df.columns if c.lower() in candidates), None)
        target = found if found else str(df.columns[-1])

    # Build dataset summary
    lines = [f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns", f"Target Column: {target}", "Columns:"]
    for c in df.columns:
        lines.append(f"- {c}: dtype={df[c].dtype}, missing={df[c].isna().mean():.1%}, unique={df[c].nunique()}")
    summary_str = "\n".join(lines)

    app = build_eda_graph()
    initial_state: EDAState = {
        "goal": goal,
        "target": target,
        "summary": summary_str,
        "df": df,
        "max_steps": max_steps,
        "agenda": [],
        "current_idx": 0,
        "question": "",
        "code": "",
        "output": "",
        "error": None,
        "retries": 0,
        "step_charts": [],
        "findings": [],
        "report": "",
    }

    yield {
        "node": "profiler",
        "preview": "Formulating hypothesis agenda via LangGraph Profiler…",
        "progress": 0.08
    }

    total_steps = max_steps
    accumulated_findings = []
    accumulated_charts = []
    final_report = ""

    for update in app.stream(initial_state, config={"recursion_limit": 50}, stream_mode="updates"):
        for node, delta in update.items():
            delta = delta or {}

            if delta.get("agenda"):
                total_steps = len(delta["agenda"])
                yield {
                    "node": "profiler",
                    "agenda": delta["agenda"],
                    "preview": f"{total_steps} sub-questions formulated",
                    "progress": 0.20
                }

            elif node == "planner":
                idx = delta.get("current_idx", 0)
                q = delta.get("question", "")
                progress_val = 0.20 + (0.55 * (idx / max(1, total_steps)))
                yield {
                    "node": "planner",
                    "step": idx,
                    "total_steps": total_steps,
                    "question": q,
                    "preview": f"Investigating: {q}",
                    "progress": progress_val
                }

            elif node == "coder":
                yield {
                    "node": "coder",
                    "preview": "Generating Python analysis & visualization script…",
                    "progress": None
                }

            elif node == "executor":
                out = delta.get("output", "")
                err = delta.get("error")
                preview_text = out[:120].replace("\n", " ") if out else (err or "Code executed")
                yield {
                    "node": "executor",
                    "preview": preview_text,
                    "progress": None
                }

            elif node == "recorder":
                idx = delta.get("current_idx", 0)
                if delta.get("findings"):
                    accumulated_findings = delta["findings"]
                if delta.get("step_charts"):
                    accumulated_charts = delta["step_charts"]

                step_chart = os.path.join(OUT_DIR, f"step_{idx}.png")
                chart_file = step_chart if os.path.exists(step_chart) else None

                progress_val = 0.20 + (0.60 * (idx / max(1, total_steps)))
                yield {
                    "node": "recorder",
                    "step": idx - 1,
                    "total_steps": total_steps,
                    "chart_path": chart_file,
                    "preview": f"Recorded findings for step {idx}/{total_steps}",
                    "progress": progress_val
                }

            elif node == "reporter":
                final_report = delta.get("report", "")
                yield {
                    "node": "reporter",
                    "report": final_report,
                    "preview": "Synthesized executive briefing",
                    "progress": 0.95
                }

    # Final completion event
    yield {
        "node": "complete",
        "report": final_report,
        "target": target,
        "insights": accumulated_findings,
        "step_charts": accumulated_charts,
        "preview": "LangGraph Autonomous EDA complete!",
        "progress": 1.0
    }
