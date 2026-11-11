"""
Autonomous EDA Engine (Deep Path).
Question-Directed Hierarchical Investigation via LangGraph:
Decomposes complex or investigative questions into targeted sub-hypotheses,
executes code & visual evidence for each, and synthesizes a direct-answer report.
"""
import json
import operator
import os
import re
import subprocess
import sys
import tempfile
import warnings
from typing import Annotated, Literal, Optional, TypedDict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv(override=True)
if os.getenv("OPENROUTER") and not os.getenv("OPENROUTER_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER")
warnings.filterwarnings("ignore", message=".*fixed sampling defaults.*")

# ---------- Model & Config ----------
MODEL = os.getenv("REASONING_MODEL", os.getenv("EDA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"))
if MODEL.startswith("openrouter:"):
    MODEL = MODEL[len("openrouter:"):]

MAX_RETRIES = 2
MAX_REVISIONS = 1
MIN_GROUP_SIZE = 20
EXEC_TIMEOUT = 60
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

llm = ChatOpenRouter(
    model=MODEL,
    api_key=os.getenv("OPENROUTER") or os.getenv("OPENROUTER_API_KEY"),
    temperature=0.1,
    max_retries=3,
)

# ---------- Pydantic Models for Structured Output ----------
class SubQuestionPlan(BaseModel):
    agenda: list[str] = Field(
        description="An ordered list of 3 to 4 specific, empirical sub-questions to investigate the user's goal. "
                    "Include: (1) baseline disparity for the queried factor, (2) controlling for primary confounders, "
                    "(3) subgroup interactions/vulnerability, and (4) multivariate attribution or threshold cliffs."
    )
    target_variable: str = Field(
        description="The primary outcome/target column being analyzed."
    )
    rationale: str = Field(
        description="A concise summary of how these sub-questions directly address the user's inquiry."
    )


class Finding(BaseModel):
    statement: str = Field(description="One sentence: key rates + group sizes written as 'n = 123' + takeaway.")
    strength: int = Field(default=3, description="1 (weak) to 5 (robust, large sample, strong effect).")
    adjusted_for: list[str] = Field(default_factory=list, description="Dataset columns held constant in this finding.")
    variable: str = Field(default="", description="The primary column this finding focuses on.")


class CriticReview(BaseModel):
    verdict: Literal["accept", "revise", "reject"] = Field(
        description="Choose 'accept' if output clearly answers the sub-question with rates and group sizes (n >= 20). "
                    "Choose 'revise' if a minor code fix will yield adequate sample sizes or missing rates. "
                    "Choose 'reject' only if data cannot support the question."
    )
    reason: str = Field(description="Brief justification for the verdict.")
    findings: list[Finding] = Field(default_factory=list, description="Extracted findings with rates and group sizes.")


decomposer_llm = llm.with_structured_output(SubQuestionPlan)
critic_llm = llm.with_structured_output(CriticReview)


# ---------- State Definition ----------
class EDAState(TypedDict):
    data_path: str
    goal: str
    summary: str
    columns: list[str]
    numeric_columns: list[str]
    id_like_columns: list[str]
    outcome: str
    agenda: list[str]
    current_idx: int
    question: str
    code: str
    output: str
    error: Optional[str]
    retries: int
    revisions: int
    verdict: str
    critique: str
    pending_findings: list[dict]
    findings: Annotated[list[dict], operator.add]
    step_charts: Annotated[list[str], operator.add]
    failed: Annotated[list[str], operator.add]
    rejected: Annotated[list[str], operator.add]


# ---------- Helpers & Text Sanitization ----------
def ask(system: str, user: str) -> str:
    content = llm.invoke([("system", system), ("user", user)]).content
    if isinstance(content, list):
        content = "".join(b if isinstance(b, str) else b.get("text", "") for b in content)
    return content


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


# ---------- Node 1: Profiler & Agenda Decomposer ----------
DECOMPOSER_SYSTEM = """You are a Principal Data Scientist planning an exploratory data analysis.
Your job is to decompose the user's analytical goal into 3 to 4 targeted, empirical sub-questions.
The sub-questions must DIRECTLY investigate the user's specific hypothesis or query.

Guidelines:
1. Sub-question 1 (Direct Disparity / Baseline):
   - Measure the exact rates or statistics for the primary factor(s) named in the user query.
2. Sub-question 2 (Confounder Control):
   - Test whether the observed relationship holds after controlling for the dataset's strongest demographic or structural covariates (e.g. sex, age, class, tier).
3. Sub-question 3 (Subgroup Vulnerability / Interactions):
   - Examine how the effect differs across specific sub-populations (e.g. vulnerable groups, high-risk segments).
4. Sub-question 4 (Multivariate Attribution / Mechanism / Thresholds):
   - Check combined factors, continuous non-linear thresholds, or relative risk contributions.

Every sub-question MUST be answerable via a concise pandas/seaborn script on the loaded DataFrame.
"""


def profiler(state: EDAState) -> dict:
    """Reads dataset, constructs summary, and generates the question-directed investigation agenda."""
    df = pd.read_csv(state["data_path"])
    target = state["outcome"]

    # Detect target if not explicitly valid
    if target not in df.columns:
        candidates = ["survived", "churn", "target", "label", "outcome", "y", "fraud"]
        found = next((c for c in df.columns if c.lower() in candidates), None)
        target = found if found else str(df.columns[-1])

    lines = [f"Shape: {df.shape[0]} rows x {df.shape[1]} columns", f"Target Column: {target}", "Columns:"]
    for c in df.columns:
        lines.append(f"- {c}: dtype={df[c].dtype}, missing={df[c].isna().mean():.1%}, unique={df[c].nunique()}")

    numeric = df.select_dtypes("number")
    n = len(df)
    id_like = [
        str(c) for c in df.columns
        if n and ((pd.api.types.is_integer_dtype(df[c]) and df[c].nunique() / n >= 0.9)
                  or (not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() / n >= 0.5))
    ]

    summary_str = "\n".join(lines)

    # Decompose the user's specific query into a 3-4 step agenda
    user_prompt = (
        f"User Inquiry / Goal: {state['goal']}\n"
        f"Target Column: {target}\n\n"
        f"Dataset Summary:\n{summary_str}\n\n"
        "Generate 3 to 4 specific, complementary sub-questions that systematically answer the user's inquiry."
    )

    agenda = []
    try:
        plan = decomposer_llm.invoke([
            ("system", DECOMPOSER_SYSTEM),
            ("user", user_prompt)
        ])
        agenda = [q.strip() for q in plan.agenda if q.strip()][:4]
    except Exception as e:
        print(f"[decomposer] structured output failed: {e}")
        # Robust fallback agenda directly mentioning the user's goal
        agenda = [
            f"Examine the baseline relationship between {state['goal']} and {target}, showing rates and group sizes.",
            f"Test whether the relationship holds after controlling for primary confounding variables.",
            f"Analyze subgroup interactions and identify which specific segments are most affected.",
            f"Assess combined factors and quantify independent contributions to {target}."
        ]

    if not agenda:
        agenda = [
            f"Analyze key factors related to {state['goal']} against {target}.",
            f"Control for main covariates and evaluate subgroup differences."
        ]

    return {
        "summary": summary_str,
        "columns": [str(c) for c in df.columns],
        "numeric_columns": [str(c) for c in numeric.columns],
        "id_like_columns": id_like,
        "outcome": target,
        "agenda": agenda,
        "current_idx": 0,
        "findings": [],
        "step_charts": [],
        "failed": [],
        "rejected": [],
    }


# ---------- Node 2: Planner (Schedules Current Sub-Question) ----------
def planner(state: EDAState) -> dict:
    idx = state.get("current_idx", 0)
    agenda = state.get("agenda", [])
    if idx < len(agenda):
        current_q = agenda[idx]
    else:
        current_q = f"Synthesizing findings for {state['goal']}"

    return {
        "question": current_q,
        "code": "",
        "output": "",
        "error": None,
        "retries": 0,
        "revisions": 0,
        "verdict": "",
        "critique": "",
        "pending_findings": []
    }


# ---------- Node 3: Coder ----------
CODER_SYSTEM = f"""You are an expert Python data analyst.
Given an empirical sub-question and dataset summary, write concise, robust Python code to answer it.
Environment:
- `df` is already loaded as a pandas DataFrame.
- `pd`, `np`, `plt`, and `sns` are already imported.
- `CHART_PATH` is a predefined string variable. Save exactly ONE visualization to `CHART_PATH` using `plt.savefig(CHART_PATH, bbox_inches='tight')`.
- Do NOT call `plt.show()`.
- Group sizes (n) are MANDATORY: Always compute and `print()` group sizes (n) alongside every percentage, mean, or comparison.
- Make compared groups coarse enough so that compared groups have n >= {MIN_GROUP_SIZE} where possible.
- Print clear, human-readable summary tables or rates with `print()`.
- Return ONLY executable Python code in a single code block.
"""


def coder(state: EDAState) -> dict:
    prompt = (
        f"Sub-Question to investigate: {state['question']}\n\n"
        f"Primary Target: {state['outcome']}\n"
        f"Dataset Summary:\n{state['summary'][:1000]}\n\n"
        f"Write Python code to analyze the data, print the numerical results (with group sizes n), and save a relevant chart to CHART_PATH."
    )
    if state.get("error"):
        prompt += (
            f"\n\nYour previous code failed with this error:\n```\n{state['error']}\n```\n"
            f"Previous code was:\n```python\n{state['code']}\n```\nFix the code."
        )
    elif state.get("critique"):
        prompt += (
            f"\n\nReviewer feedback on previous output:\n{state['critique']}\n"
            f"Previous output was:\n{state['output']}\n"
            "Update the code to resolve the feedback."
        )

    code = strip_fences(ask(CODER_SYSTEM, prompt))
    return {"code": code, "critique": ""}


# ---------- Node 4: Executor ----------
def executor(state: EDAState) -> dict:
    step_num = state["current_idx"] + 1
    chart_path = os.path.join(OUT_DIR, f"step_{step_num}.png")

    preamble = (
        "import pandas as pd, numpy as np, matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt, seaborn as sns\n"
        f"df = pd.read_csv({state['data_path']!r})\n"
        f"CHART_PATH = {chart_path!r}\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(preamble + state["code"])
        script_file = f.name

    try:
        r = subprocess.run(
            [sys.executable, script_file],
            capture_output=True,
            text=True,
            timeout=EXEC_TIMEOUT,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}
        )
    except subprocess.TimeoutExpired:
        return {"error": f"Execution timed out after {EXEC_TIMEOUT}s", "retries": state["retries"] + 1}
    finally:
        if os.path.exists(script_file):
            try:
                os.remove(script_file)
            except Exception:
                pass

    if r.returncode != 0:
        return {"error": r.stderr[-1500:], "retries": state["retries"] + 1}

    out = r.stdout[:5000].strip()
    return {"output": out, "error": None}


# ---------- Node 5: Critic ----------
CRITIC_SYSTEM = f"""You are a strict statistical reviewer of an exploratory data analysis.
You are given a sub-question, the code, and the printed output.
Evaluate whether the output provides reliable empirical evidence to answer the sub-question.

Rules:
1. Exact rates/means and sample sizes (n >= {MIN_GROUP_SIZE}) must be present.
2. If output has valid rates and group sizes, set verdict = 'accept'.
3. Extract 1 or 2 concrete findings. Each finding statement must be ONE concise sentence naming the groups,
   their rates/means, sample sizes as 'n = 123', and key takeaway.
4. If output failed or is completely empty, use 'revise' (or 'reject' if revisions exhausted).
"""


def critic(state: EDAState) -> dict:
    user_msg = (
        f"Sub-Question: {state['question']}\n\n"
        f"Printed Output:\n{state['output']}\n\n"
        f"Revisions remaining: {max(0, MAX_REVISIONS - state.get('revisions', 0))}"
    )

    try:
        review = critic_llm.invoke([
            ("system", CRITIC_SYSTEM),
            ("user", user_msg)
        ])
    except Exception as e:
        review = None

    if review is None:
        return {
            "verdict": "accept",
            "critique": "Critic review fallback",
            "pending_findings": [{
                "statement": sanitize_text(state["output"][:250].replace("\n", " ")),
                "strength": 3,
                "adjusted_for": [],
                "variable": ""
            }]
        }

    verdict = review.verdict
    can_revise = state["revisions"] < MAX_REVISIONS

    pending = []
    for f in review.findings:
        stmt = sanitize_text(f.statement)
        pending.append({
            "statement": stmt,
            "strength": max(1, min(5, int(f.strength))),
            "adjusted_for": f.adjusted_for,
            "variable": f.variable
        })

    if verdict == "revise" and not can_revise:
        verdict = "accept" if pending else "reject"

    update = {
        "verdict": verdict,
        "critique": review.reason,
        "pending_findings": pending if verdict == "accept" else []
    }
    if verdict == "revise":
        update["revisions"] = state["revisions"] + 1
        update["retries"] = 0

    return update


# ---------- Node 6: Recorder ----------
def recorder(state: EDAState) -> dict:
    idx = state["current_idx"]
    step_num = idx + 1
    chart_path = os.path.join(OUT_DIR, f"step_{step_num}.png")
    chart_saved = [chart_path] if os.path.exists(chart_path) else []

    if state.get("error"):
        return {
            "failed": [f"{state['question']} (Failed: {state['error'][:100]})"],
            "current_idx": idx + 1,
            "error": None
        }

    if state["verdict"] == "accept":
        recorded_findings = [
            {
                "question": state["question"],
                "finding": p["statement"],
                "strength": p["strength"],
                "adjusted_for": p.get("adjusted_for", []),
                "variable": p.get("variable", ""),
                "chart": chart_path if os.path.exists(chart_path) else ""
            }
            for p in state["pending_findings"]
        ]
        return {
            "findings": recorded_findings,
            "step_charts": chart_saved,
            "current_idx": idx + 1
        }

    return {
        "rejected": [f"{state['question']} ({state['critique']})"],
        "current_idx": idx + 1
    }


# ---------- Node 7: Reporter (Direct-Answer Synthesis) ----------
REPORTER_SYSTEM = """You are a Principal Data Analyst delivering a concise, structured analytical briefing in Markdown.
Your goal is to synthesize the empirical findings from the investigation steps into an executive summary.

FORMATTING & TYPOGRAPHY RULES:
- Do NOT repeat the user's question or prompt as a title or heading. The user's question is already visible.
- Do NOT use massive `#` or `##` headings. Use ONLY `###` level headings for compact, clean, readable typography.
- Use EXACTLY these four section headings:
   ### Executive Summary
   A 2-3 sentence direct answer synthesizing the primary takeaway with key rates and numbers.

   ### Key Findings & Empirical Evidence
   Structured bullet points presenting the evidence from each investigated sub-question.
   For each bullet, cite exact rates, group sizes (n = ...), and takeaways.
   Reference corresponding step charts where relevant (e.g. `[Step 1 Chart]`).

   ### Confounders & Key Nuances
   Explain what happened when controlling for other variables (what held vs what was explained by covariates).

   ### Recommended Next Steps
   3 concise, actionable follow-up steps for deeper modeling or decision-making.

- Strictly NO unsupported claims or causal language; report observational relationships and exact findings.
"""


def reporter(state: EDAState) -> dict:
    bullets = []
    for f in state["findings"]:
        adj = f" (controlled for: {', '.join(f['adjusted_for'])})" if f.get("adjusted_for") else ""
        chart_tag = f" [Refer to {os.path.basename(f['chart'])}]" if f.get("chart") else ""
        bullets.append(f"- Sub-Q: *{f['question']}*\n  Finding: {f['finding']}{adj}{chart_tag}")

    findings_text = "\n".join(bullets) if bullets else "No conclusive findings were verified."

    user_prompt = (
        f"USER INQUIRY / GOAL: {state['goal']}\n"
        f"TARGET COLUMN: {state['outcome']}\n\n"
        f"EMPIRICAL FINDINGS FROM INVESTIGATION:\n{findings_text}\n\n"
        f"FAILED OR REJECTED QUESTIONS:\n{', '.join(state['failed'] + state['rejected']) or 'None'}\n\n"
        "Synthesize these findings into the structured briefing using the required ### sections."
    )

    report_content = ask(REPORTER_SYSTEM, user_prompt)
    report_content = sanitize_text(report_content)
    # Normalize all headings to compact H3 level
    report_content = re.sub(r"^#{1,2}\s+", "### ", report_content, flags=re.MULTILINE)
    # Strip any redundant question repetition header
    report_content = re.sub(r"^###\s+Direct Answer:.*?\n+", "### Executive Summary\n", report_content, flags=re.IGNORECASE)

    path = os.path.join(OUT_DIR, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_content)

    return {}


# ---------- Routing Logic ----------
def route_after_exec(state: EDAState) -> str:
    if state.get("error"):
        return "coder" if state["retries"] < MAX_RETRIES else "recorder"
    return "critic"


def route_after_critic(state: EDAState) -> str:
    return "coder" if state["verdict"] == "revise" else "recorder"


def route_after_record(state: EDAState) -> str:
    idx = state.get("current_idx", 0)
    agenda = state.get("agenda", [])
    if idx < len(agenda):
        return "planner"
    return "reporter"


# ---------- Build Graph ----------
def build_graph():
    g = StateGraph(EDAState)
    g.add_node("profiler", profiler)
    g.add_node("planner", planner)
    g.add_node("coder", coder)
    g.add_node("executor", executor)
    g.add_node("critic", critic)
    g.add_node("recorder", recorder)
    g.add_node("reporter", reporter)

    g.add_edge(START, "profiler")
    g.add_edge("profiler", "planner")
    g.add_edge("planner", "coder")
    g.add_edge("coder", "executor")

    g.add_conditional_edges("executor", route_after_exec, ["coder", "critic", "recorder"])
    g.add_conditional_edges("critic", route_after_critic, ["coder", "recorder"])
    g.add_conditional_edges("recorder", route_after_record, ["planner", "reporter"])
    g.add_edge("reporter", END)

    return g.compile()


# ---------- Generator for Streamlit Stream ----------
def run_autonomous_eda_stream(df: pd.DataFrame, goal: str = "Explore key patterns and drivers", target: Optional[str] = None, max_steps: int = 4):
    """Generator running the Question-Directed LangGraph workflow and yielding live events and progress."""
    os.makedirs(OUT_DIR, exist_ok=True)
    temp_path = os.path.join(OUT_DIR, "current_dataset.csv")
    df.to_csv(temp_path, index=False)

    if not target or target not in df.columns:
        candidates = ["survived", "churn", "target", "label", "outcome", "y", "fraud"]
        found = next((c for c in df.columns if c.lower() in candidates), None)
        target = found if found else str(df.columns[-1])

    app = build_graph()
    initial: EDAState = {
        "data_path": temp_path,
        "goal": goal,
        "summary": "",
        "columns": [],
        "numeric_columns": [],
        "id_like_columns": [],
        "outcome": target,
        "agenda": [],
        "current_idx": 0,
        "question": "",
        "code": "",
        "output": "",
        "error": None,
        "retries": 0,
        "revisions": 0,
        "verdict": "",
        "critique": "",
        "pending_findings": [],
        "findings": [],
        "step_charts": [],
        "failed": [],
        "rejected": [],
    }

    accumulated_findings = []
    total_steps = max_steps

    for update in app.stream(initial, config={"recursion_limit": 100}, stream_mode="updates"):
        for node, delta in update.items():
            delta = delta or {}
            idx = delta.get("current_idx", 0)
            if delta.get("agenda"):
                total_steps = len(delta["agenda"])

            step_chart = os.path.join(OUT_DIR, f"step_{idx}.png")
            chart_file = step_chart if os.path.exists(step_chart) else None

            if delta.get("findings"):
                accumulated_findings.extend(delta["findings"])

            prog = min(1.0, max(0.0, idx / max(1, total_steps)))

            yield {
                "node": node,
                "delta": delta,
                "agenda": delta.get("agenda"),
                "question": delta.get("question"),
                "step": idx,
                "total_steps": total_steps,
                "progress": prog,
                "chart_path": chart_file,
                "preview": delta.get("critique") or delta.get("question") or ""
            }

    report_path = os.path.join(OUT_DIR, "report.md")
    report_content = ""
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report_content = f.read()

    yield {
        "node": "complete",
        "report": report_content,
        "target": target,
        "insights": accumulated_findings,
        "preview": "Investigation complete!"
    }
