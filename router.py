"""
AI Intent & Ambiguity Router for Data Analysis.
Routes user queries to either Quick Query (fast path), Autonomous EDA (deep path),
or ML Strategy Advisor (prescriptive ML path).
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
from typing import Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openrouter import ChatOpenRouter

load_dotenv(override=True)
if os.getenv("OPENROUTER") and not os.getenv("OPENROUTER_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER")

MODEL = os.getenv("FAST_MODEL", "nvidia/nemotron-3.5-lightning:free")
if MODEL.startswith("openrouter:"):
    MODEL = MODEL[len("openrouter:"):]

llm = ChatOpenRouter(
    model=MODEL,
    api_key=os.getenv("OPENROUTER") or os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
    max_retries=1,
    request_timeout=10000,
)


class RoutingDecision(BaseModel):
    intent: Literal["quick_query", "autonomous_eda", "modelling", "clarification_needed"] = Field(
        default="quick_query",
        description="Choose 'quick_query', 'autonomous_eda', 'modelling', or 'clarification_needed'."
    )
    confidence: float = Field(
        default=0.9,
        description="Confidence score from 0.0 to 1.0"
    )
    reasoning: str = Field(
        default="Direct query analysis",
        description="A concise one-sentence justification for why this route was selected."
    )
    clarification_prompt: Optional[str] = Field(
        default=None,
        description="If clarification is needed, a polite question asking how the user would like to proceed."
    )


ROUTER_SYSTEM = """Classify the user's data-analysis request into exactly one intent.

Intents:
- quick_query — Direct chart, single metric, simple table lookup, or conceptual Q&A (e.g. "plot age distribution", "average fare by class", "recommend an imputation method").
- autonomous_eda — Investigative, causal, multi-factor, or subgroup attribution inquiries. Includes exploring *why* an effect exists, *what drove* a disparity/penalty, mechanism testing, or demographic deep dives (e.g. "why did third-class passengers die?", "was this penalty driven by mothers, fathers, or children?", "key drivers of survival", "explore patterns across cohorts").
- modelling — ML advice: algorithm choice, feature engineering, validation, metrics (e.g. "best model to predict churn?", "LightGBM vs Logistic Regression", "modeling roadmap").
- clarification_needed — Ambiguous or fragmentary input whose intent is unclear (e.g. "fares", "check", "is it good").

Respond with ONLY a raw JSON object (no markdown, no commentary):
{"intent": "<intent>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>", "clarification_prompt": null | "<question>"}
"""


def _heuristic_fallback(user_query: str) -> Optional[RoutingDecision]:
    """Fast deterministic heuristic fallback (sub-millisecond latency)."""
    q = user_query.strip().lower()

    # Ambiguous / single word
    if len(q.split()) <= 1 and q not in ["help", "info", "eda", "model", "plot"]:
        return RoutingDecision(
            intent="clarification_needed",
            confidence=0.9,
            reasoning="Single-word query requires clarification.",
            clarification_prompt=f"Could you tell me more about what you'd like to do with '{user_query}'? For example, you could ask me to plot a chart, explore patterns in depth, or get modeling advice."
        )

    # ML Strategy keywords
    ml_keywords = [
        "model", "algorithm", "lightgbm", "xgboost", "catboost", "random forest",
        "logistic regression", "cross-validation", "train_test", "auc", "f1",
        "precision", "recall", "overfitting", "hyperparameter", "predictive model",
        "modelling strategy", "ml roadmap", "champion model", "baseline model",
        "feature selection", "feature engineering for ml", "predict ", "predicting",
        "classifier", "classification metric",
        "imputation strategy", "imputation method", "imputation technique",
        "which imputation", "should i impute", "handle missing",
    ]
    if any(k in q for k in ml_keywords):
        return RoutingDecision(
            intent="modelling",
            confidence=0.95,
            reasoning="Query matched machine learning strategy intent."
        )

    # Autonomous EDA keywords
    eda_keywords = [
        "why did", "why do", "causal", "drivers of", "investigate", "deep eda",
        "root cause", "multi-factor", "after controlling for", "confound",
        "what caused", "explore patterns", "deep dive", "risk factors",
        "disparity", "key drivers", "subgroup analysis", "autonomous eda",
        "help or hurt", "impact of", "effect of", "interaction between",
        "who died", "who survived", "patterns in"
    ]
    if any(k in q for k in eda_keywords):
        return RoutingDecision(
            intent="autonomous_eda",
            confidence=0.95,
            reasoning="Query matched deep autonomous investigation intent."
        )

    # Quick Query keywords
    quick_keywords = [
        "plot", "chart", "distribution", "histogram", "scatter", "bar",
        "box", "average", "mean", "median", "std", "count", "head", "tail",
        "show table", "missing", "correlation", "how many", "summary", "filter",
        "describe", "value_counts", "what does", "impute missing"
    ]
    if any(k in q for k in quick_keywords):
        return RoutingDecision(
            intent="quick_query",
            confidence=0.95,
            reasoning="Query matched interactive calculation, chart, or analytical advice intent."
        )

    return None


def _call_router_llm(prompt: str) -> str:
    res = llm.invoke([
        ("system", ROUTER_SYSTEM),
        ("user", prompt),
    ]).content
    if isinstance(res, list):
        return "".join(b if isinstance(b, str) else b.get("text", "") for b in res)
    return str(res)


def route_query(user_query: str, dataset_summary: str, columns: list[str], chat_history: list = None) -> RoutingDecision:
    """Classifies user queries via AI, falling back to keyword heuristics on failure."""
    # Single-word queries don't need an LLM call
    q = user_query.strip().lower()
    if len(q.split()) <= 1 and q not in ["help", "info", "eda", "model", "plot"]:
        return RoutingDecision(
            intent="clarification_needed",
            confidence=0.9,
            reasoning="Single-word query requires clarification.",
            clarification_prompt=f"Could you tell me more about what you'd like to do with '{user_query}'? For example, you could ask me to plot a chart, explore patterns in depth, or get modeling advice."
        )

    history_context = ""
    if chat_history:
        recent = chat_history[-3:]
        history_lines = [f"{m.get('role', 'user')}: {m.get('text', '')}" for m in recent if m.get('text')]
        if history_lines:
            history_context = f"Recent Conversation Context:\n" + "\n".join(history_lines) + "\n\n"

    user_prompt = (
        f"Dataset Columns: {', '.join(columns)}\n\n"
        f"Dataset Summary Snippet:\n{dataset_summary[:800]}\n\n"
        f"{history_context}"
        f"User Query: {user_query.strip()}\n\n"
        "Classify into JSON (intent, confidence, reasoning, clarification_prompt):"
    )

    # Primary path: AI classification
    executor = ThreadPoolExecutor(max_workers=1)
    fut = executor.submit(_call_router_llm, user_prompt)
    try:
        res = fut.result(timeout=8.0)
        match = re.search(r"\{.*\}", res, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            intent = data.get("intent", "quick_query")
            if intent not in ["quick_query", "autonomous_eda", "modelling", "clarification_needed"]:
                intent = "quick_query"
            return RoutingDecision(
                intent=intent,
                confidence=float(data.get("confidence", 0.9)),
                reasoning=str(data.get("reasoning", "Analyzed query intent")),
                clarification_prompt=data.get("clarification_prompt")
            )
    except Exception:
        pass
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    # Fallback: keyword heuristics (only if AI router failed)
    heuristic = _heuristic_fallback(user_query)
    if heuristic:
        return heuristic

    return RoutingDecision(
        intent="quick_query",
        confidence=0.6,
        reasoning="Defaulted to Quick Query"
    )
