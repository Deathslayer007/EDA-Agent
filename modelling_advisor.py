"""
Modelling Strategy & Advisory Engine (Prescriptive ML Path).
Provides structured, senior-level Machine Learning recommendations without executing code
or re-running redundant EDA. Builds upon the deterministic data profile and cached EDA findings.
"""
import os
from typing import Optional, List, Dict
import pandas as pd
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter

load_dotenv(override=True)
if os.getenv("OPENROUTER") and not os.getenv("OPENROUTER_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER")

MODEL = os.getenv("FAST_MODEL", "nvidia/nemotron-3.5-lightning:free")
if MODEL.startswith("openrouter:"):
    MODEL = MODEL[len("openrouter:"):]

advisor_llm = ChatOpenRouter(
    model=MODEL,
    api_key=os.getenv("OPENROUTER") or os.getenv("OPENROUTER_API_KEY"),
    temperature=0.2,
    max_retries=1,
    request_timeout=35000,
)

MODELLING_SYSTEM_PROMPT = """You are a Principal ML & Data Science Architect. Provide pragmatic, production-ready ML guidance tailored to this specific dataset.

RESPONSE APPROACH:
- Specific question (algorithm, metric, column handling, imbalance, leakage) → answer it directly, grounded in the dataset's columns, types, missingness, and row count.
- Broad strategy request → craft a bespoke plan organized around the dataset's real bottlenecks (sample size, class balance, cardinality, noise) — not a generic checklist.

GROUNDING RULES:
- Cite actual column names and specify exact transformations (encoding, interaction terms, imputation).
- Match model complexity to sample size (e.g., n < 1 K → regularized linear / shallow gradient boosting).
- Flag leakage risks (IDs, target proxies) and prescribe a sound validation scheme (e.g., Stratified K-Fold with out-of-fold encoding).
- Incorporate any prior EDA findings from this session (verified predictors, non-linearities, subgroups).
- Format: `###` headings, bold key terms, concise bullets.
"""


def format_dataset_profile(df: pd.DataFrame, target_col: Optional[str] = None) -> str:
    """Computes a deterministic, zero-latency profile of the dataset for ML modeling."""
    n_rows, n_cols = df.shape
    lines = [
        f"Rows: {n_rows:,} | Columns: {n_cols:,}",
        f"Primary Target Column: {target_col or 'Not specified'}",
        ""
    ]

    # Target stats if available
    if target_col and target_col in df.columns:
        target_series = df[target_col].dropna()
        n_unique = target_series.nunique()
        if n_unique == 2:
            val_counts = target_series.value_counts(normalize=True).to_dict()
            dist_str = ", ".join(f"'{k}': {v:.1%}" for k, v in val_counts.items())
            lines.append(f"Target Type: Binary Classification (Class Balance: {dist_str})")
        elif n_unique < 15 and pd.api.types.is_integer_dtype(target_series):
            lines.append(f"Target Type: Multiclass Classification ({n_unique} distinct classes)")
        elif pd.api.types.is_numeric_dtype(target_series):
            lines.append(f"Target Type: Regression (Mean={target_series.mean():.2f}, Std={target_series.std():.2f}, Min={target_series.min()}, Max={target_series.max()})")
        else:
            lines.append(f"Target Type: Categorical ({n_unique} unique categories)")
        lines.append("")

    # Column inventory
    lines.append("Columns Breakdown:")
    for col in df.columns:
        s = df[col]
        missing_pct = s.isna().mean()
        n_unique = s.nunique()
        dtype_str = str(s.dtype)
        lines.append(f"- {col}: dtype={dtype_str}, missing={missing_pct:.1%}, unique={n_unique}")

    return "\n".join(lines)


def stream_modelling_advice(
    query: str,
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    eda_insights: Optional[List[Dict]] = None,
    chat_history: Optional[List[Dict]] = None
):
    """Streams tailored ML strategy tokens in real-time from the LLM."""
    profile_str = format_dataset_profile(df, target_col)

    eda_context = "None available yet (Autonomous EDA has not been run in this session)."
    if eda_insights:
        bullets = []
        for idx, item in enumerate(eda_insights, 1):
            var = item.get("variable") or item.get("focus") or "Feature"
            finding = item.get("finding") or item.get("statement") or str(item)
            adjusted = f" (controlled for: {', '.join(item['adjusted_for'])})" if item.get("adjusted_for") else ""
            bullets.append(f"{idx}. [{var}{adjusted}]: {finding}")
        eda_context = "\n".join(bullets)

    history_context = ""
    if chat_history:
        recent = chat_history[-3:]
        history_lines = [f"{m.get('role', 'user')}: {m.get('text', '')}" for m in recent if m.get('text')]
        if history_lines:
            history_context = "RECENT CONVERSATION CONTEXT:\n" + "\n".join(history_lines) + "\n\n"

    user_q = (query or "").strip()
    if not user_q:
        instruction = (
            f"Provide a tailored Machine Learning architectural blueprint for predicting '{target_col}'. "
            "Focus on the most critical modeling decisions for this dataset: model candidates suited for this sample size and tabular structure, "
            "key feature engineering recipes for the specific columns, validation strategy to avoid overfitting, and the best evaluation metric."
        )
    else:
        instruction = (
            f"Address the user's specific modeling inquiry: '{user_q}'. "
            "Provide a direct, thorough, and highly tailored recommendation grounded in this dataset's specific columns, target distribution, and prior EDA findings. "
            "Focus strictly on what the user asked without dumping unrelated boilerplate sections."
        )

    prompt = (
        f"DATASET PROFILE:\n{profile_str}\n\n"
        f"PRIOR EDA INSIGHTS DISCOVERED IN SESSION:\n{eda_context}\n\n"
        f"{history_context}"
        f"USER MODELLING INQUIRY: {user_q or f'Bespoke ML strategy for {target_col}'}\n\n"
        f"TASK: {instruction}"
    )

    try:
        for chunk in advisor_llm.stream([
            ("system", MODELLING_SYSTEM_PROMPT),
            ("user", prompt)
        ]):
            text = chunk.content if hasattr(chunk, "content") else str(chunk)
            if isinstance(text, list):
                text = "".join(b if isinstance(b, str) else b.get("text", "") for b in text)
            if text:
                yield text
    except Exception as e:
        yield (
            f"\n\n### Modelling Advisory Notice\n\n"
            f"Could not complete real-time generation: `{e}`\n\n"
            f"**Baseline Recommendations for {target_col or 'your target'}:**\n"
            f"- For tabular classification, start with a 5-fold Stratified CV baseline using Logistic Regression (with StandardScaling).\n"
            f"- Train LightGBM or XGBoost to evaluate non-linear feature interactions.\n"
            f"- Evaluate using PR-AUC (Average Precision) if the positive class is rare, or ROC-AUC if roughly balanced.\n"
        )


def run_modelling_advice(
    query: str,
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    eda_insights: Optional[List[Dict]] = None,
    chat_history: Optional[List[Dict]] = None
) -> str:
    """Generates comprehensive ML strategy advisory report based on data profile, prior EDA insights, and conversation context."""
    chunks = list(stream_modelling_advice(query, df, target_col, eda_insights, chat_history))
    response = "".join(chunks)
    import re
    response = re.sub(r"^#{1,2}\s+", "### ", response.strip(), flags=re.MULTILINE)
    return response
