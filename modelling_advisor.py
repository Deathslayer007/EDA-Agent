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

MODEL = os.getenv("REASONING_MODEL", os.getenv("EDA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"))
if MODEL.startswith("openrouter:"):
    MODEL = MODEL[len("openrouter:"):]

advisor_llm = ChatOpenRouter(
    model=MODEL,
    api_key=os.getenv("OPENROUTER") or os.getenv("OPENROUTER_API_KEY"),
    temperature=0.2,
    max_retries=3,
)

MODELLING_SYSTEM_PROMPT = """You are a Principal Machine Learning & Data Science Architect.
Your task is to provide rigorous, actionable, production-ready Machine Learning strategy recommendations
for the user's tabular dataset and analytical inquiry.

GUIDELINES:
- Ground all recommendations directly in the provided dataset schema, column types, target distribution, and prior EDA findings.
- If prior EDA findings are provided, explicitly reference them (e.g. known strong predictors, non-linearities, collinearity, or missingness mechanisms).
- Do NOT provide generic textbook boilerplate; tailor every recommendation to the specific variables and characteristics of this dataset.
- Structure your response using these exact Markdown sections:

### 1. Problem Formulation & Objective
- Define the formal ML task (e.g. Binary Classification, Multiclass Classification, Regression).
- Characterize the target distribution (e.g. class imbalance ratio, skewness, zero-inflation).
- Baseline performance expectation and domain risk profile.

### 2. Recommended Algorithm Candidates & Trade-offs
- **Baseline Benchmark**: A simple, interpretable model (e.g. Logistic Regression with L2 penalty, shallow Decision Tree).
- **Primary Champion**: A modern high-performance model (e.g. LightGBM, XGBoost, or CatBoost) with justification for why it fits this data structure (handling tabular features, categoricals, missing values).
- **Secondary / Alternative**: Neural tabular, Random Forest, or regularized GLM. Detail interpretability vs throughput vs accuracy trade-offs.

### 3. Validation & Splitting Scheme
- Cross-validation architecture (e.g. Stratified 5-Fold CV, GroupKFold if grouped entities exist, or TimeSeriesSplit if temporal ordering exists).
- Data leakage prevention rules (e.g. target encoding and imputation parameters fit strictly inside training folds).

### 4. Feature Engineering & Preprocessing Recipes
- High-cardinality & categorical handling (Target encoding vs One-Hot vs Ordinal).
- Missing data strategy: Distinguish between MNAR (add missing indicator feature) vs MAR (stratified/conditioned median imputation).
- Interaction terms & domain aggregations (specifically incorporating relationships highlighted in prior EDA).
- Feature scaling & outlier treatment (whether required for the chosen models).

### 5. Evaluation Metric Selection & Justification
- Explain why naive metrics (e.g. overall Accuracy on imbalanced targets) are misleading.
- Recommend primary metric (e.g. PR-AUC / Average Precision, ROC-AUC, F1-macro, Expected Cost/Loss) and secondary monitoring metrics.

### 6. Implementation Roadmap
- An actionable 4-step checklist for building and deploying this ML pipeline.
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


def run_modelling_advice(
    query: str,
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    eda_insights: Optional[List[Dict]] = None,
    chat_history: Optional[List[Dict]] = None
) -> str:
    """Generates comprehensive ML strategy advisory report based on data profile, prior EDA insights, and conversation context."""
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

    prompt = (
        f"DATASET PROFILE:\n{profile_str}\n\n"
        f"PRIOR EDA INSIGHTS DISCOVERED IN SESSION:\n{eda_context}\n\n"
        f"{history_context}"
        f"USER MODELLING QUESTION / INQUIRY:\n{query.strip()}\n\n"
        "Provide a complete, structured Machine Learning Strategy and Advisory Report tailored to this exact dataset and question."
    )

    try:
        response = advisor_llm.invoke([
            ("system", MODELLING_SYSTEM_PROMPT),
            ("user", prompt)
        ]).content

        if isinstance(response, list):
            response = "".join(b if isinstance(b, str) else b.get("text", "") for b in response)

        import re
        response = re.sub(r"^#{1,2}\s+", "### ", response.strip(), flags=re.MULTILINE)
        return response
    except Exception as e:
        return (
            f"### Modelling Advisory Error\n\n"
            f"An error occurred while generating modeling advice via the LLM: `{e}`\n\n"
            f"**Fallback Recommendations for {target_col or 'your target'}:**\n"
            f"- For tabular classification, start with a 5-fold Stratified CV baseline using Logistic Regression (with StandardScaling).\n"
            f"- Train LightGBM or XGBoost with default hyperparameters to evaluate non-linear feature interactions.\n"
            f"- Evaluate using PR-AUC (Average Precision) if the positive class is rare, or ROC-AUC if class balance is roughly equal."
        )
