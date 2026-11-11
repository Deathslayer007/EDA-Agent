"""
AI Intent & Ambiguity Router for Data Analysis.
Routes user queries to either Quick Query (fast path) or Autonomous EDA (deep path).
"""
import os
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
    max_retries=3,
)


class RoutingDecision(BaseModel):
    intent: Literal["quick_query", "autonomous_eda", "modelling", "clarification_needed"] = Field(
        description="Choose 'quick_query' for specific, closed-ended charts, metrics, or table lookups. "
                    "Choose 'autonomous_eda' for open-ended, investigative, causal, or multi-factor pattern exploration. "
                    "Choose 'modelling' for predictive modeling questions, ML algorithm selection, validation strategies, feature engineering for ML, or metric guidance. "
                    "Choose 'clarification_needed' if the query is too ambiguous, vague, or just a single keyword."
    )
    confidence: float = Field(
        default=0.9,
        description="Confidence score from 0.0 to 1.0"
    )
    reasoning: str = Field(
        description="A concise one-sentence justification for why this route was selected."
    )
    clarification_prompt: Optional[str] = Field(
        default=None,
        description="If clarification is needed, a polite question asking how the user would like to proceed."
    )


ROUTER_SYSTEM = """You are an intelligent data triage assistant.
Your job is to analyze the user's natural language request on a dataset and classify it into one of four execution paths:

1. 'quick_query' (Fast Path):
   - The user is asking for a specific, unambiguous chart, metric, calculation, or table lookup.
   - Examples:
     * "Plot age distribution"
     * "What is the average fare by class?"
     * "Show survival rate by sex"
     * "Correlation between SibSp and Parch"
     * "Show the first 10 rows"
   - These can be solved in a single Python snippet in ~1-2 seconds.

2. 'autonomous_eda' (Deep Path):
   - The user asks an investigative, causal, comparative, or open-ended analytical question that requires multi-step hypothesis testing, confounding control, sample size validation, and structured reporting.
   - Examples:
     * "Why did passengers in third class die?"
     * "What are the main drivers of survival and how do they interact?"
     * "Explore this dataset and find interesting patterns"
     * "Does embarkation port matter after controlling for class and sex?"
     * "Identify key risk factors"

3. 'modelling' (ML Strategy & Advisory):
   - The user asks questions or requests advice regarding machine learning, predictive modeling, algorithm selection, feature engineering recipes for ML, validation strategy, or evaluation metrics.
   - Examples:
     * "Which machine learning algorithm is best to predict survival on this dataset?"
     * "How should I handle class imbalance and missing values for modeling?"
     * "What features should I engineer to predict the outcome?"
     * "Would LightGBM or Logistic Regression be better here?"
     * "What evaluation metric should I use for this dataset?"
     * "Give me a modeling roadmap for this table"

4. 'clarification_needed':
   - The user provided a single keyword, fragmented thought, or ambiguous input where their intention is unclear.
   - Examples: "fares", "check", "survival", "ticket", "is it good"
   - Provide a helpful clarification question in clarification_prompt.
"""

router_llm = llm.with_structured_output(RoutingDecision)


def route_query(user_query: str, dataset_summary: str, columns: list[str], chat_history: list = None) -> RoutingDecision:
    """Classify the user's natural language query using structured reasoning and multi-turn context."""
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
        f"User Query: {user_query.strip()}"
    )
    try:
        decision = router_llm.invoke([
            ("system", ROUTER_SYSTEM),
            ("user", user_prompt),
        ])
        return decision
    except Exception as e:
        # Fallback to quick_query if router encounters API issues
        return RoutingDecision(
            intent="quick_query",
            confidence=0.5,
            reasoning=f"Defaulted to Quick Query due to router fallback: {str(e)[:80]}",
            clarification_prompt=None
        )
