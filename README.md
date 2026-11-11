<div align="center">

# 📊 AI Data Analyst & Autonomous EDA Agent

### An Intelligent Multi-Agent Conversational Data Analysis Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-1C3C3C?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-API-6366F1?logo=openai&logoColor=white)](https://openrouter.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E.svg)](LICENSE)

**Upload any CSV → Ask questions in plain English → Get instant charts, deep causal investigations, or ML strategy advice — all powered by autonomous AI agents.**

[Features](#-key-features) · [Architecture](#-system-architecture) · [Quick Start](#-quick-start) · [Usage Guide](#-usage-guide) · [Examples](#-example-queries) · [Tech Stack](#-tech-stack) · [Contributing](#-contributing)

</div>

---

## 🧩 What Is This?

This is a **conversational AI data analyst** built with Streamlit, LangGraph, and LangChain. Instead of writing code yourself, you simply upload a dataset and ask questions in natural language. An **AI Intent Router** automatically classifies your query and dispatches it to one of three specialized execution paths:

| Path | What It Does | Speed |
|:-----|:-------------|:------|
| ⚡ **Quick Query** | Single-shot charts, metrics, and table lookups | ~1–2 seconds |
| 🧠 **Autonomous EDA** | Multi-step hypothesis-driven causal investigation with structured reporting | ~30–90 seconds |
| 🤖 **Modelling Advisor** | Senior-level ML strategy with algorithm selection, validation, and feature engineering | ~10–20 seconds |

No query falls through the cracks — ambiguous inputs trigger a **Clarification** path that asks the user to refine their question.

---

## 🚀 Key Features

### ⚡ Quick Query — Instant Analysis in 1–2 Seconds
- Generates and executes Python code on-the-fly from natural language
- Automatically renders **Matplotlib/Seaborn charts** inline in the chat
- Extracts tabular results into interactive **Streamlit dataframes**
- Auto-retries once with self-correction if the generated code errors out
- Expandable code viewer so you can inspect and reuse the generated Python

### 🧠 Question-Directed Autonomous EDA — Deep Investigation
- Decomposes your complex question into a **3–4 step targeted investigation agenda**:
  1. **Baseline Disparity** — measure the raw effect of the queried factor
  2. **Confounder Control** — test if the relationship holds after controlling for covariates
  3. **Subgroup Vulnerability** — identify which specific segments are most affected
  4. **Multivariate Attribution** — quantify independent contributions and threshold effects
- Each step: generates code → executes it → saves a chart → undergoes **AI critic review** (validates sample sizes n ≥ 20, rates, and statistical rigor)
- Failed steps are **auto-retried** with error feedback; weak steps get **revision requests** from the critic
- Synthesizes a polished **executive briefing** with 4 structured sections (Executive Summary, Key Findings, Confounders & Nuances, Recommended Next Steps)
- All findings are **cached** in the session for the Modelling Advisor to consume

### 🤖 Modelling Advisor — Prescriptive ML Strategy
- Generates a **6-section ML strategy report** tailored to your specific dataset:
  1. Problem Formulation & Objective
  2. Algorithm Candidates & Trade-offs (Baseline → Champion → Alternative)
  3. Validation & Splitting Scheme (Stratified K-Fold, GroupKFold, leakage prevention)
  4. Feature Engineering & Preprocessing Recipes
  5. Evaluation Metric Selection & Justification
  6. Implementation Roadmap
- **Zero redundant computation** — instantly consumes a deterministic dataset profile + any cached EDA insights from the session (< 50ms)
- Grounds every recommendation in the actual column types, missingness patterns, and class balance of your data

### 🎯 Pure AI Routing — No Keyword Heuristics
- Every query is classified by a low-latency MoE model in **~500ms**
- Returns a structured decision with `intent`, `confidence` score, and `reasoning`
- Gracefully degrades: falls back to Quick Query if the router API fails

### 💾 Two-Tier Memory Caching
- **Tier 1**: Deterministic dataset profile (schema, types, missingness, class balance) — computed once, reused everywhere
- **Tier 2**: EDA investigation findings — cached in `st.session_state["eda_insights"]` and automatically consumed by the Modelling Advisor, eliminating redundant re-runs

---

## 🏗️ System Architecture

```
                                  User Prompt & Dataset
                                            │
                                            ▼
               ┌─────────────────────────────────────────────────────────┐
               │    AI Intent Router (Nemotron 3.5 Lightning: ~500ms)    │
               └────────────────────────────┬────────────────────────────┘
                                            │
                 ┌──────────────────────────┼──────────────────────────┐
                 │                          │                          │
                 ▼                          ▼                          ▼
           [⚡ Quick Query]           [🧠 Autonomous EDA]        [🤖 Modelling Advisor]
         (Nemotron 3.5 Lightning)  (Nemotron 3 Ultra 550B)    (Nemotron 3 Ultra 550B)
                 │                          │                          │
          Single Python Code         Decomposes into 3-4        Reads Data Profile +
          & Instant Chart (<2s)      Sub-Hypotheses (Agenda)    Cached EDA Insights
                                            │                          │
                                    ┌───────┴───────┐            Generates 6-Part ML
                                    │  Per Sub-Q:   │            Strategy Report
                                    │  Plan → Code  │            (Zero Code / No Rerun)
                                    │  → Execute    │
                                    │  → Critic     │
                                    │  → Record     │
                                    └───────┬───────┘
                                            │
                                    Synthesizes Direct-Answer
                                    Report & Caches Insights
                                            │
                                            ▼
                               st.session_state["eda_insights"]
```

### LangGraph EDA Agent Flow

The Autonomous EDA engine is built as a **stateful multi-agent graph** using LangGraph:

```
START → Profiler → Planner → Coder → Executor ─┬─→ Critic ─┬─→ Recorder ─┬─→ Planner (next sub-Q)
                                                │           │             │
                                          (on error,       (on "revise") └─→ Reporter → END
                                           retry ≤ 2)       back to Coder
                                           back to Coder
```

| Node | Role |
|:-----|:-----|
| **Profiler** | Reads the dataset, builds a schema summary, and decomposes the user's question into 3–4 targeted sub-questions |
| **Planner** | Schedules the current sub-question for investigation |
| **Coder** | Generates Python code to answer the sub-question with charts and group-size-aware statistics |
| **Executor** | Runs the code in a subprocess with a 60-second timeout |
| **Critic** | AI reviewer that validates output quality — checks for sample sizes (n ≥ 20), rates, and statistical rigor. Verdicts: `accept`, `revise`, or `reject` |
| **Recorder** | Stores accepted findings and chart paths; advances to the next sub-question |
| **Reporter** | Synthesizes all findings into a structured executive briefing with Markdown formatting |

---

## 🤖 Model Allocation

| Path | Module | Model | Purpose |
|:-----|:-------|:------|:--------|
| **Router** | `router.py` | `nvidia/nemotron-3.5-lightning:free` | Low-latency 3-way intent classification (~500ms) |
| **Quick Query** | `quick_query.py` | `nvidia/nemotron-3.5-lightning:free` | Single-shot code generation & plot creation (~1s) |
| **Autonomous EDA** | `profiler.py` | `nvidia/nemotron-3-ultra-550b-a55b:free` | Multi-agent hypothesis decomposition, code gen, critique, and synthesis |
| **Modelling Advisor** | `modelling_advisor.py` | `nvidia/nemotron-3-ultra-550b-a55b:free` | Deep reasoning for ML strategy, problem formulation, and metrics |

> **Note**: All models are accessed via [OpenRouter](https://openrouter.ai/) and are currently using free-tier endpoints. You can swap in any OpenRouter-compatible model by editing `.env`.

---

## 📂 Project Structure

```
📦 AI-Data-Analyst/
├── 📄 app.py                  # Streamlit web app — chat interface, routing, rendering
├── 📄 router.py               # AI Intent Router — classifies queries into 3 execution paths
├── 📄 quick_query.py          # Fast-path engine — generates & executes Python in-memory
├── 📄 profiler.py             # LangGraph Autonomous EDA — multi-agent investigation pipeline
├── 📄 modelling_advisor.py    # ML Strategy Advisor — prescriptive modeling recommendations
├── 📄 requirements.txt       # Python dependencies
├── 📄 .env                    # API keys and model configuration (not committed)
├── 📄 .gitignore              # Ignores .env, __pycache__/, outputs/
├── 📄 Titanic-Dataset.csv     # Built-in demo dataset for quick testing
└── 📁 outputs/                # Generated artifacts (auto-created at runtime)
    ├── current_dataset.csv    #   Working copy of the uploaded dataset
    ├── report.md              #   Latest EDA report in Markdown
    └── step_*.png             #   Investigation charts (step_1.png, step_2.png, …)
```

---

## ⚡ Quick Start

### Prerequisites

- **Python 3.10+** (tested on Python 3.13)
- An **[OpenRouter API Key](https://openrouter.ai/keys)** (free tier works out of the box)
- Git (for cloning)

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/AI-Data-Analyst.git
cd AI-Data-Analyst
```

### 2. Create a Virtual Environment (Recommended)

```bash
# On macOS / Linux
python -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Models (defaults shown — change to use different providers or tiers)
FAST_MODEL=nvidia/nemotron-3.5-lightning:free
REASONING_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
EDA_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
```

<details>
<summary>💡 <b>How to get your OpenRouter API Key</b></summary>

1. Go to [openrouter.ai](https://openrouter.ai/) and create an account
2. Navigate to [API Keys](https://openrouter.ai/keys)
3. Click **"Create Key"** and copy the generated key
4. Paste it into your `.env` file as `OPENROUTER_API_KEY=sk-or-...`

The default models use free-tier endpoints, so **no credit card is needed** to get started.
</details>

### 5. Launch the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📖 Usage Guide

### Step 1 — Load a Dataset

You have two options:
- **Upload your own CSV** using the file uploader in the sidebar
- **Click "🚢 Load Titanic Demo"** to instantly load the built-in Titanic dataset

### Step 2 — Select a Target Column

Pick the primary target/outcome column from the dropdown in the sidebar. The system auto-detects common target names (`Survived`, `Churn`, `Target`, `Label`, etc.), but you can override it.

> The target column is used by the Autonomous EDA engine for hypothesis testing and baseline separation, and by the Modelling Advisor for problem formulation.

### Step 3 — Ask Questions in Natural Language

Type any question into the chat input box. The AI Router will automatically determine the best execution path:

| If Your Question Is… | It Routes To… | You Get… |
|:----------------------|:--------------|:---------|
| A specific chart, metric, or lookup | ⚡ **Quick Query** | Instant chart + data table + generated code |
| An investigative, causal, or "why" question | 🧠 **Autonomous EDA** | Multi-step investigation report + step charts |
| About ML models, features, or evaluation | 🤖 **Modelling Advisor** | 6-section ML strategy report |
| Too vague or ambiguous | 💬 **Clarification** | A helpful follow-up question |

### Step 4 — Explore Results

- **Charts** render inline in the chat — you can expand the code that generated them
- **EDA reports** include an executive summary, key findings, and recommended next steps
- **Step charts** are displayed in an expandable gallery
- **EDA insights are automatically cached** — when you later ask a modeling question, the advisor incorporates those findings

---

## 💡 Example Queries

### ⚡ Quick Query — Fast, Single-Shot Analysis

```
"Plot age distribution"
"What is the median fare by passenger class?"
"Show survival rate by sex"
"Correlation between SibSp and Parch"
"Show the first 10 rows"
"How many missing values are there per column?"
"Create a boxplot of fare by embarkation port"
```

### 🧠 Autonomous EDA — Deep Multi-Step Investigation

```
"Why did third-class passengers die?"
"Did having family on board help or hurt survival?"
"Does embarkation port matter after controlling for class and sex?"
"Analyze the patterns and key drivers in the dataset"
"Identify key risk factors for not surviving"
"What is the interaction between age, sex, and class on survival?"
```

### 🤖 Modelling Advisor — ML Strategy & Recommendations

```
"Which machine learning model should I train to predict survival?"
"How should I handle class imbalance and missing values for modeling?"
"What features should I engineer to predict the outcome?"
"What evaluation metric should I use for this dataset?"
"Give me a complete modeling roadmap for this table"
"Would LightGBM or Logistic Regression be better here?"
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|:------|:-----------|:--------|
| **Frontend** | [Streamlit](https://streamlit.io/) | Chat-based web UI with real-time rendering |
| **Orchestration** | [LangGraph](https://langchain-ai.github.io/langgraph/) | Stateful multi-agent graph for the Autonomous EDA pipeline |
| **LLM Framework** | [LangChain](https://python.langchain.com/) + [langchain-openrouter](https://pypi.org/project/langchain-openrouter/) | Structured output, prompt management, model abstraction |
| **LLM Provider** | [OpenRouter](https://openrouter.ai/) | Unified API gateway to NVIDIA Nemotron and other models |
| **Models** | NVIDIA Nemotron 3.5 Lightning & Nemotron 3 Ultra 550B | Fast routing + deep reasoning |
| **Data Processing** | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [Statsmodels](https://www.statsmodels.org/) | Data manipulation, statistical analysis |
| **Visualization** | [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/) | Chart generation (AI-generated code) |
| **Structured Output** | [Pydantic](https://docs.pydantic.dev/) | Type-safe schema enforcement for router decisions, critic reviews, and investigation plans |

---

## 📋 Requirements

Full dependency list from [`requirements.txt`](requirements.txt):

```
langgraph
langchain
langchain-google-genai
langchain-openrouter
pandas
numpy
scipy
statsmodels
matplotlib
seaborn
python-dotenv
streamlit
```

Install with:
```bash
pip install -r requirements.txt
```

> **Python version**: 3.10 or newer. Tested and developed on Python 3.13.

---

## ⚙️ Configuration

All configuration is managed through the `.env` file:

| Variable | Required | Default | Description |
|:---------|:---------|:--------|:------------|
| `OPENROUTER_API_KEY` | ✅ Yes | — | Your OpenRouter API key |
| `FAST_MODEL` | No | `nvidia/nemotron-3.5-lightning:free` | Model for Router + Quick Query (low-latency) |
| `REASONING_MODEL` | No | `nvidia/nemotron-3-ultra-550b-a55b:free` | Model for Autonomous EDA + Modelling Advisor (deep reasoning) |
| `EDA_MODEL` | No | (falls back to `REASONING_MODEL`) | Optional separate model override for EDA only |

### Swapping Models

You can use **any model available on OpenRouter**. For example:

```env
# Use Anthropic Claude for reasoning
REASONING_MODEL=anthropic/claude-sonnet-4-20250514

# Use Google Gemini for fast queries
FAST_MODEL=google/gemini-2.0-flash-001

# Use Meta Llama for EDA
EDA_MODEL=meta-llama/llama-3.1-405b-instruct
```

---

## 🗂️ Output Artifacts

When running an Autonomous EDA investigation, the system generates artifacts in the `outputs/` directory:

| File | Description |
|:-----|:------------|
| `current_dataset.csv` | Working copy of the uploaded dataset (used by the subprocess executor) |
| `report.md` | Final synthesized EDA report in Markdown |
| `step_1.png` … `step_N.png` | Visualization chart for each investigation sub-question |

> The `outputs/` directory is listed in `.gitignore` and is not committed to the repository.

---

## 🧪 Troubleshooting

<details>
<summary><b>❌ "AI model request failed" or router errors</b></summary>

- Verify your `OPENROUTER_API_KEY` is correct in `.env`
- Check your OpenRouter dashboard for rate limits or quota
- The free-tier models may be temporarily unavailable — try again in a few minutes
- The router falls back to Quick Query if the API is unreachable

</details>

<details>
<summary><b>❌ "Execution Error" on Quick Query</b></summary>

- The LLM-generated code failed to run. The system auto-retries once with the error message
- If it persists, try rephrasing your question to be more specific
- Check the generated code in the "🔍 View Generated Code" expander for hints

</details>

<details>
<summary><b>❌ Charts not rendering / blank figures</b></summary>

- Ensure `matplotlib` and `seaborn` are installed: `pip install matplotlib seaborn`
- The app uses the `Agg` backend — charts are rendered server-side, not in a GUI window

</details>

<details>
<summary><b>❌ EDA investigation seems stuck</b></summary>

- Each code execution step has a 60-second timeout
- Complex queries on very large datasets may take longer — consider filtering or sampling your data
- Watch the live status panel for progress on which agent node is executing

</details>

<details>
<summary><b>❌ Modelling Advisor gives generic advice</b></summary>

- Run an Autonomous EDA investigation first — the advisor uses cached EDA insights to give data-grounded recommendations
- Make sure you've selected the correct target column in the sidebar

</details>

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feature/your-feature`
3. **Commit your changes**: `git commit -m "Add your feature"`
4. **Push to your branch**: `git push origin feature/your-feature`
5. **Open a Pull Request**

### Ideas for Contributions

- 🧪 Add unit tests for the router, quick query executor, and LangGraph pipeline
- 📊 Add Plotly support for interactive charts
- 📁 Support additional file formats (`.xlsx`, `.parquet`, `.json`)
- 🌙 Dark mode theme support
- 📥 PDF/HTML export for EDA reports
- 💬 Multi-turn conversation memory

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[OpenRouter](https://openrouter.ai/)** — Unified LLM API gateway
- **[NVIDIA](https://build.nvidia.com/)** — Nemotron model family (3.5 Lightning & Ultra 550B)
- **[LangChain](https://python.langchain.com/) & [LangGraph](https://langchain-ai.github.io/langgraph/)** — Agent orchestration framework
- **[Streamlit](https://streamlit.io/)** — Rapid web app framework for data science

---

<div align="center">

**Built for modular, autonomous data science workflows.**

⭐ Star this repo if you found it useful!

</div>
