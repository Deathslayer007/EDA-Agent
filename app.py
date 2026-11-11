"""
AI Data Analyst & Autonomous EDA Agent.
Unified Single-Page Conversational Interface:
Automatically routes queries through the AI Intent Router to:
  1. ⚡ Quick Query (Fast path: charts, aggregations, analytical Q&A)
  2. 🧠 Deep Autonomous EDA (Multi-agent LangGraph causal investigation)
  3. 🤖 ML Strategy Advisor (Prescriptive ML strategy & architecture)
  4. 💬 Clarification (Follow-up prompt for ambiguous queries)
"""
import io
import os
import time
import zipfile
from typing import Optional

import pandas as pd
import streamlit as st

from router import route_query
from quick_query import run_quick_query
from profiler import run_autonomous_eda_stream, OUT_DIR
from modelling_advisor import run_modelling_advice, stream_modelling_advice

# ── Avatars ──────────────────────────────────────────────────────────
USER_AVATAR = "👤"
AI_AVATAR = "🤖"

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Analyst & Autonomous EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling (Theme-safe for both Dark and Light modes) ────────────────
st.markdown("""
<style>
    /* Consistent chat message spacing */
    .stChatMessage { margin-bottom: 6px !important; }

    /* Border containers theme safety (prevents white-on-white in dark mode) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: transparent !important;
    }

    /* Route badge pills - Theme adaptive */
    .route-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .route-quick {
        background-color: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.35);
    }
    .route-deep {
        background-color: rgba(168, 85, 247, 0.15);
        color: #C084FC;
        border: 1px solid rgba(168, 85, 247, 0.35);
    }
    .route-ml {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .route-clarify {
        background-color: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }
</style>
""", unsafe_allow_html=True)


# ── Session state ────────────────────────────────────────────────────
_defaults = {
    "messages": [],
    "df": None,
    "df_name": "",
    "target_col": None,
    "eda_insights": [],
    "eda_report": "",
    "eda_charts": [],
    "injected_prompt": None,
    "cleaning_applied": [],
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Helpers ──────────────────────────────────────────────────────────
def auto_clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Auto-cleans numeric strings, booleans, and date columns."""
    df_clean = df.copy()
    actions = []
    for col in df_clean.columns:
        s = df_clean[col]
        if s.dtype == "object":
            sample = s.dropna().astype(str).str.strip()
            if len(sample) > 0 and sample.str.match(r"^[\$€£]?\s*-?\d{1,3}(,\d{3})*(\.\d+)?$").mean() > 0.8:
                try:
                    df_clean[col] = sample.str.replace(r"[\$€£, ]", "", regex=True).astype(float)
                    actions.append(f"'{col}' → numeric")
                    continue
                except Exception:
                    pass
            unique_vals = set(sample.str.lower().unique())
            if unique_vals.issubset({"yes", "no", "y", "n"}):
                df_clean[col] = sample.str.lower().isin(["yes", "y"])
                actions.append(f"'{col}' → boolean")
                continue
            elif unique_vals.issubset({"true", "false", "t", "f"}):
                df_clean[col] = sample.str.lower().isin(["true", "t"])
                actions.append(f"'{col}' → boolean")
                continue
            if any(kw in col.lower() for kw in ["date", "time", "timestamp"]):
                try:
                    df_clean[col] = pd.to_datetime(s, errors="coerce")
                    actions.append(f"'{col}' → datetime")
                except Exception:
                    pass
    return df_clean, actions


def load_dataset_file(uploaded_file) -> pd.DataFrame:
    fname = uploaded_file.name.lower()
    if fname.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif fname.endswith(".tsv"):
        return pd.read_csv(uploaded_file, sep="\t")
    elif fname.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    elif fname.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)
    elif fname.endswith(".json"):
        return pd.read_json(uploaded_file)
    return pd.read_csv(uploaded_file)


def get_dataset_summary(df: pd.DataFrame) -> str:
    lines = [f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns", "Columns:"]
    for c in df.columns:
        lines.append(f"- {c}: {df[c].dtype}, {df[c].isna().mean():.1%} missing, {df[c].nunique()} unique")
    return "\n".join(lines)


def format_display_text(text: str) -> str:
    if not text:
        return ""
    if any(m in text for m in ["#", "|---", "**", "- "]):
        return text
    if "\n" in text and any("  " in line for line in text.splitlines()):
        return f"```\n{text}\n```"
    return text


def stream_text(text: str, delay: float = 0.012):
    """Smoothly streams text word-by-word into Streamlit's st.write_stream."""
    if not text:
        return
    words = text.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        time.sleep(delay)


def create_charts_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        if os.path.exists(OUT_DIR):
            for f in os.listdir(OUT_DIR):
                if f.endswith(".png"):
                    z.write(os.path.join(OUT_DIR, f), arcname=f)
    return buf.getvalue()


def build_suggestions(df: pd.DataFrame, target: Optional[str]) -> list[str]:
    """Generates 3 suggestion chips representing the 3 primary routes."""
    chips = []
    num = [c for c in df.select_dtypes("number").columns if c != target]
    cat = [c for c in df.select_dtypes(exclude="number").columns if c != target]

    # 1. Quick Query chip
    if num:
        chips.append(f"Plot distribution of {num[0]}")
    elif cat and target:
        chips.append(f"Show {target} rate by {cat[0]}")
    else:
        chips.append("Show summary statistics")

    # 2. Deep Autonomous EDA chip
    if target:
        chips.append(f"Why did passengers survive or die, and what are key drivers of {target}?" if "survive" in str(target).lower() else f"What are the key drivers and risk patterns affecting {target}?")
    else:
        chips.append("Investigate key patterns and anomalies in this dataset")

    # 3. ML Strategy Advisor chip
    if target:
        chips.append(f"Which ML model, features, and validation strategy are best for {target}?")
    else:
        chips.append("Recommend an ML architecture, feature engineering, and validation scheme")

    return chips[:3]


# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 AI Data Analyst")
    st.caption("Conversational Data Intelligence with AI Intent Router & LangGraph")

    uploaded_file = st.file_uploader(
        "Upload dataset",
        type=["csv", "tsv", "xlsx", "xls", "parquet", "json"],
        help="Upload CSV, Excel, Parquet, or JSON dataset."
    )
    if uploaded_file is not None:
        try:
            raw_df = load_dataset_file(uploaded_file)
            cleaned_df, cleaning_log = auto_clean_dataset(raw_df)
            st.session_state["df"] = cleaned_df
            st.session_state["df_name"] = uploaded_file.name
            st.session_state["cleaning_applied"] = cleaning_log
        except Exception as e:
            st.error(f"Error loading file: {e}")

    if st.button("🚢 Load Titanic Demo", use_container_width=True):
        if os.path.exists("Titanic-Dataset.csv"):
            raw_df = pd.read_csv("Titanic-Dataset.csv")
            cleaned_df, cleaning_log = auto_clean_dataset(raw_df)
            st.session_state["df"] = cleaned_df
            st.session_state["df_name"] = "Titanic-Dataset.csv"
            st.session_state["cleaning_applied"] = cleaning_log
            st.rerun()

    if st.session_state["df"] is not None:
        df = st.session_state["df"]
        st.divider()

        # Dataset info
        st.caption(f"**{st.session_state['df_name']}** — {df.shape[0]:,} rows × {df.shape[1]} cols")

        # Target column selector
        default_idx = 0
        auto_targets = ["survived", "churn", "target", "label", "outcome", "y", "fraud", "is_fraud"]
        found = next((c for c in df.columns if c.lower() in auto_targets), None)
        if found:
            default_idx = list(df.columns).index(found)
        st.session_state["target_col"] = st.selectbox(
            "Target column (Outcome)",
            list(df.columns),
            index=default_idx,
            help="Select your primary dependent/outcome variable for analysis and modeling."
        )

        # Quick stats metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Cols", df.shape[1])
        missing_pct = df.isna().mean().mean() * 100
        c3.metric("Missing", f"{missing_pct:.1f}%")

        # Data preview & auto-clean log
        with st.expander("🔍 Preview Data & Cleaning"):
            st.dataframe(df.head(8), use_container_width=True)
            if st.session_state["cleaning_applied"]:
                st.caption("Auto-cleaning applied:")
                for a in st.session_state["cleaning_applied"]:
                    st.caption(f"• {a}")
            else:
                st.caption("No type conversions needed.")

        # Settings
        with st.expander("⚙️ Investigation Settings"):
            max_steps = st.slider("LangGraph EDA Max Steps", 2, 5, 3, help="Controls how many sub-hypotheses the agenda explores.")

        # Downloads
        st.divider()
        st.caption("Export Artifacts:")
        d1, d2 = st.columns(2)
        with d1:
            report_path = os.path.join(OUT_DIR, "report.md")
            report_data = st.session_state.get("eda_report", "")
            if not report_data and os.path.exists(report_path):
                with open(report_path, "r", encoding="utf-8") as f:
                    report_data = f.read()
            st.download_button(
                "📥 EDA Report",
                data=report_data,
                file_name=f"EDA_Report_{st.session_state.get('target_col', 'data')}.md",
                mime="text/markdown",
                disabled=not report_data,
                use_container_width=True
            )
        with d2:
            zb = create_charts_zip()
            st.download_button(
                "📦 Step Charts",
                data=zb,
                file_name="eda_step_charts.zip",
                mime="application/zip",
                disabled=len(zb) == 0,
                use_container_width=True
            )

        # Reset & Clear
        st.divider()
        clr1, clr2 = st.columns(2)
        with clr1:
            if st.button("🧹 Clear Chat", use_container_width=True):
                st.session_state["messages"] = []
                st.session_state["eda_insights"] = []
                st.session_state["eda_report"] = ""
                st.session_state["eda_charts"] = []
                st.rerun()
        with clr2:
            if st.button("🔄 Reset All", use_container_width=True):
                for k, v in _defaults.items():
                    st.session_state[k] = v if not isinstance(v, list) else []
                st.rerun()


# ═════════════════════════════════════════════════════════════════════
# MAIN UNIFIED CHAT INTERFACE
# ═════════════════════════════════════════════════════════════════════

if st.session_state["df"] is None:
    # ── Landing / Welcome Hero Banner ────────────────────────────────
    st.markdown("## 📊 AI Data Analyst & Autonomous EDA Agent")
    st.markdown(
        "Upload any tabular dataset in the sidebar (or click **🚢 Load Titanic Demo**) "
        "and ask questions in plain English. The **AI Intent Router** automatically classifies "
        "your inquiry and routes it to the specialized analytical path."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.info(
            "### ⚡ Quick Query\n\n"
            "**Fast Path (~1–2s)**\n\n"
            "Instant charts, aggregations, cross-tabs, and analytical Q&A (e.g. imputation methods, statistical interpretations)."
        )
    with c2:
        st.info(
            "### 🧠 Deep Autonomous EDA\n\n"
            "**Multi-Agent LangGraph (~15–25s)**\n\n"
            "Decomposes complex causal inquiries into a multi-step empirical agenda, executes code, saves step charts, and writes an executive briefing."
        )
    with c3:
        st.info(
            "### 🤖 ML Strategy Advisor\n\n"
            "**Prescriptive ML (~10–15s)**\n\n"
            "Bespoke modeling advice: algorithm candidates (baseline vs. champion), validation schemes, feature engineering recipes, and metric choices."
        )

else:
    df = st.session_state["df"]
    summary_str = get_dataset_summary(df)
    cols_list = [str(c) for c in df.columns]
    target_name = st.session_state.get("target_col") or cols_list[0]

    # Context Header Pill
    eda_count = len(st.session_state.get("eda_insights", []))
    cache_tag = f" • ✅ **{eda_count} EDA insights cached**" if eda_count > 0 else ""
    st.caption(
        f"Active Dataset: **{st.session_state['df_name']}** ({df.shape[0]:,} rows × {df.shape[1]} columns) • "
        f"Target: `{target_name}`{cache_tag}"
    )

    # ── Render chat history ──────────────────────────────────────────
    for msg in st.session_state["messages"]:
        avatar = USER_AVATAR if msg["role"] == "user" else AI_AVATAR
        with st.chat_message(msg["role"], avatar=avatar):
            if msg.get("badge"):
                st.markdown(msg["badge"], unsafe_allow_html=True)
            if msg.get("text"):
                st.markdown(format_display_text(msg["text"]))
            if msg.get("table") is not None:
                st.dataframe(msg["table"], use_container_width=True)
            if msg.get("figure") is not None:
                st.pyplot(msg["figure"])
            if msg.get("code"):
                with st.expander("🔍 View Python Code"):
                    st.code(msg["code"], language="python")
            if msg.get("step_charts"):
                with st.expander("📈 View Investigation Step Charts"):
                    sc = st.columns(min(len(msg["step_charts"]), 3))
                    for i, p in enumerate(msg["step_charts"]):
                        if os.path.exists(p):
                            sc[i % 3].image(p, caption=os.path.basename(p), use_container_width=True)

    # ── Suggestion chips (shown when chat is empty) ───────────────────
    if not st.session_state["messages"]:
        st.markdown("**Suggestions:**")
        chips = build_suggestions(df, target_name)
        chip_cols = st.columns(len(chips))
        for i, text in enumerate(chips):
            with chip_cols[i]:
                if st.button(text, key=f"sug_{i}", use_container_width=True):
                    st.session_state["injected_prompt"] = text
                    st.rerun()

    # ── Chat input handling ──────────────────────────────────────────
    user_prompt = st.chat_input("Ask a question about your data (or request deep EDA / ML strategy)...")

    # Pick up injected prompt from button clicks if any
    if st.session_state.get("injected_prompt"):
        user_prompt = st.session_state["injected_prompt"]
        st.session_state["injected_prompt"] = None

    if user_prompt:
        # Append and display user message
        st.session_state["messages"].append({"role": "user", "text": user_prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(user_prompt)

        # Assistant processing with AI Intent Router
        with st.chat_message("assistant", avatar=AI_AVATAR):
            # 1. AI Intent Classification
            with st.spinner("Analyzing intent with AI Router…"):
                decision = route_query(
                    user_prompt,
                    summary_str,
                    cols_list,
                    chat_history=st.session_state["messages"]
                )

            # ── PATH 1: Quick Query ───────────────────────────────────
            if decision.intent == "quick_query":
                badge = '<span class="route-badge route-quick">⚡ Quick Query</span>'
                st.markdown(badge, unsafe_allow_html=True)

                with st.spinner("Executing analytical query…"):
                    res = run_quick_query(
                        user_prompt,
                        df,
                        summary_str,
                        chat_history=st.session_state["messages"]
                    )

                ans = res.get("text", "")
                tbl = res.get("table")
                fig = res.get("figure")
                code = res.get("code")

                if res.get("error"):
                    st.error(res["error"])
                    ans = f"Error: {res['error']}"
                else:
                    if ans:
                        st.write_stream(stream_text(ans))
                    if tbl is not None:
                        st.dataframe(tbl, use_container_width=True)
                    if fig is not None:
                        st.pyplot(fig)
                    if code:
                        with st.expander("🔍 View Python Code"):
                            st.code(code, language="python")

                st.session_state["messages"].append({
                    "role": "assistant",
                    "badge": badge,
                    "text": ans,
                    "table": tbl,
                    "figure": fig,
                    "code": code,
                    "intent": "quick_query"
                })

            # ── PATH 2: Deep Autonomous EDA (LangGraph) ───────────────
            elif decision.intent == "autonomous_eda":
                badge = '<span class="route-badge route-deep">🧠 Deep Autonomous EDA (LangGraph)</span>'
                st.markdown(badge, unsafe_allow_html=True)

                report = ""
                step_charts = []
                steps_setting = max_steps if "max_steps" in locals() else 3

                prog = st.progress(0.0)
                with st.status("Investigating via LangGraph Multi-Agent Team…", expanded=True) as status:
                    for ev in run_autonomous_eda_stream(
                        df,
                        goal=user_prompt,
                        target=target_name,
                        max_steps=steps_setting
                    ):
                        node = ev.get("node")
                        preview = ev.get("preview", "")

                        if ev.get("progress") is not None:
                            prog.progress(float(ev["progress"]))

                        if node == "profiler":
                            if ev.get("agenda"):
                                status.write("**Targeted Empirical Agenda:**")
                                for i, q in enumerate(ev["agenda"], 1):
                                    status.write(f"  {i}. {q}")
                            else:
                                status.write(f"Formulating hypotheses… {preview}")
                        elif node == "planner":
                            s = ev.get("step", 0) + 1
                            t = ev.get("total_steps", steps_setting)
                            status.write(f"**Step {s}/{t}:** {ev.get('question') or preview}")
                        elif node == "coder":
                            status.write("Generating analysis code & statistical chart script…")
                        elif node == "executor":
                            status.write("Executing in RAM & verifying group sizes (n = …)…")
                        elif node == "recorder":
                            status.write("Recording empirical findings & step chart…")
                        elif node == "reporter":
                            status.write("Synthesizing direct-answer executive briefing…")
                        elif node == "complete":
                            report = ev.get("report", "")
                            if ev.get("insights"):
                                st.session_state["eda_insights"].extend(ev["insights"])
                            st.session_state["eda_report"] = report
                            prog.progress(1.0)
                            status.update(label="Autonomous EDA Complete!", state="complete", expanded=False)

                        if ev.get("chart_path") and ev["chart_path"] not in step_charts:
                            step_charts.append(ev["chart_path"])

                st.session_state["eda_charts"] = step_charts

                if report:
                    st.write_stream(stream_text(report, delay=0.005))

                if step_charts:
                    with st.expander("📈 View Investigation Step Charts", expanded=True):
                        sc = st.columns(min(len(step_charts), 3))
                        for i, p in enumerate(step_charts):
                            if os.path.exists(p):
                                sc[i % 3].image(p, caption=os.path.basename(p), use_container_width=True)

                st.session_state["messages"].append({
                    "role": "assistant",
                    "badge": badge,
                    "text": report,
                    "step_charts": step_charts,
                    "intent": "autonomous_eda"
                })

            # ── PATH 3: ML Strategy Advisor ───────────────────────────
            elif decision.intent == "modelling":
                badge = '<span class="route-badge route-ml">🤖 ML Strategy Advisor</span>'
                st.markdown(badge, unsafe_allow_html=True)

                advice_stream = stream_modelling_advice(
                    user_prompt,
                    df,
                    target_col=target_name,
                    eda_insights=st.session_state.get("eda_insights", []),
                    chat_history=st.session_state["messages"]
                )
                advice = st.write_stream(advice_stream)

                st.session_state["messages"].append({
                    "role": "assistant",
                    "badge": badge,
                    "text": advice,
                    "intent": "modelling"
                })

            # ── PATH 4: Clarification Needed ──────────────────────────
            else:
                badge = '<span class="route-badge route-clarify">💬 Clarification</span>'
                st.markdown(badge, unsafe_allow_html=True)
                q = decision.clarification_prompt or (
                    f"Could you specify what you'd like to investigate regarding `{target_name}` or which columns you're curious about? "
                    "For example: 'Plot fare distribution', 'Why did passengers die?', or 'Which ML model is best?'"
                )
                st.write_stream(stream_text(q))

                st.session_state["messages"].append({
                    "role": "assistant",
                    "badge": badge,
                    "text": q,
                    "intent": "clarification_needed"
                })

        st.rerun()
