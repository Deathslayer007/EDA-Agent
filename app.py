"""
AI Data Analyst — Streamlit Chat Interface.
Simple full-width chat with sidebar for dataset config.
Auto-routes every query through the AI router.
"""
import io
import os
import zipfile
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from router import route_query, RoutingDecision
from quick_query import run_quick_query
from profiler import run_autonomous_eda_stream, OUT_DIR
from modelling_advisor import run_modelling_advice

# ── Avatars ──────────────────────────────────────────────────────────
USER_AVATAR = "👤"
AI_AVATAR = "🤖"

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Minimal CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Route badge pills */
    .route-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .route-quick   { background: #DEF7EC; color: #03543F; }
    .route-deep    { background: #DBEAFE; color: #1E40AF; }
    .route-ml      { background: #F3E8FF; color: #6B21A8; }
    .route-clarify { background: #FEF9C3; color: #854D0E; }

    /* Tighten chat spacing */
    .stChatMessage { margin-bottom: 4px !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state ────────────────────────────────────────────────────
_defaults = {
    "messages": [],
    "df": None,
    "df_name": "",
    "target_col": None,
    "eda_insights": [],
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


def create_charts_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        if os.path.exists(OUT_DIR):
            for f in os.listdir(OUT_DIR):
                if f.endswith(".png"):
                    z.write(os.path.join(OUT_DIR, f), arcname=f)
    return buf.getvalue()


def build_suggestions(df: pd.DataFrame, target: Optional[str]) -> list[str]:
    chips = []
    num = [c for c in df.select_dtypes("number").columns if c != target]
    cat = [c for c in df.select_dtypes(exclude="number").columns if c != target]
    if num:
        chips.append(f"Show distribution of {num[0]}")
    if cat and target:
        chips.append(f"{target} rate by {cat[0]}")
    if len(num) >= 2:
        chips.append(f"Correlation: {num[0]} vs {num[1]}")
    if target and cat:
        chips.append(f"What drives {target}?")
    return chips[:4]


# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 AI Data Analyst")

    uploaded_file = st.file_uploader(
        "Upload dataset",
        type=["csv", "tsv", "xlsx", "xls", "parquet", "json"],
    )
    if uploaded_file is not None:
        try:
            raw_df = load_dataset_file(uploaded_file)
            cleaned_df, cleaning_log = auto_clean_dataset(raw_df)
            st.session_state["df"] = cleaned_df
            st.session_state["df_name"] = uploaded_file.name
            st.session_state["cleaning_applied"] = cleaning_log
        except Exception as e:
            st.error(f"Error: {e}")

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

        # Target column
        default_idx = 0
        auto_targets = ["survived", "churn", "target", "label", "outcome", "y", "fraud", "is_fraud"]
        found = next((c for c in df.columns if c.lower() in auto_targets), None)
        if found:
            default_idx = list(df.columns).index(found)
        st.session_state["target_col"] = st.selectbox(
            "Target column", list(df.columns), index=default_idx
        )

        # Quick stats
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Cols", df.shape[1])
        missing_pct = df.isna().mean().mean() * 100
        c3.metric("Missing", f"{missing_pct:.1f}%")

        # Data preview
        with st.expander("Preview data"):
            st.dataframe(df.head(10), use_container_width=True)
            if st.session_state["cleaning_applied"]:
                st.caption("Auto-cleaned:")
                for a in st.session_state["cleaning_applied"]:
                    st.caption(f"• {a}")

        # Settings
        with st.expander("Settings"):
            model_tier = st.selectbox(
                "Model",
                ["Fast (Nemotron 3.5 Lightning)", "Premium (Nemotron 3 Ultra 550B)"],
                index=0
            )
            max_steps = st.slider("EDA steps", 3, 6, 4)

        # Downloads
        st.divider()
        d1, d2 = st.columns(2)
        with d1:
            report_path = os.path.join(OUT_DIR, "report.md")
            report_data = ""
            if os.path.exists(report_path):
                with open(report_path, "r", encoding="utf-8") as f:
                    report_data = f.read()
            st.download_button("📥 Report", data=report_data, file_name="report.md",
                               mime="text/markdown", disabled=not report_data,
                               use_container_width=True)
        with d2:
            zb = create_charts_zip()
            st.download_button("📦 Charts", data=zb, file_name="charts.zip",
                               mime="application/zip", disabled=len(zb) == 0,
                               use_container_width=True)

        # Clear buttons
        st.divider()
        clr1, clr2 = st.columns(2)
        with clr1:
            if st.button("Clear chat", use_container_width=True):
                st.session_state["messages"] = []
                st.session_state["eda_insights"] = []
                st.rerun()
        with clr2:
            if st.button("Reset all", use_container_width=True):
                for k, v in _defaults.items():
                    st.session_state[k] = v if not isinstance(v, list) else []
                st.rerun()


# ═════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ═════════════════════════════════════════════════════════════════════

if st.session_state["df"] is None:
    # ── Landing ──────────────────────────────────────────────────────
    st.markdown("## 📊 AI Data Analyst")
    st.markdown("Upload a dataset in the sidebar and ask questions in plain English. "
                "The system automatically picks the right analysis path.")

    c1, c2, c3 = st.columns(3)
    c1.info("**⚡ Quick Query**\n\nCharts, stats, and tables in seconds.")
    c2.info("**🧠 Deep EDA**\n\nBreaks down complex questions, runs multi-step analysis, writes a report.")
    c3.info("**📐 ML Advisor**\n\nAlgorithm picks, validation strategy, and feature engineering for your data.")

else:
    df = st.session_state["df"]
    summary_str = get_dataset_summary(df)
    cols_list = [str(c) for c in df.columns]
    target_name = st.session_state.get("target_col") or cols_list[0]

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
                with st.expander("View code"):
                    st.code(msg["code"], language="python")
            if msg.get("step_charts"):
                with st.expander("View step charts"):
                    sc = st.columns(min(len(msg["step_charts"]), 3))
                    for i, p in enumerate(msg["step_charts"]):
                        if os.path.exists(p):
                            sc[i % 3].image(p, caption=os.path.basename(p),
                                            use_container_width=True)

    # ── Suggestion chips (only when chat is empty) ───────────────────
    if not st.session_state["messages"]:
        st.caption("Try one of these, or type your own question:")
        chips = build_suggestions(df, target_name)
        chip_cols = st.columns(len(chips))
        for i, text in enumerate(chips):
            with chip_cols[i]:
                if st.button(text, key=f"sug_{i}", use_container_width=True):
                    st.session_state["injected_prompt"] = text
                    st.rerun()

    # ── Chat input ───────────────────────────────────────────────────
    user_prompt = st.chat_input("Ask about your data…")
    if st.session_state.get("injected_prompt"):
        user_prompt = st.session_state.pop("injected_prompt")

    # ── Handle new prompt ────────────────────────────────────────────
    if user_prompt:
        st.session_state["messages"].append({"role": "user", "text": user_prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(user_prompt)

        with st.chat_message("assistant", avatar=AI_AVATAR):
            # Auto-route
            with st.spinner("Routing…"):
                decision = route_query(
                    user_prompt, summary_str, cols_list,
                    chat_history=st.session_state["messages"]
                )

            # ── Quick Query ──────────────────────────────────────────
            if decision.intent == "quick_query":
                badge = '<span class="route-badge route-quick">⚡ Quick Query</span>'
                st.markdown(badge, unsafe_allow_html=True)

                with st.spinner("Running…"):
                    res = run_quick_query(
                        user_prompt, df, summary_str,
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
                        st.markdown(format_display_text(ans))
                    if tbl is not None:
                        st.dataframe(tbl, use_container_width=True)
                    if fig is not None:
                        st.pyplot(fig)
                    if code:
                        with st.expander("View code"):
                            st.code(code, language="python")

                st.session_state["messages"].append({
                    "role": "assistant", "badge": badge, "text": ans,
                    "table": tbl, "figure": fig, "code": code,
                    "intent": "quick_query"
                })

            # ── Deep EDA ─────────────────────────────────────────────
            elif decision.intent == "autonomous_eda":
                badge = '<span class="route-badge route-deep">🧠 Deep EDA</span>'
                st.markdown(badge, unsafe_allow_html=True)

                report = ""
                step_charts = []

                prog = st.progress(0.0)
                with st.status("Investigating…", expanded=True) as status:
                    for ev in run_autonomous_eda_stream(
                        df, goal=user_prompt, target=target_name, max_steps=4
                    ):
                        node = ev["node"]
                        preview = ev.get("preview", "")

                        if "progress" in ev:
                            prog.progress(float(ev["progress"]))

                        if node == "profiler":
                            if ev.get("agenda"):
                                status.write("**Plan:**")
                                for i, q in enumerate(ev["agenda"], 1):
                                    status.write(f"  {i}. {q}")
                            else:
                                status.write(f"Profiling… {preview}")
                        elif node == "planner":
                            s = ev.get("step", 0) + 1
                            t = ev.get("total_steps", 4)
                            status.write(f"Step {s}/{t}: {ev.get('question') or preview}")
                        elif node == "coder":
                            status.write("Writing code…")
                        elif node == "executor":
                            status.write("Running analysis…")
                        elif node == "critic":
                            status.write(f"Reviewing: {preview}")
                        elif node == "recorder":
                            status.write("Saving findings…")
                        elif node == "reporter":
                            status.write("Writing report…")
                        elif node == "complete":
                            report = ev.get("report", "")
                            if ev.get("insights"):
                                st.session_state["eda_insights"].extend(ev["insights"])
                            prog.progress(1.0)
                            status.update(label="Done", state="complete", expanded=False)

                        if ev.get("chart_path") and ev["chart_path"] not in step_charts:
                            step_charts.append(ev["chart_path"])

                if report:
                    st.markdown(report)

                if step_charts:
                    with st.expander("View step charts", expanded=True):
                        sc = st.columns(min(len(step_charts), 3))
                        for i, p in enumerate(step_charts):
                            if os.path.exists(p):
                                sc[i % 3].image(p, caption=os.path.basename(p),
                                                use_container_width=True)

                st.session_state["messages"].append({
                    "role": "assistant", "badge": badge, "text": report,
                    "step_charts": step_charts, "intent": "autonomous_eda"
                })

            # ── ML Advisor ───────────────────────────────────────────
            elif decision.intent == "modelling":
                badge = '<span class="route-badge route-ml">📐 ML Advisor</span>'
                st.markdown(badge, unsafe_allow_html=True)

                with st.spinner("Thinking…"):
                    advice = run_modelling_advice(
                        user_prompt, df, target_col=target_name,
                        eda_insights=st.session_state.get("eda_insights", []),
                        chat_history=st.session_state["messages"]
                    )

                st.markdown(advice)

                st.session_state["messages"].append({
                    "role": "assistant", "badge": badge,
                    "text": advice, "intent": "modelling"
                })

            # ── Clarification ────────────────────────────────────────
            else:
                badge = '<span class="route-badge route-clarify">💬 Clarification</span>'
                st.markdown(badge, unsafe_allow_html=True)
                q = decision.clarification_prompt or \
                    "Could you be more specific? Which columns or what kind of analysis?"
                st.markdown(q)

                st.session_state["messages"].append({
                    "role": "assistant", "badge": badge,
                    "text": q, "intent": "clarification_needed"
                })

        st.rerun()
