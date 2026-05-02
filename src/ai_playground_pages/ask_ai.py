import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import hashlib
import os
import traceback
import io
from contextlib import redirect_stdout
from pathlib import Path
from dotenv import load_dotenv

def _load_env_file():
    """Load .env from the current folder or nearest parent folder."""
    start = Path(__file__).resolve().parent
    candidates = [start / ".env", *[p / ".env" for p in start.parents], Path.cwd() / ".env"]

    for env_path in candidates:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            return env_path

    # Fallback: let python-dotenv use its default discovery behavior.
    load_dotenv()
    return None


LOADED_ENV_PATH = _load_env_file()


def _get_groq_api_key():
    """Resolve API key from Streamlit secrets first, then environment variables."""
    try:
        secret_key = (st.secrets.get("GROQ_API_KEY", "") or "").strip()
    except Exception:
        secret_key = ""

    if secret_key:
        return secret_key, "streamlit_secrets"

    env_key = (os.getenv("GROQ_API_KEY", "") or "").strip()
    if env_key:
        return env_key, "environment"

    return "", "missing"

# ==============================
# CONFIG
# ==============================
GROQ_API_KEY, GROQ_API_KEY_SOURCE = _get_groq_api_key()
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_ID     = "llama-3.3-70b-versatile"  # best free Groq model
MAX_FILE_SIZE_MB = 50
MAX_CHART_ROWS   = 100


# ==============================
# LLM CALL  (Groq)
# ==============================
def call_nvidia_llm(prompt, retries=2, max_tokens=1000):
    """Kept original name to avoid touching every call-site — backed by Groq now."""
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not set. Add it to .env or paste it in the config above.")
        st.stop()

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 1.0,
        "stream": False,
    }

    for attempt in range(retries + 1):
        try:
            response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            elif response.status_code == 401:
                st.error("❌ 401 — Groq key is invalid or malformed.")
                st.caption(f"Response: {response.text}")
                st.stop()
            elif response.status_code == 403:
                st.error("❌ 403 — Groq access denied.")
                st.caption(
                    "Key exists but authorization failed. In Streamlit Cloud, set GROQ_API_KEY in App Settings -> Secrets. "
                    "If already set, rotate/regenerate the key in console.groq.com and redeploy."
                )
                st.caption(f"Response: {response.text}")
                st.stop()
            elif response.status_code == 429:
                if attempt < retries:
                    import time
                    time.sleep(2 ** attempt)
                    continue
                else:
                    st.warning("⚠️ 429 — Groq rate limit hit. Wait a moment and retry.")
                    st.stop()
            else:
                st.error(f"❌ Unexpected error (HTTP {response.status_code})")
                st.caption(f"Response: {response.text}")
                st.stop()

        except requests.exceptions.Timeout:
            if attempt == retries:
                st.error("Request timed out. Please try again.")
                st.stop()
        except requests.exceptions.RequestException as e:
            st.error(f"Network error: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()

    st.error("Max retries exceeded.")
    st.stop()


# ==============================
# ROBUST CSV LOADER
# ==============================
def load_csv_robust(uploaded_file):
    issues = []
    delimiters = [",", ";", "\t", "|", ":"]
    encodings  = ["utf-8", "latin-1", "iso-8859-1", "cp1252"]

    df             = None
    encoding_used  = None
    delimiter_used = None

    for enc in encodings:
        try:
            uploaded_file.seek(0)
            for delim in delimiters:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=enc, delimiter=delim, engine="python")
                    if df.shape[0] > 0 and df.shape[1] > 1:
                        encoding_used  = enc
                        delimiter_used = delim
                        break
                except (pd.errors.ParserError, pd.errors.EmptyDataError):
                    continue
            if df is not None and df.shape[0] > 0:
                break
        except (UnicodeDecodeError, LookupError):
            continue

    if df is None or df.shape[0] == 0:
        raise ValueError(
            "Could not parse CSV with any encoding/delimiter combination. "
            "Verify file is a valid CSV and try re-saving from Excel or Google Sheets."
        )

    if encoding_used != "utf-8":
        issues.append(f"Encoding: {encoding_used} (not UTF-8)")
    if delimiter_used != ",":
        issues.append(f"Delimiter detected: '{delimiter_used}' (not comma)")

    original_cols = df.columns.tolist()
    df.columns = [
        str(col).strip().replace("\\", "_").replace("/", "_").replace(":", "_")
        for col in df.columns
    ]
    for orig, clean in zip(original_cols, df.columns):
        if orig != clean:
            issues.append(f"Column '{orig}' → '{clean}' (cleaned special chars/whitespace)")

    return df, issues


# ==============================
# CACHE KEY
# ==============================
def get_df_hash(df):
    content = df.head(10).to_string() + str(df.shape)
    return hashlib.md5(content.encode()).hexdigest()


# ==============================
# EDA
# ==============================
@st.cache_data(show_spinner=False)
def analyze_data(df_hash_key, df):
    summary = {}
    summary["shape"]       = df.shape
    summary["dtypes"]      = df.dtypes.astype(str)
    summary["missing"]     = df.isnull().sum()
    summary["missing_pct"] = (df.isnull().sum() / len(df) * 100).round(2)
    summary["duplicates"]  = int(df.duplicated().sum())

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    summary["numerical"]   = num_cols
    summary["categorical"] = cat_cols

    if num_cols:
        desc = df[num_cols].describe().T
        desc["skewness"]     = df[num_cols].skew()
        desc["outlier_flag"] = (
            (df[num_cols] < (desc["mean"] - 3 * desc["std"])) |
            (df[num_cols] > (desc["mean"] + 3 * desc["std"]))
        ).sum()
        summary["num_stats"] = desc

    cat_stats = {}
    for col in cat_cols:
        cat_stats[col] = {
            "unique":    df[col].nunique(),
            "top_value": df[col].mode()[0] if not df[col].mode().empty else "N/A",
            "top_freq":  int(df[col].value_counts().iloc[0]) if df[col].value_counts().shape[0] > 0 else 0,
        }
    summary["cat_stats"] = cat_stats

    if len(num_cols) >= 2:
        corr  = df[num_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        top_corr = (
            upper.stack()
            .sort_values(ascending=False)
            .head(5)
            .reset_index()
        )
        top_corr.columns = ["Feature A", "Feature B", "Correlation"]
        summary["top_correlations"] = top_corr

    return summary


# ==============================
# COLUMN CHART
# ==============================
@st.cache_data(show_spinner=False)
def get_column_chart_data(df_hash_key, col_name, col_dtype, df):
    col_data = df[col_name].dropna()
    if pd.api.types.is_numeric_dtype(col_data):
        vc = col_data.value_counts().sort_index()
        if vc.shape[0] > MAX_CHART_ROWS:
            counts, bin_edges = np.histogram(col_data, bins=50)
            labels = [f"{bin_edges[i]:.2f}" for i in range(len(counts))]
            vc = pd.Series(counts, index=labels)
        return vc, "numeric"
    else:
        vc = col_data.value_counts().head(MAX_CHART_ROWS)
        return vc, "categorical"


# ==============================
# PROMPT — EDA
# ==============================
def build_analysis_prompt(summary, df_head):
    num_stats_str = summary["num_stats"].to_string() if "num_stats" in summary else "N/A"
    cat_stats_str = json.dumps(summary["cat_stats"], indent=2) if "cat_stats" in summary else "N/A"
    corr_str      = summary["top_correlations"].to_string() if "top_correlations" in summary else "N/A"

    return f"""
You are a senior data scientist. Analyze this dataset and respond ONLY with a valid JSON object.
No markdown, no backticks, no preamble. Just raw JSON.

Dataset Info:
- Shape: {summary['shape']}
- Data Types: {summary['dtypes'].to_dict()}
- Missing Values: {summary['missing'].to_dict()}
- Missing %: {summary['missing_pct'].to_dict()}
- Duplicate Rows: {summary['duplicates']}
- Numerical Columns: {summary['numerical']}
- Categorical Columns: {summary['categorical']}

Numerical Statistics (mean, std, min, max, skewness, outlier_flag):
{num_stats_str}

Categorical Statistics (unique count, top value, frequency):
{cat_stats_str}

Top Feature Correlations:
{corr_str}

Top 5 rows:
{df_head}

Respond with this exact JSON structure:
{{
  "problem_type": "classification | regression | clustering | unknown",
  "target_column_guess": "column name or null",
  "high_level_summary": "2-3 sentence overview",
  "key_observations": ["observation 1", "observation 2", "observation 3"],
  "preprocessing_steps": ["step 1", "step 2", "step 3"],
  "suggested_models": [
    {{"name": "Model Name", "reason": "why this fits"}},
    {{"name": "Model Name", "reason": "why this fits"}}
  ],
  "risks": ["risk 1", "risk 2"],
  "data_quality_score": 0-100
}}
"""


# ==============================
# PROMPT — TRAINING CODE GENERATION
# ==============================
def build_training_code_prompt(df, summary, model_name, problem_type, target_col):
    col_info    = {col: str(dtype) for col, dtype in df.dtypes.items()}
    sample_rows = df.head(3).to_dict(orient="records")

    ts_keywords      = {"timestamp", "date", "time", "datetime", "period", "hour", "day", "month", "year"}
    col_names_lower  = {c.lower() for c in df.columns}
    has_datetime_col = any(str(dtype).startswith("datetime") for dtype in df.dtypes.values)
    has_ts_keyword   = bool(col_names_lower & ts_keywords)
    is_timeseries    = has_datetime_col or has_ts_keyword

    split_instruction = (
        "IMPORTANT — this dataset appears to be a time series. "
        "Do NOT use random shuffle. Use a chronological split: "
        "train = first 80%% of rows, test = last 20%% of rows (preserve row order). "
        "This prevents data leakage from future rows into the training set."
        if is_timeseries else
        "Split data: 80%% train, 20%% test using train_test_split with random_state=42."
    )

    return f"""
You are an expert ML engineer writing production-quality Python code.

Your task: Write a complete, self-contained Python training script for the dataset described below.

DATASET CONTEXT:
- Shape: {df.shape}
- Columns and dtypes: {json.dumps(col_info)}
- Target column: {target_col}
- Problem type: {problem_type}
- Selected model: {model_name}
- Sample rows (first 3): {json.dumps(sample_rows, default=str)}
- Missing value counts: {summary['missing'].to_dict()}
- Numerical columns: {summary['numerical']}
- Categorical columns: {summary['categorical']}
- Is time-series dataset: {is_timeseries}

STRICT REQUIREMENTS:
1. The dataframe is already loaded and available as variable `df` (pandas DataFrame). Do NOT load any files.
2. Use only these libraries: pandas, numpy, sklearn (any submodule), and optionally xgboost or lightgbm if appropriate.
3. Handle missing values — for time series prefer forward-fill (df.ffill()), otherwise impute or drop.
4. CRITICAL — After ANY operation that may alter the index (dropna, ffill, filtering, concat, merge),
   you MUST call df = df.reset_index(drop=True) immediately afterward.
   Never use df[col][0] or series[0] style integer access — always use .iloc[0] instead.
5. Encode categorical columns using LabelEncoder. Drop or ignore pure ID/index columns.
6. {split_instruction}
7. Train the model and evaluate on the test set.
8. For regression: compute MAE, RMSE, R2, and MAPE (guard against division by zero with a small epsilon).
   For classification: compute accuracy, f1_weighted, precision_weighted, recall_weighted.
9. At the end, populate a dict called `results` with:
   - "model_name": string
   - "metrics": dict of metric_name -> float
   - "feature_importances": dict of feature_name -> importance_value (if model supports it, else empty dict)
   - "train_size": int
   - "test_size": int
   - "split_strategy": "chronological" or "random"
10. Print nothing to stdout (no print statements).
11. Do NOT wrap in functions or classes. Write flat, sequential code.
12. Do NOT include any markdown, backticks, comments explaining the task, or import of dotenv/streamlit/matplotlib.
13. The LAST line of code must be: results = results

Output ONLY the raw Python code. No explanation, no markdown fences.
"""


# ==============================
# EXECUTE GENERATED CODE
# ==============================
def execute_generated_code(code, df):
    import builtins
    df_clean  = df.copy().reset_index(drop=True)
    namespace = {
        "__builtins__": builtins,
        "df": df_clean,
        "pd": pd,
        "np": np,
        "results": {},
    }

    stdout_capture = io.StringIO()
    error_str      = None

    try:
        with redirect_stdout(stdout_capture):
            exec(compile(code, "<llm_generated>", "exec"), namespace)
    except Exception:
        error_str = traceback.format_exc()

    results    = namespace.get("results", {})
    stdout_str = stdout_capture.getvalue()
    return results, stdout_str, error_str


# ==============================
# PARSE LLM JSON
# ==============================
def parse_llm_json(raw):
    try:
        clean = raw.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean), None
    except json.JSONDecodeError as e:
        return None, str(e)


# ==============================
# UI — AI SUMMARY
# ==============================
def render_ai_summary(parsed):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Problem Type", parsed.get("problem_type", "—").capitalize())
    with col2:
        st.metric("Target Column (guessed)", parsed.get("target_column_guess") or "—")
    with col3:
        score = parsed.get("data_quality_score", "—")
        st.metric("Data Quality Score", f"{score}/100" if isinstance(score, int) else "—")

    st.markdown("**Summary**")
    st.write(parsed.get("high_level_summary", "—"))

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Key Observations**")
        for obs in parsed.get("key_observations", []):
            st.write(f"- {obs}")
        st.markdown("**Risks**")
        for risk in parsed.get("risks", []):
            st.write(f"- {risk}")
    with col_b:
        st.markdown("**Preprocessing Steps**")
        for i, step in enumerate(parsed.get("preprocessing_steps", []), 1):
            st.write(f"{i}. {step}")
        st.markdown("**Suggested Models**")
        for m in parsed.get("suggested_models", []):
            st.write(f"**{m['name']}** — {m['reason']}")


# ==============================
# UI — TRAINING RESULTS
# ==============================
def render_training_results(results):
    if not results:
        st.warning("No results returned from the generated code.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model",          results.get("model_name",     "—"))
    c2.metric("Train Samples",  results.get("train_size",     "—"))
    c3.metric("Test Samples",   results.get("test_size",      "—"))
    c4.metric("Split Strategy", results.get("split_strategy", "—").capitalize())

    metrics       = results.get("metrics", {})
    metrics_lower = {k.lower(): float(v) for k, v in metrics.items()}
    r2   = metrics_lower.get("r2") or metrics_lower.get("r2_score")
    mape = metrics_lower.get("mape")

    if r2 is not None:
        if r2 < 0:
            st.error(
                f"**Negative R² detected ({r2:.4f})** — model is worse than a naive mean baseline.\n\n"
                "- **Random split on time-series** — use chronological split instead.\n"
                "- **Underfitting** — try XGBoost or Random Forest.\n"
                "- **Missing lag features** — demand forecasting needs lag/rolling features."
            )
        elif r2 < 0.3:
            st.warning(f"**Low R² ({r2:.4f})** — model explains less than 30% of variance.")

    if mape is not None and mape > 20:
        st.warning(f"**High MAPE ({mape:.2f}%)** — large prediction errors relative to actuals.")

    st.markdown("#### Evaluation Metrics")
    if metrics:
        metric_df = pd.DataFrame(
            [(k.upper(), round(float(v), 4)) for k, v in metrics.items()],
            columns=["Metric", "Value"]
        )
        st.dataframe(metric_df, use_container_width=True, hide_index=True)
    else:
        st.info("No metrics returned.")

    fi = results.get("feature_importances", {})
    if fi:
        st.markdown("#### Feature Importances")
        st.bar_chart(pd.Series(fi).sort_values(ascending=False).head(20))


# ==============================
# MAIN APP
# ==============================
def explore_data_page():
    st.title("Explore Your Data with AI Assistance")
    st.caption("Upload a CSV dataset for automated EDA and AI-generated insights.")

    if not GROQ_API_KEY:
        st.error("❌ GROQ_API_KEY not found or not set.")
        if GROQ_API_KEY_SOURCE == "streamlit_secrets":
            st.caption("Key source: Streamlit Secrets")
        elif GROQ_API_KEY_SOURCE == "environment":
            st.caption("Key source: Environment variable")
            if LOADED_ENV_PATH is not None:
                st.caption(f"Detected .env at: {LOADED_ENV_PATH}")
        else:
            st.caption("No GROQ_API_KEY found in Streamlit Secrets or environment variables.")
        st.markdown(
            """
            **How to fix:**
            1. Go to [console.groq.com](https://console.groq.com) → API Keys → Create key
            2. For Streamlit Cloud: App Settings → Secrets, add:
               ```toml
               GROQ_API_KEY = "gsk_xxxxxxxxxxxx"
               ```
            3. For local runs: add `GROQ_API_KEY=gsk_xxxxxxxxxxxx` to project `.env`
            4. Restart/redeploy the app
            """
        )
        return

    col1, col2 = st.columns(2)
    with col1:
        st.info(
            "**What This Tool Does**\n\n"
            "- Auto-profile datasets (missing values, distributions, correlations)\n"
            "- LLM analysis (problem type, model suggestions, data quality)"
        )
    with col2:
        st.warning(
            "**Limitations**\n\n"
            "- CSV files only, max **50 MB**\n"
            "- scikit-learn based (XGBoost / LightGBM optional, no deep learning)\n"
            "- AI outputs are best-effort — review before production use\n"
            "- LLM code may need a retry if it makes incorrect data assumptions"
        )

    st.divider()

    uploaded_file = st.file_uploader("Upload dataset (CSV)", type=["csv"], help="Max 50 MB • CSV only")

    if uploaded_file:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(f"File too large ({file_size_mb:.1f} MB). Maximum: {MAX_FILE_SIZE_MB} MB.")
            return

        try:
            df, load_issues = load_csv_robust(uploaded_file)
            if load_issues:
                with st.expander("Load Details", expanded=False):
                    for issue in load_issues:
                        st.caption(f"ℹ️ {issue}")
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")
            return

        df_hash = get_df_hash(df)

        if st.session_state.get("df_hash") != df_hash:
            for key in ["ai_summary_raw", "ai_summary_parsed", "ai_summary_error",
                        "training_results", "training_code", "training_error"]:
                st.session_state.pop(key, None)
            st.session_state["df_hash"] = df_hash

        st.subheader("Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        summary = analyze_data(df_hash, df)

        # ── About Data ────────────────────────────────────────────────────────
        with st.expander("About Data", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rows",           df.shape[0])
            col2.metric("Columns",        df.shape[1])
            col3.metric("Missing Cells",  int(summary["missing"].sum()))
            col4.metric("Duplicate Rows", summary["duplicates"])
            st.divider()

            tab1, tab2, tab3, tab4 = st.tabs(["Column Types", "Missing Values", "Numerical Stats", "Categorical Stats"])
            with tab1:
                type_df = summary["dtypes"].reset_index()
                type_df.columns = ["Column", "Type"]
                st.dataframe(type_df, use_container_width=True)
            with tab2:
                missing_df = pd.DataFrame({
                    "Column":        summary["missing"].index,
                    "Missing Count": summary["missing"].values,
                    "Missing %":     summary["missing_pct"].values,
                })
                missing_df = missing_df[missing_df["Missing Count"] > 0]
                if missing_df.empty:
                    st.success("No missing values found.")
                else:
                    st.dataframe(missing_df, use_container_width=True)
            with tab3:
                if "num_stats" in summary:
                    st.dataframe(summary["num_stats"].round(3), use_container_width=True)
                else:
                    st.info("No numerical columns found.")
            with tab4:
                if summary["cat_stats"]:
                    cat_df = pd.DataFrame(summary["cat_stats"]).T.reset_index()
                    cat_df.columns = ["Column", "Unique Values", "Top Value", "Top Frequency"]
                    st.dataframe(cat_df, use_container_width=True)
                else:
                    st.info("No categorical columns found.")

        # ── Column Deep Dive ──────────────────────────────────────────────────
        with st.expander("Column Deep Dive", expanded=False):
            selected_col = st.selectbox("Select a column", df.columns.tolist(), key="deep_dive_col")
            if selected_col:
                col_data = df[selected_col]
                c1, c2, c3 = st.columns(3)
                c1.metric("Unique Values", col_data.nunique())
                c2.metric("Missing",       int(col_data.isnull().sum()))
                c3.metric("Missing %",     f"{col_data.isnull().mean() * 100:.1f}%")

                if pd.api.types.is_numeric_dtype(col_data):
                    c4, c5, c6 = st.columns(3)
                    c4.metric("Mean",     f"{col_data.mean():.3f}")
                    c5.metric("Std Dev",  f"{col_data.std():.3f}")
                    c6.metric("Skewness", f"{col_data.skew():.3f}")

                chart_data, chart_type = get_column_chart_data(df_hash, selected_col, str(col_data.dtype), df)
                if chart_data.shape[0] == 0:
                    st.info("No data to display.")
                else:
                    if chart_type == "numeric" and col_data.nunique() > MAX_CHART_ROWS:
                        st.caption(f"Distribution shown as 50 bins ({col_data.nunique():,} unique values).")
                    elif chart_type == "categorical" and col_data.nunique() > MAX_CHART_ROWS:
                        st.caption(f"Top {MAX_CHART_ROWS} of {col_data.nunique():,} unique values shown.")
                    st.bar_chart(chart_data)

        # ── Correlations ──────────────────────────────────────────────────────
        if "top_correlations" in summary:
            with st.expander("Top Feature Correlations", expanded=False):
                st.dataframe(
                    summary["top_correlations"].style.background_gradient(subset=["Correlation"], cmap="YlOrRd"),
                    use_container_width=True,
                )

        # ── AI Summary ────────────────────────────────────────────────────────
        with st.expander("AI Summary", expanded=True):
            if "ai_summary_parsed" not in st.session_state:
                with st.spinner("Generating AI analysis..."):
                    prompt = build_analysis_prompt(summary, df.head().to_string())
                    raw    = call_nvidia_llm(prompt)
                    st.session_state["ai_summary_raw"] = raw
                    parsed, err = parse_llm_json(raw)
                    if parsed:
                        st.session_state["ai_summary_parsed"] = parsed
                    else:
                        st.session_state["ai_summary_parsed"] = None
                        st.session_state["ai_summary_error"]  = err

            if st.session_state.get("ai_summary_parsed"):
                render_ai_summary(st.session_state["ai_summary_parsed"])
                st.download_button(
                    label="Download AI Summary",
                    data=json.dumps(st.session_state["ai_summary_parsed"], indent=2),
                    file_name="ai_summary.json",
                    mime="application/json",
                )
            else:
                st.warning("Could not parse structured response. Raw output:")
                st.write(st.session_state.get("ai_summary_raw", "No response."))

if __name__ == "__main__":
    st.set_page_config(page_title="AI Playground", layout="wide")
    explore_data_page()