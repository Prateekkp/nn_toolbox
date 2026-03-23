import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import hashlib
import os
import traceback
import io
import sys
from contextlib import redirect_stdout
from dotenv import load_dotenv

load_dotenv()

# ==============================
# CONFIG
# ==============================
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL_ID = "meta/llama-3.1-70b-instruct"
MAX_FILE_SIZE_MB = 50
MAX_CHART_ROWS = 100


# ==============================
# LLM CALL
# ==============================
def call_nvidia_llm(prompt, retries=2, max_tokens=1000):
    if not NVIDIA_API_KEY:
        return "Error: NVIDIA_API_KEY not set in environment."

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": max_tokens
    }

    for attempt in range(retries + 1):
        try:
            response = requests.post(NVIDIA_API_URL, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            elif response.status_code == 429:
                import time
                time.sleep(2 ** attempt)
            else:
                return f"API Error {response.status_code}: {response.text}"
        except requests.exceptions.Timeout:
            if attempt == retries:
                return "Error: Request timed out. Please try again."
        except Exception as e:
            return f"Error: {str(e)}"

    return "Error: Max retries exceeded."


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
    summary["shape"] = df.shape
    summary["dtypes"] = df.dtypes.astype(str)
    summary["missing"] = df.isnull().sum()
    summary["missing_pct"] = (df.isnull().sum() / len(df) * 100).round(2)
    summary["duplicates"] = int(df.duplicated().sum())

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    summary["numerical"] = num_cols
    summary["categorical"] = cat_cols

    if num_cols:
        desc = df[num_cols].describe().T
        desc["skewness"] = df[num_cols].skew()
        desc["outlier_flag"] = (
            (df[num_cols] < (desc["mean"] - 3 * desc["std"])) |
            (df[num_cols] > (desc["mean"] + 3 * desc["std"]))
        ).sum()
        summary["num_stats"] = desc

    cat_stats = {}
    for col in cat_cols:
        cat_stats[col] = {
            "unique": df[col].nunique(),
            "top_value": df[col].mode()[0] if not df[col].mode().empty else "N/A",
            "top_freq": int(df[col].value_counts().iloc[0]) if df[col].value_counts().shape[0] > 0 else 0
        }
    summary["cat_stats"] = cat_stats

    if len(num_cols) >= 2:
        corr = df[num_cols].corr().abs()
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
    corr_str = summary["top_correlations"].to_string() if "top_correlations" in summary else "N/A"

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
    col_info = {col: str(dtype) for col, dtype in df.dtypes.items()}
    sample_rows = df.head(3).to_dict(orient="records")

    # Detect time-series: datetime-typed columns OR ts-related column name keywords
    ts_keywords = {"timestamp", "date", "time", "datetime", "period", "hour", "day", "month", "year"}
    col_names_lower = {c.lower() for c in df.columns}
    has_datetime_col = any(str(dtype).startswith("datetime") for dtype in df.dtypes.values)
    has_ts_keyword = bool(col_names_lower & ts_keywords)
    is_timeseries = has_datetime_col or has_ts_keyword

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
    """
    Runs LLM-generated code in a controlled namespace with df injected.
    Returns (results_dict, stdout_str, error_str).
    """
    # CRITICAL: __builtins__ must be present so import statements inside exec()
    # resolve installed packages normally. Without it, `import sklearn` raises
    # ModuleNotFoundError even when sklearn is installed.
    import builtins
    # Defensively reset the index so generated code never hits KeyError: 0
    # when the LLM uses series[0] after a dropna/ffill that gaps the index.
    df_clean = df.copy().reset_index(drop=True)
    namespace = {
        "__builtins__": builtins,
        "df": df_clean,
        "pd": pd,
        "np": np,
        "results": {}
    }

    stdout_capture = io.StringIO()
    error_str = None

    try:
        with redirect_stdout(stdout_capture):
            exec(compile(code, "<llm_generated>", "exec"), namespace)
    except Exception:
        error_str = traceback.format_exc()

    results = namespace.get("results", {})
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
    c1.metric("Model", results.get("model_name", "—"))
    c2.metric("Train Samples", results.get("train_size", "—"))
    c3.metric("Test Samples", results.get("test_size", "—"))
    c4.metric("Split Strategy", results.get("split_strategy", "—").capitalize())

    # ── Metric diagnostics ──────────────────────────────────────────────────
    metrics = results.get("metrics", {})

    # Normalise key lookup — LLM may return r2, R2, r2_score, R2_Score etc.
    metrics_lower = {k.lower(): float(v) for k, v in metrics.items()}
    r2   = metrics_lower.get("r2") or metrics_lower.get("r2_score")
    mape = metrics_lower.get("mape")

    if r2 is not None:
        if r2 < 0:
            st.error(
                f"**Negative R\u00b2 detected ({r2:.4f})** \u2014 the model is performing worse than a "
                "naive mean baseline. This is a real and honest result, not a bug. Common causes:\n\n"
                "- **Random split on time-series data** \u2014 future rows leak into training, causing "
                "the model to fail on temporally held-out test data. Re-train using a chronological "
                "split (train = first 80%, test = last 20%).\n"
                "- **Underfitting** \u2014 the selected model may be too simple for this data. "
                "Try a tree-based model such as XGBoost or Random Forest.\n"
                "- **Missing lag features** \u2014 demand / load forecasting typically requires lag and "
                "rolling-window features. Consider adding them before training."
            )
        elif r2 < 0.3:
            st.warning(
                f"**Low R\u00b2 ({r2:.4f})** \u2014 the model explains less than 30% of variance in the target. "
                "Consider feature engineering, a stronger model, or verifying the split strategy."
            )

    if mape is not None and mape > 20:
        st.warning(
            f"**High MAPE ({mape:.2f}%)** \u2014 prediction errors are large relative to actual values. "
            "Common with seasonal demand data that lacks lag or calendar features."
        )

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
        fi_series = pd.Series(fi).sort_values(ascending=False).head(20)
        st.bar_chart(fi_series)


# ==============================
# MAIN APP
# ==============================
def explore_data_page():
    st.title("Explore Your Data with AI Assistance")
    st.caption("Upload a CSV dataset for automated EDA and AI-generated insights.")

    # ── Always-visible info boxes ──────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.info(
            "**What This Tool Does**\n\n"
            "- Auto-profile datasets (missing values, distributions, correlations)\n"
            "- LLM analysis (problem type, model suggestions, data quality)\n"
            "- Auto-generate & execute custom training scripts\n"
            "- Return metrics and feature importances"
        )
    with col2:
        st.warning(
            "**Limitations**\n\n"
            "- CSV files only, max **50 MB**\n"
            "- scikit-learn based (XGBoost / LightGBM optional, no deep learning)\n"
            "- AI outputs are best-effort \u2014 review before production use\n"
            "- LLM code may need a retry if it makes incorrect data assumptions"
        )

    if not NVIDIA_API_KEY:
        st.error("NVIDIA_API_KEY not found. Add it to your .env file.")
        return

    st.divider()

    uploaded_file = st.file_uploader("Upload dataset (CSV)", type=["csv"], help="Max 50 MB per file \u2022 CSV only")

    if uploaded_file:

        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(f"File too large ({file_size_mb:.1f} MB). Maximum allowed: {MAX_FILE_SIZE_MB} MB.")
            return

        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            try:
                df = pd.read_csv(uploaded_file, encoding="latin-1")
                st.info("File loaded using latin-1 encoding.")
            except Exception as e:
                st.error(f"Could not read file: {e}")
                return
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

        # ==============================
        # ABOUT DATA
        # ==============================
        with st.expander("About Data", expanded=False):

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rows", df.shape[0])
            col2.metric("Columns", df.shape[1])
            col3.metric("Missing Cells", int(summary["missing"].sum()))
            col4.metric("Duplicate Rows", summary["duplicates"])

            st.divider()

            tab1, tab2, tab3, tab4 = st.tabs(["Column Types", "Missing Values", "Numerical Stats", "Categorical Stats"])

            with tab1:
                type_df = summary["dtypes"].reset_index()
                type_df.columns = ["Column", "Type"]
                st.dataframe(type_df, use_container_width=True)

            with tab2:
                missing_df = pd.DataFrame({
                    "Column": summary["missing"].index,
                    "Missing Count": summary["missing"].values,
                    "Missing %": summary["missing_pct"].values
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

        # ==============================
        # COLUMN DEEP DIVE
        # ==============================
        with st.expander("Column Deep Dive", expanded=False):
            selected_col = st.selectbox("Select a column", df.columns.tolist(), key="deep_dive_col")

            if selected_col:
                col_data = df[selected_col]
                c1, c2, c3 = st.columns(3)
                c1.metric("Unique Values", col_data.nunique())
                c2.metric("Missing", int(col_data.isnull().sum()))
                c3.metric("Missing %", f"{col_data.isnull().mean() * 100:.1f}%")

                if pd.api.types.is_numeric_dtype(col_data):
                    c4, c5, c6 = st.columns(3)
                    c4.metric("Mean", f"{col_data.mean():.3f}")
                    c5.metric("Std Dev", f"{col_data.std():.3f}")
                    c6.metric("Skewness", f"{col_data.skew():.3f}")

                chart_data, chart_type = get_column_chart_data(df_hash, selected_col, str(col_data.dtype), df)

                if chart_data.shape[0] == 0:
                    st.info("No data to display.")
                else:
                    if chart_type == "numeric" and col_data.nunique() > MAX_CHART_ROWS:
                        st.caption(f"Distribution shown as 50 bins (column has {col_data.nunique():,} unique values).")
                    elif chart_type == "categorical" and col_data.nunique() > MAX_CHART_ROWS:
                        st.caption(f"Showing top {MAX_CHART_ROWS} most frequent values ({col_data.nunique():,} unique total).")
                    st.bar_chart(chart_data)

        # ==============================
        # CORRELATIONS
        # ==============================
        if "top_correlations" in summary:
            with st.expander("Top Feature Correlations", expanded=False):
                st.dataframe(
                    summary["top_correlations"].style.background_gradient(
                        subset=["Correlation"], cmap="YlOrRd"
                    ),
                    use_container_width=True
                )

        # ==============================
        # AI SUMMARY
        # ==============================
        with st.expander("AI Summary", expanded=True):

            if "ai_summary_parsed" not in st.session_state:
                with st.spinner("Generating AI analysis..."):
                    prompt = build_analysis_prompt(summary, df.head().to_string())
                    raw = call_nvidia_llm(prompt)
                    st.session_state["ai_summary_raw"] = raw

                    parsed, err = parse_llm_json(raw)
                    if parsed:
                        st.session_state["ai_summary_parsed"] = parsed
                    else:
                        st.session_state["ai_summary_parsed"] = None
                        st.session_state["ai_summary_error"] = err

            if st.session_state.get("ai_summary_parsed"):
                render_ai_summary(st.session_state["ai_summary_parsed"])

                if st.download_button(
                    label="Download AI Summary",
                    data=json.dumps(st.session_state["ai_summary_parsed"], indent=2),
                    file_name="ai_summary.json",
                    mime="application/json"
                ):
                    pass
            else:
                st.warning("Could not parse structured response. Raw output:")
                st.write(st.session_state.get("ai_summary_raw", "No response."))

        # ==============================
        # TRAIN MODEL
        # ==============================
        parsed_summary = st.session_state.get("ai_summary_parsed")

        if parsed_summary:
            with st.expander("🤖 Train Model", expanded=True):

                st.markdown(
                    "The LLM will generate a **custom training script** for your dataset and selected model, "
                    "then execute it in real time and show you the results."
                )

                suggested_models = [m["name"] for m in parsed_summary.get("suggested_models", [])]
                if not suggested_models:
                    suggested_models = ["Random Forest", "Logistic Regression", "XGBoost"]

                col_left, col_right = st.columns(2)

                with col_left:
                    selected_model = st.selectbox(
                        "Select model to train",
                        options=suggested_models,
                        key="selected_model"
                    )

                with col_right:
                    all_cols = df.columns.tolist()
                    guessed_target = parsed_summary.get("target_column_guess")
                    default_idx = all_cols.index(guessed_target) if guessed_target in all_cols else 0
                    target_col = st.selectbox(
                        "Target column",
                        options=all_cols,
                        index=default_idx,
                        key="target_col"
                    )

                problem_type = parsed_summary.get("problem_type", "classification")

                st.caption(
                    f"Problem type: **{problem_type}** \u00b7 "
                    f"Dataset: **{df.shape[0]:,} rows \u00d7 {df.shape[1]} columns**"
                )

                train_btn = st.button("\u26a1 Generate Code & Train", type="primary", key="train_btn")

                if train_btn:
                    for key in ["training_results", "training_code", "training_error"]:
                        st.session_state.pop(key, None)

                    with st.spinner(f"Asking LLM to write training code for **{selected_model}**..."):
                        code_prompt = build_training_code_prompt(
                            df, summary, selected_model, problem_type, target_col
                        )
                        raw_code = call_nvidia_llm(code_prompt, max_tokens=2000)

                    # Strip accidental markdown fences
                    clean_code = raw_code.strip()
                    if clean_code.startswith("```"):
                        clean_code = "\n".join(clean_code.split("\n")[1:])
                    if clean_code.endswith("```"):
                        clean_code = "\n".join(clean_code.split("\n")[:-1])
                    clean_code = clean_code.strip()

                    st.session_state["training_code"] = clean_code

                    with st.spinner("Executing generated code..."):
                        results, stdout_str, error_str = execute_generated_code(clean_code, df)

                    st.session_state["training_results"] = results
                    st.session_state["training_error"] = error_str

                if "training_results" in st.session_state:
                    error = st.session_state.get("training_error")

                    if error:
                        st.error("The generated code raised an error during execution.")
                        with st.expander("Execution Error", expanded=True):
                            st.code(error, language="python")
                        st.info(
                            "This can happen if the LLM made an assumption about your data that didn't hold. "
                            "Try clicking **Generate Code & Train** again \u2014 the LLM may produce a corrected script."
                        )
                    else:
                        st.success("Training complete!")
                        render_training_results(st.session_state["training_results"])

                    if "training_code" in st.session_state:
                        with st.expander("View Generated Code", expanded=False):
                            st.code(st.session_state["training_code"], language="python")
                            st.download_button(
                                label="Download training script",
                                data=st.session_state["training_code"],
                                file_name=f"train_{selected_model.lower().replace(' ', '_')}.py",
                                mime="text/plain",
                                key="download_code_btn"
                            )


if __name__ == "__main__":
    st.set_page_config(page_title="AI Playground", layout="wide")
    explore_data_page()