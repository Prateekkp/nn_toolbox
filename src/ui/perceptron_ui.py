import streamlit as st
import random
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

GATES = {
    "AND": {"data": [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)], "separable": True},
    "OR":  {"data": [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)], "separable": True},
    "XOR": {"data": [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)], "separable": False},
}

# ══════════════════════════════════════════════════════════════════════════════
# DATASET VALIDATION — Full edge case handler
# ══════════════════════════════════════════════════════════════════════════════

class DatasetValidationResult:
    def __init__(self):
        self.valid = False
        self.X = None          # np.ndarray (n_samples, n_features)
        self.y = None          # np.ndarray (n_samples,)
        self.n_features = 0
        self.n_samples = 0
        self.feature_cols = []
        self.target_col = ""
        self.warnings = []     # non-fatal issues shown to user
        self.errors = []       # fatal issues — block training
        self.info = []         # informational notices


def validate_dataset(df, feature_cols, target_col):
    """
    Validate uploaded dataframe against all known edge cases.
    Returns a DatasetValidationResult with rich diagnostics.
    """
    result = DatasetValidationResult()
    result.feature_cols = feature_cols
    result.target_col = target_col

    # ── 1. Empty dataframe ───────────────────────────────────────────────────
    if df.empty:
        result.errors.append("The uploaded CSV is empty.")
        return result

    # ── 2. Feature columns exist ─────────────────────────────────────────────
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        result.errors.append(f"Columns not found in CSV: {missing_cols}")
        return result

    if target_col not in df.columns:
        result.errors.append(f"Target column '{target_col}' not found in CSV.")
        return result

    # ── 3. No feature columns selected ──────────────────────────────────────
    if len(feature_cols) == 0:
        result.errors.append("Please select at least one feature column.")
        return result

    # ── 4. Feature = Target overlap ──────────────────────────────────────────
    if target_col in feature_cols:
        result.errors.append("Target column cannot also be a feature column.")
        return result

    # ── 5. Extract and clean ──────────────────────────────────────────────────
    subset_cols = feature_cols + [target_col]
    df_clean = df[subset_cols].copy()

    # ── 6. Missing values ────────────────────────────────────────────────────
    n_before = len(df_clean)
    df_clean = df_clean.dropna()
    n_after = len(df_clean)
    dropped = n_before - n_after

    if n_after == 0:
        result.errors.append("All rows have missing values. No data to train on.")
        return result
    if dropped > 0:
        result.warnings.append(
            f"{dropped} row(s) with missing values were dropped. "
            f"Proceeding with {n_after} rows."
        )

    # ── 7. Minimum samples ───────────────────────────────────────────────────
    if n_after < 4:
        result.errors.append(
            f"Only {n_after} valid rows found. Need at least 4 samples to train."
        )
        return result

    # ── 8. Non-numeric features ──────────────────────────────────────────────
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df_clean[col]):
            result.errors.append(
                f"Feature column '{col}' contains non-numeric values. "
                f"Perceptron requires numeric inputs. Encode categoricals first."
            )
            return result

    # ── 9. Non-numeric target ────────────────────────────────────────────────
    if not pd.api.types.is_numeric_dtype(df_clean[target_col]):
        result.errors.append(
            f"Target column '{target_col}' is non-numeric. "
            f"Please encode class labels as integers (0 and 1)."
        )
        return result

    # ── 10. Target multiclass check ───────────────────────────────────────────
    unique_targets = sorted(df_clean[target_col].unique())
    n_classes = len(unique_targets)

    if n_classes == 1:
        result.errors.append(
            f"Target column has only one unique value ({unique_targets[0]}). "
            f"Need exactly 2 classes for binary classification."
        )
        return result

    if n_classes > 2:
        result.errors.append(
            f"Multiclass target detected ({n_classes} classes: {unique_targets}). "
            f"A single perceptron supports binary classification only. "
            f"Tip: For multiclass problems, use One-vs-Rest with multiple perceptrons "
            f"or the MLP section of this toolbox."
        )
        return result

    # ── 11. Target values not 0/1 — auto remap ────────────────────────────────
    if set(unique_targets) != {0, 1}:
        if set(unique_targets) == {-1, 1}:
            df_clean[target_col] = df_clean[target_col].map({-1: 0, 1: 1})
            result.warnings.append(
                "Target values {-1, 1} automatically remapped to {0, 1}."
            )
        else:
            label_map = {unique_targets[0]: 0, unique_targets[1]: 1}
            df_clean[target_col] = df_clean[target_col].map(label_map)
            result.warnings.append(
                f"Target labels {unique_targets} auto-encoded: "
                f"{unique_targets[0]} → 0, {unique_targets[1]} → 1."
            )

    # ── 12. Class imbalance check ─────────────────────────────────────────────
    class_counts = df_clean[target_col].value_counts()
    minority = class_counts.min()
    majority = class_counts.max()
    imbalance_ratio = majority / minority if minority > 0 else float("inf")

    if imbalance_ratio > 5:
        result.warnings.append(
            f"Severe class imbalance (ratio {imbalance_ratio:.1f}:1). "
            f"Class 0: {class_counts.get(0, 0)}, Class 1: {class_counts.get(1, 0)}. "
            f"Accuracy may be misleading — consider resampling your dataset."
        )
    elif imbalance_ratio > 2:
        result.warnings.append(
            f"Mild class imbalance (ratio {imbalance_ratio:.1f}:1). "
            f"Class 0: {class_counts.get(0, 0)}, Class 1: {class_counts.get(1, 0)}."
        )

    # ── 13. Constant feature columns (zero variance) ──────────────────────────
    for col in feature_cols:
        if df_clean[col].nunique() == 1:
            result.warnings.append(
                f"Feature '{col}' has a constant value ({df_clean[col].iloc[0]}). "
                f"It contributes no information — consider removing it."
            )

    # ── 14. Duplicate rows ────────────────────────────────────────────────────
    n_duplicates = df_clean.duplicated().sum()
    if n_duplicates > 0:
        result.warnings.append(
            f"{n_duplicates} duplicate row(s) found and included in training."
        )

    # ── 15. Contradictory labels (same features, different labels) ────────────
    feature_df = df_clean[feature_cols]
    dup_mask = feature_df.duplicated(keep=False)
    if dup_mask.any():
        label_groups = df_clean[dup_mask].groupby(feature_cols)[target_col].nunique()
        contradictions = int((label_groups > 1).sum())
        if contradictions > 0:
            result.warnings.append(
                f"{contradictions} contradictory sample(s): same features mapped to "
                f"different labels. Dataset may not be linearly separable."
            )

    # ── 16. Feature scale warning ─────────────────────────────────────────────
    large_scale_cols = []
    for col in feature_cols:
        col_range = df_clean[col].max() - df_clean[col].min()
        if col_range > 100:
            large_scale_cols.append(col)
    if large_scale_cols:
        result.warnings.append(
            f"Features {large_scale_cols} have large value ranges. "
            f"Consider normalizing (min-max or z-score) for faster convergence."
        )

    # ── 17. Large dataset notice ──────────────────────────────────────────────
    if n_after > 10_000:
        result.info.append(
            f"Large dataset ({n_after:,} rows). Training may take a moment."
        )

    # ── 18. Single feature info ───────────────────────────────────────────────
    if len(feature_cols) == 1:
        result.info.append(
            "Single feature selected. Decision boundary is a threshold on 1 axis."
        )

    # ── 19. High-dimensional notice ───────────────────────────────────────────
    if len(feature_cols) > 2:
        result.info.append(
            f"{len(feature_cols)} features selected. "
            f"Decision boundary cannot be visualized directly in 2D — "
            f"per-sample prediction table will be shown instead."
        )

    # ── All checks passed ─────────────────────────────────────────────────────
    X = df_clean[feature_cols].values.astype(float)
    y = df_clean[target_col].values.astype(int)

    result.valid = True
    result.X = X
    result.y = y
    result.n_features = X.shape[1]
    result.n_samples = X.shape[0]

    return result


# ══════════════════════════════════════════════════════════════════════════════
# PERCEPTRON TRAINING — N-feature, vectorized
# ══════════════════════════════════════════════════════════════════════════════

def train_perceptron(X, y, weights, bias, learning_rate, epochs):
    """
    Train a perceptron on N-feature binary data.
    Returns: weights, bias, losses, log_lines, converged, converged_epoch
    """
    w = weights.copy().astype(float)
    b = float(bias)
    losses = []
    log_lines = []
    converged = False
    converged_epoch = -1

    w_str = "  ".join([f"w{i+1}={v:.4f}" for i, v in enumerate(w)])
    log_lines.append(f"Initializing...\n   {w_str}  b={b:.4f}\n{'─'*60}\n")

    log_interval = max(1, epochs // 200)

    for epoch in range(epochs):
        total_error = 0
        for xi, yi in zip(X, y):
            y_pred = 1 if (np.dot(w, xi) + b) >= 0 else 0
            error = int(yi) - y_pred
            total_error += abs(error)
            if error != 0:
                w += learning_rate * error * xi
                b += learning_rate * error

        losses.append(total_error)

        if epoch % log_interval == 0 or epoch == epochs - 1:
            w_str = "  ".join([f"w{i+1}={v:.4f}" for i, v in enumerate(w)])
            log_lines.append(
                f"Epoch {epoch+1:>4}/{epochs}  |  "
                f"Loss: {total_error:.4f}  |  {w_str}  b={b:.4f}\n"
            )

        if total_error == 0:
            converged = True
            converged_epoch = epoch + 1
            log_lines.append(f"{'─'*60}\n")
            log_lines.append(f"Converged at epoch {converged_epoch}! Loss = 0\n")
            break

    log_lines.append(f"{'─'*60}\n")
    w_str = "  ".join([f"w{i+1}={v:.4f}" for i, v in enumerate(w)])
    log_lines.append(
        f"Training complete. Final Loss: {losses[-1]:.4f}\n"
        f"   {w_str}  b={b:.4f}\n"
    )

    return w, b, losses, log_lines, converged, converged_epoch


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def compute_accuracy(X, y, w, b):
    preds = (X @ w + b >= 0).astype(int)
    correct = int(np.sum(preds == y))
    return correct, len(y)


def plot_loss_curve(losses):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(losses) + 1)), y=losses,
        mode="lines", line=dict(color="#2563EB", width=2.5), name="Total Error",
        fill="tozeroy", fillcolor="rgba(37,99,235,0.08)"
    ))
    fig.update_layout(
        title=dict(text="Training Loss per Epoch", font=dict(size=15, color="#111827")),
        xaxis=dict(title="Epoch", gridcolor="#E5E7EB", linecolor="#D1D5DB", color="#374151"),
        yaxis=dict(title="Total Absolute Error", gridcolor="#E5E7EB", linecolor="#D1D5DB", color="#374151"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#F9FAFB",
        font=dict(color="#374151"),
        margin=dict(t=50, b=40, l=50, r=20),
    )
    return fig


def plot_decision_boundary_2d(X, y, w, b, feature_cols, title="Decision Boundary"):
    df = pd.DataFrame(X, columns=feature_cols)
    df["Label"] = np.where(y == 0, "Class 0", "Class 1")
    fig = go.Figure()
    for label, color in [("Class 0", "#EF4444"), ("Class 1", "#16A34A")]:
        s = df[df["Label"] == label]
        fig.add_trace(go.Scatter(
            x=s[feature_cols[0]], y=s[feature_cols[1]],
            mode="markers",
            marker=dict(size=14, color=color, line=dict(width=2, color="#111827")),
            name=label
        ))
    w1, w2 = w[0], w[1]
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    if abs(w2) > 1e-9:
        x1_range = np.linspace(x1_min, x1_max, 300)
        x2_boundary = -(w1 * x1_range + b) / w2
        fig.add_trace(go.Scatter(
            x=x1_range, y=x2_boundary, mode="lines",
            line=dict(color="#D97706", width=2.5, dash="dash"),
            name="Decision Boundary"
        ))
    elif abs(w1) > 1e-9:
        fig.add_vline(x=-b/w1, line=dict(color="#D97706", width=2.5, dash="dash"),
                      annotation_text="Boundary")
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#111827")),
        xaxis=dict(title=feature_cols[0], gridcolor="#E5E7EB", linecolor="#D1D5DB", color="#374151"),
        yaxis=dict(title=feature_cols[1], gridcolor="#E5E7EB", linecolor="#D1D5DB", color="#374151"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#F9FAFB",
        font=dict(color="#374151"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=60, b=40, l=50, r=20),
    )
    return fig


def plot_1d_threshold(X, y, w, b, feature_cols):
    threshold = -b / w[0] if abs(w[0]) > 1e-9 else None
    colors = ["#EF4444" if yi == 0 else "#16A34A" for yi in y]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=X[:, 0], y=np.zeros(len(X)), mode="markers",
        marker=dict(size=14, color=colors, line=dict(width=2, color="#111827")),
        name="Samples"
    ))
    if threshold is not None:
        fig.add_vline(x=threshold, line=dict(color="#D97706", width=2.5, dash="dash"),
                      annotation_text=f"Threshold = {threshold:.4f}",
                      annotation_font_color="#D97706")
    fig.update_layout(
        title=dict(text="1D Decision Threshold", font=dict(size=15, color="#111827")),
        xaxis=dict(title=feature_cols[0], gridcolor="#E5E7EB", linecolor="#D1D5DB", color="#374151"),
        yaxis=dict(visible=False),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#F9FAFB",
        font=dict(color="#374151"),
        margin=dict(t=50, b=40, l=50, r=20),
    )
    return fig


def build_prediction_table(X, y, w, b, feature_cols):
    preds = (X @ w + b >= 0).astype(int)
    df = pd.DataFrame(X, columns=feature_cols)
    df["Actual"] = y
    df["Predicted"] = preds
    df["Correct"] = np.where(df["Actual"] == df["Predicted"], "✅", "❌")
    return df


def render_log(placeholder, lines):
    combined = "".join(lines)
    placeholder.markdown(
        f"""<div style="background-color:#0e1117;color:#00ff88;
            font-family:'Courier New',monospace;font-size:12.5px;
            padding:12px 16px;border-radius:8px;height:300px;
            overflow-y:auto;white-space:pre;border:1px solid #2a2a2a;
        ">{combined}</div>""",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "weights": None, "bias": None,
        "losses": [], "trained": False,
        "training_log": [], "converged": False,
        "converged_epoch": -1, "last_gate": None,
        "n_features": 2, "feature_cols": ["X1", "X2"],
        "X_train": None, "y_train": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _reset_state():
    for k in ["weights", "bias", "losses", "trained", "training_log",
              "converged", "converged_epoch", "X_train", "y_train"]:
        if k in st.session_state:
            del st.session_state[k]
    _init_state()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════

def perceptron_page():
    st.title("Perceptron — Binary Linear Classifier")
    st.caption(
        "Learns a linear boundary via: **ŷ = step(w·x + b)** "
        "and updates: **Δw = η(y−ŷ)x**, **Δb = η(y−ŷ)**"
    )

    _init_state()

    X, y, feature_cols, gate_name = None, None, ["X1", "X2"], "Custom"
    data_ready = False

    # ══════════════════════════════════════════════════════════════════════════
    # DATA SOURCE
    # ══════════════════════════════════════════════════════════════════════════
    data_source = st.radio("Select Data Source", ["Logic Gate", "Upload CSV"], horizontal=True)

    # ── LOGIC GATE ─────────────────────────────────────────────────────────────
    if data_source == "Logic Gate":
        gate_name = st.selectbox("Select Logic Gate", list(GATES.keys()))
        gate_info = GATES[gate_name]
        raw = gate_info["data"]
        X = np.array([[r[0], r[1]] for r in raw], dtype=float)
        y = np.array([r[2] for r in raw], dtype=int)
        feature_cols = ["X1", "X2"]
        data_ready = True

        if not gate_info["separable"]:
            st.warning(
                "**XOR is not linearly separable.** A single perceptron cannot "
                "achieve zero error. This is a classic demonstration of perceptron limits — "
                "XOR requires a multi-layer network (MLP) to solve."
            )

        if st.session_state.last_gate != gate_name:
            st.session_state.last_gate = gate_name
            _reset_state()

        st.subheader(f"{gate_name} Truth Table")
        st.dataframe(
            pd.DataFrame(raw, columns=["X1", "X2", "Output"]),
            hide_index=True, use_container_width=True
        )

    # ── CSV UPLOAD ─────────────────────────────────────────────────────────────
    else:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded_file is None:
            st.info(
                "Upload a CSV with numeric features and a binary target column. "
                "Supports any number of features. Target must have exactly 2 classes."
            )
        else:
            # Parse
            try:
                df_uploaded = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Could not parse CSV: {e}")
                return

            # Shape guard
            if df_uploaded.empty:
                st.error("The uploaded CSV is empty.")
                return
            if len(df_uploaded.columns) < 2:
                st.error("CSV must have at least 2 columns (1 feature + 1 target).")
                return

            st.subheader("Data Preview")
            st.dataframe(df_uploaded.head(10), hide_index=True, use_container_width=True)

            with st.expander("Dataset Statistics", expanded=False):
                st.write(f"**Shape:** {df_uploaded.shape[0]} rows × {df_uploaded.shape[1]} columns")
                st.dataframe(df_uploaded.describe().round(4), use_container_width=True)

            # Column selection
            st.subheader("Column Selection")
            all_cols = list(df_uploaded.columns)

            col_a, col_b = st.columns(2)
            with col_a:
                feature_cols_sel = st.multiselect(
                    "Feature columns (X) — one or more",
                    all_cols,
                    help="Numeric input columns for the perceptron."
                )
            with col_b:
                remaining = [c for c in all_cols if c not in feature_cols_sel]
                target_col_sel = st.selectbox(
                    "Target column (y) — binary",
                    ["— select —"] + remaining,
                    help="Binary output column. Must have exactly 2 unique classes."
                )

            if not feature_cols_sel or target_col_sel == "— select —":
                st.info("Select feature and target columns above to continue.")
            else:
                # Validate
                vr = validate_dataset(df_uploaded, feature_cols_sel, target_col_sel)

                for err in vr.errors:
                    st.error(f"❌ {err}")
                for warn in vr.warnings:
                    st.warning(f"⚠️ {warn}")
                for info in vr.info:
                    st.info(f"ℹ️ {info}")

                if vr.valid:
                    X, y = vr.X, vr.y
                    feature_cols = vr.feature_cols
                    data_ready = True
                    st.success(
                        f"✅ Dataset ready — **{vr.n_samples} samples**, "
                        f"**{vr.n_features} feature(s)**, "
                        f"Class 0: **{int(np.sum(y==0))}**, Class 1: **{int(np.sum(y==1))}**"
                    )

    # ══════════════════════════════════════════════════════════════════════════
    # HYPERPARAMETERS
    # ══════════════════════════════════════════════════════════════════════════
    if not data_ready:
        return

    st.divider()
    st.subheader("Hyperparameters")
    n_features = X.shape[1]

    c1, c2 = st.columns(2)
    with c1:
        learning_rate = st.number_input(
            "Learning Rate (η)", value=0.1, min_value=0.0001,
            max_value=1.0, step=0.01, format="%.4f"
        )
    with c2:
        epochs = st.slider("Maximum Epochs", 10, 2000, 100)

    mode = st.radio("Weight Initialization", ["Random", "Manual"], horizontal=True)

    weights_init = np.zeros(n_features)
    bias_init = 0.0

    if mode == "Manual":
        with st.expander("Manual Weight Input", expanded=True):
            input_cols = st.columns(min(n_features + 1, 4))
            for i in range(n_features):
                weights_init[i] = input_cols[i % len(input_cols)].number_input(
                    f"w{i+1} ({feature_cols[i]})", value=0.0,
                    min_value=-1.0, max_value=1.0,
                    step=0.1, format="%.4f", key=f"mw_{i}"
                )
            bias_init = input_cols[n_features % len(input_cols)].number_input(
                "Bias (b)", value=0.0,
                min_value=-1.0, max_value=1.0,
                step=0.1, format="%.4f", key="mb"
            )
    else:
        st.info("Weights initialized from Uniform(−1, 1) at each training run.")

    # ══════════════════════════════════════════════════════════════════════════
    # TRAIN
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()

    btn_col, reset_col = st.columns([4, 1])
    with reset_col:
        if st.button("🔄 Reset", use_container_width=True):
            _reset_state()
            st.rerun()

    log_expander = st.expander("📋 Live Training Log", expanded=False)
    with log_expander:
        log_placeholder = st.empty()

    if st.session_state.training_log:
        render_log(log_placeholder, st.session_state.training_log)

    with btn_col:
        train_clicked = st.button(
            "▶ Train Perceptron", type="primary", use_container_width=True
        )

    if train_clicked:
        if mode == "Random":
            w_init = np.random.uniform(-1, 1, n_features)
            b_init_val = random.uniform(-1, 1)
        else:
            w_init = weights_init.copy()
            b_init_val = float(bias_init)

        w_final, b_final, losses, log_lines, converged, conv_epoch = train_perceptron(
            X, y, w_init, b_init_val, learning_rate, epochs
        )

        st.session_state.weights         = w_final
        st.session_state.bias            = b_final
        st.session_state.losses          = losses
        st.session_state.trained         = True
        st.session_state.training_log    = log_lines
        st.session_state.converged       = converged
        st.session_state.converged_epoch = conv_epoch
        st.session_state.n_features      = n_features
        st.session_state.feature_cols    = feature_cols
        st.session_state.X_train         = X
        st.session_state.y_train         = y

        render_log(log_placeholder, log_lines)

        if converged:
            st.success(f"✅ Converged at epoch **{conv_epoch}** with zero error!")
        else:
            st.warning(
                f"⚠️ Did not converge in {epochs} epochs — "
                f"final loss: **{losses[-1]:.4f}**. "
                f"Dataset may not be linearly separable."
            )

    # ══════════════════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════════════════
    if not (st.session_state.trained and st.session_state.weights is not None):
        return

    st.divider()
    st.subheader("Results")

    w      = st.session_state.weights
    b      = st.session_state.bias
    X_tr   = st.session_state.X_train
    y_tr   = st.session_state.y_train
    fcols  = st.session_state.feature_cols
    n_feat = st.session_state.n_features

    correct, total = compute_accuracy(X_tr, y_tr, w, b)
    accuracy = correct / total * 100

    # Metric cards — up to 6 per row
    all_metrics = [(f"w{i+1} ({fcols[i]})", f"{w[i]:.4f}") for i in range(n_feat)]
    all_metrics += [("Bias (b)", f"{b:.4f}"), ("Accuracy", f"{accuracy:.1f}%"), ("Correct", f"{correct}/{total}")]

    for chunk_start in range(0, len(all_metrics), 5):
        chunk = all_metrics[chunk_start:chunk_start+5]
        cols = st.columns(len(chunk))
        for col, (label, val) in zip(cols, chunk):
            col.metric(label, val)

    # Tabs
    tab_loss, tab_boundary, tab_preds = st.tabs([
        "📉 Loss Curve", "📊 Decision Boundary", "🔍 Predictions"
    ])

    with tab_loss:
        st.plotly_chart(plot_loss_curve(st.session_state.losses), use_container_width=True)

    with tab_boundary:
        if n_feat == 1:
            st.plotly_chart(plot_1d_threshold(X_tr, y_tr, w, b, fcols), use_container_width=True)
        elif n_feat == 2:
            st.plotly_chart(
                plot_decision_boundary_2d(X_tr, y_tr, w, b, fcols, title=f"{gate_name} — Decision Boundary"),
                use_container_width=True
            )
            st.caption("Dashed line: **w1·x1 + w2·x2 + b = 0**")
        else:
            st.info(
                f"Decision boundary cannot be plotted directly for {n_feat} features. "
                f"Per-sample predictions are shown in the Predictions tab."
            )

    with tab_preds:
        df_preds = build_prediction_table(X_tr, y_tr, w, b, fcols)
        st.dataframe(df_preds, hide_index=True, use_container_width=True)
        st.caption(f"Accuracy: {correct}/{total} = {accuracy:.1f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # LIVE PREDICTION
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("Try a Prediction")
    st.caption("Enter input values to test the trained perceptron.")

    # Step size: binary (gate or binary CSV cols) → 1, continuous CSV → 0.01
    is_binary_data = data_source == "Logic Gate" or (
        X_tr is not None and np.all(np.isin(X_tr, [0, 1]))
    )

    pred_cols_ui = st.columns(min(n_feat, 4))
    pred_inputs = []
    for i in range(n_feat):
        col_idx = i % len(pred_cols_ui)
        if data_source == "Logic Gate":
            # Hard-lock to 0 or 1 only
            val = pred_cols_ui[col_idx].selectbox(
                fcols[i], options=[0, 1], key=f"pi_{i}"
            )
        elif is_binary_data:
            val = pred_cols_ui[col_idx].number_input(
                fcols[i], value=0.0, min_value=0.0, max_value=1.0,
                step=1.0, format="%.0f", key=f"pi_{i}"
            )
        else:
            val = pred_cols_ui[col_idx].number_input(
                fcols[i], value=0.0, step=0.01, format="%.4f", key=f"pi_{i}"
            )
        pred_inputs.append(val)

    if st.button("🔍 Predict", type="primary"):
        x_in = np.array(pred_inputs)
        ws = float(np.dot(w, x_in) + b)
        pred = 1 if ws >= 0 else 0
        color = "#22C55E" if pred == 1 else "#EF4444"
        st.markdown(
            f"<h3 style='color:{color}'>Predicted Class: {pred}</h3>",
            unsafe_allow_html=True
        )
        with st.expander("Computation Breakdown"):
            terms = " + ".join([f"({w[i]:.4f} × {x_in[i]:.4f})" for i in range(n_feat)])
            st.code(
                f"weighted_sum = {terms} + ({b:.4f})\n"
                f"           = {ws:.6f}\n"
                f"step({'≥0' if ws >= 0 else '<0'})  →  class {pred}",
                language="text"
            )