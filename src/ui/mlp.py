import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(a):
    return a * (1 - a)


def softmax(z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def standardize_numeric(df_numeric):
    means = df_numeric.mean()
    stds = df_numeric.std().replace(0, 1)
    return (df_numeric - means) / stds, means, stds


def preprocess_features(df, numeric_cols, categorical_cols, means=None, stds=None, dummy_columns=None):
    if numeric_cols:
        numeric_df = df[numeric_cols].astype(float)
        if means is None or stds is None:
            numeric_scaled, means, stds = standardize_numeric(numeric_df)
        else:
            numeric_scaled = (numeric_df - means) / stds
    else:
        numeric_scaled = pd.DataFrame(index=df.index)

    if categorical_cols:
        categorical_df = df[categorical_cols].astype(str)
        categorical_dummies = pd.get_dummies(categorical_df, drop_first=False)
        if dummy_columns is not None:
            categorical_dummies = categorical_dummies.reindex(columns=dummy_columns, fill_value=0)
        else:
            dummy_columns = categorical_dummies.columns.tolist()
    else:
        categorical_dummies = pd.DataFrame(index=df.index)

    features_df = pd.concat([numeric_scaled, categorical_dummies], axis=1)
    return features_df, means, stds, dummy_columns


def validate_dataset_size(num_rows, num_features, max_rows=20000, max_features=500):
    if num_rows > max_rows:
        return False, f"Dataset too large: {num_rows} rows (max {max_rows})."
    if num_features > max_features:
        return False, f"Too many features after encoding: {num_features} (max {max_features})."
    return True, ""


def mlp_page():
    st.title("Multi-Layer Neural Network")

    st.caption(
        "This module demonstrates how a multi-layer neural network with a hidden layer "
        "can successfully learn classification problems using forward and backward propagation."
    )


# -------------------------------------------------
# Data Source
# -------------------------------------------------
    data_source = st.radio("Select Data Source", ["Logic Gate", "Upload CSV"], horizontal=True)
    state_prefix = "logic_gate" if data_source == "Logic Gate" else "csv"

# -------------------------------------------------
# Gate Datasets
# -------------------------------------------------
    AND_GATE = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
    OR_GATE = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
    XOR_GATE = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]

    data_ready = False
    task_type = None
    class_labels = None
    feature_columns = None
    numeric_cols = []
    categorical_cols = []
    categorical_values = {}
    feature_means = None
    feature_stds = None
    feature_dummy_columns = None

    if data_source == "Logic Gate":
        gate_choice = st.selectbox("Select Logic Gate", ["AND", "OR", "XOR"])
        if gate_choice == "AND":
            data = AND_GATE
        elif gate_choice == "OR":
            data = OR_GATE
        else:
            data = XOR_GATE

        X = np.array([[row[0], row[1]] for row in data])
        y = np.array([[row[2]] for row in data])
        data_ready = True
        task_type = "binary"
        class_labels = [0, 1]
        feature_columns = ["X1", "X2"]
        feature_dummy_columns = []
    else:
        gate_choice = "Custom"
        csv_source = st.radio("CSV Source", ["Upload CSV", "Use Sample Iris Dataset"], horizontal=True)
        df_uploaded = None

        if csv_source == "Use Sample Iris Dataset":
            try:
                df_uploaded = pd.read_csv("data/IRIS.csv")
                st.info("Loaded sample dataset: IRIS.csv")
            except FileNotFoundError:
                st.error("Sample dataset not found: data/IRIS.csv")
        else:
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded_file is not None:
                df_uploaded = pd.read_csv(uploaded_file)

        if df_uploaded is not None:
            st.subheader("Uploaded Data Preview")
            st.dataframe(df_uploaded.head(10), hide_index=True, use_container_width=True)

            column_options = list(df_uploaded.columns)
            target_col = st.selectbox("Select target column", column_options)

            if target_col:
                df_model = df_uploaded.dropna(subset=[target_col]).copy()
                feature_columns = [col for col in df_model.columns if col != target_col]

                if not feature_columns:
                    st.warning("Please select a target column with at least one feature column remaining.")
                else:
                    X_df = df_model[feature_columns]
                    y_raw = df_model[target_col]

                    unique_targets = pd.Series(y_raw).dropna().unique()
                    if len(unique_targets) < 2:
                        st.warning("Target column must contain at least two classes.")
                    else:
                        task_type = "binary" if len(unique_targets) == 2 else "multiclass"
                        st.info(f"Detected task type: {task_type} classification")

                        numeric_cols = X_df.select_dtypes(include=["number"]).columns.tolist()
                        categorical_cols = [col for col in feature_columns if col not in numeric_cols]

                        for col in categorical_cols:
                            categorical_values[col] = X_df[col].astype(str).unique().tolist()

                        try:
                            X_features, feature_means, feature_stds, feature_dummy_columns = preprocess_features(
                                X_df,
                                numeric_cols,
                                categorical_cols
                            )
                        except MemoryError:
                            st.error("Dataset not supported: ran out of memory during preprocessing. Try fewer rows or fewer categorical levels.")
                            X_features = None

                        if X_features is not None:
                            is_valid, message = validate_dataset_size(X_features.shape[0], X_features.shape[1])
                            if not is_valid:
                                st.error(f"Dataset not supported: {message}")
                                X_features = None

                        if X_features is None:
                            data_ready = False
                        elif task_type == "binary":
                            if set(unique_targets).issubset({0, 1}):
                                y = y_raw.astype(int).to_numpy().reshape(-1, 1)
                                class_labels = [0, 1]
                            else:
                                encoded, class_labels = pd.factorize(y_raw)
                                y = encoded.reshape(-1, 1)
                        else:
                            encoded, class_labels = pd.factorize(y_raw)
                            y = np.eye(len(class_labels))[encoded]

                        if X_features is not None:
                            X = X_features.to_numpy(dtype=float)
                            data_ready = True

# -------------------------------------------------
# Data Preview
# -------------------------------------------------
    if data_ready and data_source == "Logic Gate":
        st.subheader(f"{gate_choice} Data")
        df_preview = pd.DataFrame(
            np.column_stack([X, y]),
            columns=["X1", "X2", "Output"]
        )
        st.dataframe(df_preview, hide_index=True, use_container_width=True)

# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------
    st.divider()
    st.subheader("Training Parameters")

    learning_rate = st.number_input("Learning Rate (η)", value=0.5, step=0.1)
    epochs = st.slider("Epochs", min_value=100, max_value=5000, value=2000, step=100)

    st.subheader("Network Size")
    hidden_neurons = st.slider("Hidden Neurons", min_value=1, max_value=16, value=4, step=1)
    st.caption("Disclaimer: For MLP tasks, prefer more than 2 hidden neurons for better learning capacity.")

# -------------------------------------------------
# Weight Initialization
# -------------------------------------------------
    np.random.seed(42)

    input_dim = X.shape[1] if data_ready else 2
    hidden_dim = hidden_neurons
    if task_type == "multiclass" and class_labels is not None:
        output_dim = len(class_labels)
    else:
        output_dim = 1

    W_hidden = np.random.uniform(-1, 1, (input_dim, hidden_dim))
    b_hidden = np.random.uniform(-1, 1, (1, hidden_dim))

    W_output = np.random.uniform(-1, 1, (hidden_dim, output_dim))
    b_output = np.random.uniform(-1, 1, (1, output_dim))

# -------------------------------------------------
# Training
# -------------------------------------------------
    st.divider()
    if st.button("Train Network", disabled=not data_ready):
        losses = []

        for epoch in range(epochs):
            # ---------- Forward Propagation ----------
            Z_hidden = np.dot(X, W_hidden) + b_hidden
            A_hidden = sigmoid(Z_hidden)

            Z_output = np.dot(A_hidden, W_output) + b_output
            if task_type == "multiclass":
                y_pred = softmax(Z_output)
                loss = -np.mean(np.sum(y * np.log(y_pred + 1e-8), axis=1))
            else:
                y_pred = sigmoid(Z_output)
                loss = -np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))

            losses.append(loss)

            # ---------- Backward Propagation ----------
            dZ_output = y_pred - y
            dW_output = np.dot(A_hidden.T, dZ_output) / X.shape[0]
            db_output = np.mean(dZ_output, axis=0, keepdims=True)

            dA_hidden = np.dot(dZ_output, W_output.T)
            dZ_hidden = dA_hidden * sigmoid_derivative(A_hidden)
            dW_hidden = np.dot(X.T, dZ_hidden) / X.shape[0]
            db_hidden = np.mean(dZ_hidden, axis=0, keepdims=True)

            # ---------- Weight Updates ----------
            W_output -= learning_rate * dW_output
            b_output -= learning_rate * db_output

            W_hidden -= learning_rate * dW_hidden
            b_hidden -= learning_rate * db_hidden

        st.success("Training completed successfully.")

        st.session_state[f"{state_prefix}_losses"] = losses
        st.session_state[f"{state_prefix}_W_hidden"] = W_hidden
        st.session_state[f"{state_prefix}_b_hidden"] = b_hidden
        st.session_state[f"{state_prefix}_W_output"] = W_output
        st.session_state[f"{state_prefix}_b_output"] = b_output
        st.session_state[f"{state_prefix}_task_type"] = task_type
        st.session_state[f"{state_prefix}_class_labels"] = class_labels
        st.session_state[f"{state_prefix}_feature_columns"] = feature_columns
        st.session_state[f"{state_prefix}_numeric_cols"] = numeric_cols
        st.session_state[f"{state_prefix}_categorical_cols"] = categorical_cols
        st.session_state[f"{state_prefix}_categorical_values"] = categorical_values
        st.session_state[f"{state_prefix}_feature_means"] = feature_means
        st.session_state[f"{state_prefix}_feature_stds"] = feature_stds
        st.session_state[f"{state_prefix}_feature_dummy_columns"] = feature_dummy_columns
        st.session_state[f"{state_prefix}_trained"] = True

# -------------------------------------------------
# Loss Curve
# -------------------------------------------------
    if st.session_state.get(f"{state_prefix}_losses") is not None:
        st.divider()
        st.subheader("Training Loss Curve")

        fig = px.line(
            y=st.session_state[f"{state_prefix}_losses"],
            labels={"x": "Epoch", "y": "Mean Squared Error"},
            title="MLP Training Loss"
        )
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
    if st.session_state.get(f"{state_prefix}_trained", False):
        st.divider()
        st.subheader("Prediction")

        if data_source == "Logic Gate":
            col1, col2 = st.columns(2)

            with col1:
                x1 = st.number_input("Input x₁", 0, 1, step=1)

            with col2:
                x2 = st.number_input("Input x₂", 0, 1, step=1)

            input_df = pd.DataFrame([[x1, x2]], columns=["X1", "X2"])
            X_test = input_df.to_numpy(dtype=float)
        else:
            input_values = {}
            for col in st.session_state[f"{state_prefix}_feature_columns"]:
                if col in st.session_state[f"{state_prefix}_categorical_cols"]:
                    input_values[col] = st.selectbox(
                        f"{col}",
                        st.session_state[f"{state_prefix}_categorical_values"].get(col, [])
                    )
                else:
                    default_value = 0.0
                    if st.session_state[f"{state_prefix}_feature_means"] is not None and col in st.session_state[f"{state_prefix}_feature_means"]:
                        default_value = float(st.session_state[f"{state_prefix}_feature_means"][col])
                    input_values[col] = st.number_input(f"{col}", value=default_value)

            input_df = pd.DataFrame([input_values])
            X_features, _, _, _ = preprocess_features(
                input_df,
                st.session_state[f"{state_prefix}_numeric_cols"],
                st.session_state[f"{state_prefix}_categorical_cols"],
                st.session_state[f"{state_prefix}_feature_means"],
                st.session_state[f"{state_prefix}_feature_stds"],
                st.session_state[f"{state_prefix}_feature_dummy_columns"]
            )
            X_test = X_features.to_numpy(dtype=float)

        if st.button("Predict"):
            Z_hidden = np.dot(X_test, st.session_state[f"{state_prefix}_W_hidden"]) + st.session_state[f"{state_prefix}_b_hidden"]
            A_hidden = sigmoid(Z_hidden)

            Z_output = np.dot(A_hidden, st.session_state[f"{state_prefix}_W_output"]) + st.session_state[f"{state_prefix}_b_output"]
            if st.session_state[f"{state_prefix}_task_type"] == "multiclass":
                y_pred = softmax(Z_output)
                class_index = int(np.argmax(y_pred, axis=1)[0])
                class_label = st.session_state[f"{state_prefix}_class_labels"][class_index]
                st.success(f"Predicted Class: {class_label}")
                st.caption("Probabilities: " + ", ".join([f"{val:.4f}" for val in y_pred[0]]))
            else:
                y_pred = sigmoid(Z_output)
                st.success(f"Predicted Output: {int(y_pred[0][0] >= 0.5)}")
                st.caption(f"Raw Output Value: {y_pred[0][0]:.4f}")

