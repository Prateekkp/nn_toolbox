import streamlit as st
import random
import plotly.express as px
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def perceptron_page():
    st.title("Perceptron – Logic Gates Simulator")

    # -------------------------------------------------
    # Gate Datasets
    # -------------------------------------------------
    and_gate = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
    or_gate = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]

    gate_choice = st.selectbox("Select Logic Gate", ["AND", "OR"])
    data = and_gate if gate_choice == "AND" else or_gate

    # -------------------------------------------------
    # Truth Table
    # -------------------------------------------------
    st.subheader(f"{gate_choice} Gate Truth Table")

    df_gate = pd.DataFrame(data, columns=["X1", "X2", "Output"])
    st.dataframe(df_gate, hide_index=True, use_container_width=True)

    # -------------------------------------------------
    # Session State Initialization
    # -------------------------------------------------
    if "w1" not in st.session_state:
        st.session_state.w1 = random.uniform(-1, 1)
        st.session_state.w2 = random.uniform(-1, 1)
        st.session_state.w3 = random.uniform(-1, 1)
        st.session_state.b = random.uniform(-1, 1)
        st.session_state.losses = []
        st.session_state.trained = False

    # -------------------------------------------------
    # Parameter Selection
    # -------------------------------------------------
    st.divider()
    st.subheader("Parameter Selection")

    mode = st.radio(
        "Choose Initialization Mode",
        ["Random", "Manual"],
        horizontal=True
    )

    learning_rate = st.number_input(
        "Learning Rate",
        value=0.1,
        step=0.01
    )

    epochs = st.slider(
        "Maximum Epochs",
        min_value=10,
        max_value=1000,
        value=100
    )

    # Manual mode UI
    if mode == "Manual":
        with st.expander("Manual Parameter Input", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                w1 = st.number_input(
                    "Weight (w1)",
                    min_value=-1.0,
                    max_value=1.0,
                    value=st.session_state.w1,
                    step=0.1
                )
                w2 = st.number_input(
                    "Weight (w2)",
                    min_value=-1.0,
                    max_value=1.0,
                    value=st.session_state.w2,
                    step=0.1
                )

            with col2:
                w3 = st.number_input(
                    "Weight (w3)",
                    min_value=-1.0,
                    max_value=1.0,
                    value=st.session_state.w3,
                    step=0.1
                )
                b = st.number_input(
                    "Bias (b)",
                    min_value=-1.0,
                    max_value=1.0,
                    value=st.session_state.b,
                    step=0.1
                )
    else:
        w1 = w2 = w3 = b = None
        st.info("Weights and bias will be randomly initialized during training.")

    # -------------------------------------------------
    # Activation Function
    # -------------------------------------------------
    def activation(x):
        return 1 if x >= 0 else 0

    # -------------------------------------------------
    # Training
    # -------------------------------------------------
    st.divider()

    if st.button("Train Perceptron"):
        if mode == "Random":
            w1 = random.uniform(-1, 1)
            w2 = random.uniform(-1, 1)
            w3 = random.uniform(-1, 1)
            b = random.uniform(-1, 1)

        losses = []

        for epoch in range(epochs):
            total_error = 0

            for x1, x2, y in data:
                weighted_sum = w1 * x1 + w2 * x2 + w3 * 1 + b
                y_pred = activation(weighted_sum)

                error = y - y_pred
                total_error += abs(error)

                if error != 0:
                    w1 += learning_rate * error * x1
                    w2 += learning_rate * error * x2
                    w3 += learning_rate * error * 1
                    b += learning_rate * error

            losses.append(total_error)

        st.session_state.w1 = w1
        st.session_state.w2 = w2
        st.session_state.w3 = w3
        st.session_state.b = b
        st.session_state.losses = losses
        st.session_state.trained = True

        st.success("Training completed successfully.")

    # -------------------------------------------------
    # Final Learned Parameters
    # -------------------------------------------------
    if st.session_state.trained:
        st.divider()
        st.subheader("Final Learned Parameters")
        st.caption("These parameters are learned after training convergence.")

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Weight (w1):** {st.session_state.w1:.3f}")
            st.write(f"**Weight (w2):** {st.session_state.w2:.3f}")

        with col2:
            st.write(f"**Weight (w3):** {st.session_state.w3:.3f}")
            st.write(f"**Bias (b):** {st.session_state.b:.3f}")

    # -------------------------------------------------
    # Loss Curve
    # -------------------------------------------------
    if st.session_state.losses:
        st.divider()
        st.subheader("Training Loss Curve")

        fig = px.line(
            y=st.session_state.losses,
            labels={"x": "Epoch", "y": "Total Error"},
            title="Perceptron Training Error Over Epochs"
        )
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------
    if st.session_state.trained:
        st.divider()
        st.subheader("Prediction")
        st.caption("Test the trained perceptron on new binary inputs.")

        x1_pred = st.number_input("Input X1", 0, 1, step=1)
        x2_pred = st.number_input("Input X2", 0, 1, step=1)

        if st.button("Predict"):
            weighted_sum = (
                st.session_state.w1 * x1_pred +
                st.session_state.w2 * x2_pred +
                st.session_state.w3 * 1 +
                st.session_state.b
            )
            prediction = activation(weighted_sum)
            st.success(f"Predicted Output: {prediction}")

    # -------------------------------------------------
    # Perceptron Documentation
    # -------------------------------------------------
    st.divider()
    st.subheader("About the Perceptron")

    st.markdown("""
    ### What is a Perceptron?
    A perceptron is the simplest type of neural network used for **binary classification**.
    It learns by adjusting its weights whenever it makes a wrong prediction.

    ### Mathematical Model
    The perceptron computes a weighted sum:

    **z = w₁x₁ + w₂x₂ + w₃·1 + b**

    The output is produced using a step activation function:

    - Output = 1, if z ≥ 0  
    - Output = 0, if z < 0  

    ### Learning Rule
    For each training example:
    - If the prediction is correct, no update is made
    - If the prediction is incorrect, weights are updated as:

    **wᵢ = wᵢ + η · error · xᵢ**

    where η is the learning rate.

    ### Key Limitation
    A single perceptron can only solve **linearly separable problems**
    such as AND and OR gates. It **cannot** solve XOR.

    This simulator demonstrates both the learning capability and the limitation
    of a single-layer perceptron.
    """)


if __name__ == "__main__":
    st.set_page_config(page_title="Perceptron Gates", layout="centered")
    perceptron_page()
