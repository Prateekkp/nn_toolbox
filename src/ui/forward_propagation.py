import streamlit as st
import numpy as np
import random

# -------------------------------------------------
# Sigmoid Activation Function
# -------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward_propagation_page():
    st.title("Forward Propagation in a NN")

    st.caption(
        "This module demonstrates how inputs move forward through a neural network "
        "to produce an output. No training or weight updates are performed here."
    )

# -------------------------------------------------
# Input Section
# -------------------------------------------------
    st.divider()
    st.subheader("Input Layer")

    col1, col2 = st.columns(2)

    with col1:
        x1 = st.number_input("Input x₁", value=0.30, step=0.1)

    with col2:
        x2 = st.number_input("Input x₂", value=0.90, step=0.1)

    X = np.array([[x1], [x2]])

# -------------------------------------------------
# Hidden Layer Parameters
# -------------------------------------------------
    st.divider()
    st.subheader("Hidden Layer Parameters")

    hidden_neurons = st.radio(
        "Number of hidden neurons",
        [1, 2],
        horizontal=True
    )

    mode = st.radio(
        "Choose Initialization Mode",
        ["Random", "Manual"],
        horizontal=True
    )

    st.caption(f"Weights and bias for {hidden_neurons} hidden neuron{'s' if hidden_neurons > 1 else ''}")

    col1, col2 = st.columns(2)

    if mode == "Manual":
        with col1:
            st.markdown("**Hidden Neuron 1**")
            w11 = st.number_input("w₁₁", value=0.5, step=0.1)
            w21 = st.number_input("w₂₁", value=0.5, step=0.1)
            b1 = st.number_input("b₁", value=0.0, step=0.1)

        if hidden_neurons == 2:
            with col2:
                st.markdown("**Hidden Neuron 2**")
                w12 = st.number_input("w₁₂", value=-0.5, step=0.1)
                w22 = st.number_input("w₂₂", value=0.5, step=0.1)
                b2 = st.number_input("b₂", value=0.0, step=0.1)
        else:
            w12 = w22 = b2 = None
    else:
        w11 = random.uniform(-1, 1)
        w21 = random.uniform(-1, 1)
        b1 = random.uniform(-1, 1)
        if hidden_neurons == 2:
            w12 = random.uniform(-1, 1)
            w22 = random.uniform(-1, 1)
            b2 = random.uniform(-1, 1)
        else:
            w12 = w22 = b2 = None
        st.info("Weights and bias are randomly initialized.")

    if hidden_neurons == 2:
        W_hidden = np.array([
            [w11, w12],
            [w21, w22]
        ])

        b_hidden = np.array([[b1], [b2]])
    else:
        W_hidden = np.array([
            [w11],
            [w21]
        ])

        b_hidden = np.array([[b1]])

# -------------------------------------------------
# Output Layer Parameters
# -------------------------------------------------
    if mode == "Manual":
        st.divider()
        st.subheader("Output Layer Parameters")

        if hidden_neurons == 2:
            w1o = st.number_input("Weight from hidden neuron 1 (w₁o)", value=1.0, step=0.1)
            w2o = st.number_input("Weight from hidden neuron 2 (w₂o)", value=1.0, step=0.1)
        else:
            w1o = st.number_input("Weight from hidden neuron 1 (w₁o)", value=1.0, step=0.1)
            w2o = None
        bo = st.number_input("Output bias (bₒ)", value=0.0, step=0.1)
    else:
        w1o = random.uniform(-1, 1)
        w2o = random.uniform(-1, 1) if hidden_neurons == 2 else None
        bo = random.uniform(-1, 1)

    W_output = np.array([[w1o, w2o]]) if hidden_neurons == 2 else np.array([[w1o]])
    b_output = np.array([[bo]])

# -------------------------------------------------
# Forward Propagation
# -------------------------------------------------
    st.divider()
    st.subheader("Activation Option")

    use_activation = st.checkbox("Apply activation function", value=True)

    # -------------------------------------------------
    # Forward Propagation
    # -------------------------------------------------
    st.divider()
    st.subheader("Forward Propagation Computation")

    if st.button("Run Forward Propagation"):
        # Hidden layer computation
        Z_hidden = np.dot(W_hidden.T, X) + b_hidden
        A_hidden = sigmoid(Z_hidden) if use_activation else Z_hidden

        # Output layer computation
        Z_output = np.dot(W_output, A_hidden) + b_output
        A_output = sigmoid(Z_output) if use_activation else Z_output

        st.markdown("### Hidden Layer Output")
        st.write("**Weighted Sum (Z):**")
        st.write(", ".join([f"{value:.4f}" for value in Z_hidden.flatten()]))

        st.write("**Activated Output (A):**")
        st.write(", ".join([f"{value:.4f}" for value in A_hidden.flatten()]))

        st.markdown("### Output Layer Result")
        st.write("**Weighted Sum (Z):**")
        st.write(f"{Z_output.flatten()[0]:.4f}")

        output_label = "Final Output (After Sigmoid):" if use_activation else "Final Output (Linear):"
        st.write(f"**{output_label}**")
        st.success(f"{A_output[0][0]:.4f}")

