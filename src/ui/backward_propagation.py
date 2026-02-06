import streamlit as st
import numpy as np
import random

# -------------------------------------------------
# Activation Functions
# -------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(a):
    # derivative using activated output
    return a * (1 - a)


def backward_propagation_page():
    st.title("Backward Propagation in a NN")

    st.caption(
        "This module demonstrates how a neural network updates its weights using "
        "error gradients. It uses a fixed architecture with one hidden layer."
    )

# -------------------------------------------------
# Input & Target
# -------------------------------------------------
    st.divider()
    st.subheader("Input and Target")

    col1, col2, col3 = st.columns(3)

    with col1:
        x1 = st.number_input("Input x₁", value=0.30, step=0.1)

    with col2:
        x2 = st.number_input("Input x₂", value=0.90, step=0.1)

    with col3:
        y_true = st.number_input("Target y", value=1.0, step=0.1)

    X = np.array([[x1], [x2]])

# -------------------------------------------------
# Learning Rate
# -------------------------------------------------
    learning_rate = st.number_input("Learning Rate (η)", value=0.1, step=0.01)

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
            w1o = st.number_input("w₁o", value=1.0, step=0.1)
            w2o = st.number_input("w₂o", value=1.0, step=0.1)
        else:
            w1o = st.number_input("w₁o", value=1.0, step=0.1)
            w2o = None
        bo = st.number_input("bₒ", value=0.0, step=0.1)
    else:
        w1o = random.uniform(-1, 1)
        w2o = random.uniform(-1, 1) if hidden_neurons == 2 else None
        bo = random.uniform(-1, 1)

    W_output = np.array([[w1o, w2o]]) if hidden_neurons == 2 else np.array([[w1o]])
    b_output = np.array([[bo]])

# -------------------------------------------------
# Activation Option
# -------------------------------------------------
    st.divider()
    st.subheader("Activation Option")

    use_activation = st.checkbox("Apply activation function", value=True)

# -------------------------------------------------
# Backward Propagation
# -------------------------------------------------
    st.divider()
    st.subheader("Backward Propagation")

    if st.button("Run Backward Propagation"):

    # ---------- Forward Pass ----------
        Z_hidden = np.dot(W_hidden.T, X) + b_hidden
        A_hidden = sigmoid(Z_hidden) if use_activation else Z_hidden

        Z_output = np.dot(W_output, A_hidden) + b_output
        y_pred = sigmoid(Z_output) if use_activation else Z_output

    # ---------- Loss ----------
        loss = 0.5 * (y_true - y_pred[0][0]) ** 2

    # ---------- Output Layer Gradients ----------
        dL_dy = y_pred - y_true
        dy_dz = sigmoid_derivative(y_pred) if use_activation else 1
        dL_dz_output = dL_dy * dy_dz

        dL_dW_output = dL_dz_output * A_hidden.T
        dL_db_output = dL_dz_output

    # ---------- Hidden Layer Gradients ----------
        dL_dA_hidden = W_output.T * dL_dz_output
        dA_dZ_hidden = sigmoid_derivative(A_hidden) if use_activation else 1
        dL_dZ_hidden = dL_dA_hidden * dA_dZ_hidden

        dL_dW_hidden = np.dot(X, dL_dZ_hidden.T)
        dL_db_hidden = dL_dZ_hidden

    # ---------- Weight Updates ----------
        W_output_new = W_output - learning_rate * dL_dW_output
        b_output_new = b_output - learning_rate * dL_db_output

        W_hidden_new = W_hidden - learning_rate * dL_dW_hidden
        b_hidden_new = b_hidden - learning_rate * dL_db_hidden

    # -------------------------------------------------
    # Display Results
    # -------------------------------------------------
        st.markdown("### Loss")
        st.success(f"{loss:.6f}")

        st.markdown("### Updated Output Layer Parameters")
        st.write("**Updated Weights:**")
        st.write(", ".join([f"{value:.4f}" for value in W_output_new.flatten()]))
        st.write("**Updated Bias:**")
        st.write(f"{b_output_new.flatten()[0]:.4f}")

        st.markdown("### Updated Hidden Layer Parameters")
        st.write("**Updated Weights:**")
        st.write(", ".join([f"{value:.4f}" for value in W_hidden_new.flatten()]))
        st.write("**Updated Bias:**")
        st.write(", ".join([f"{value:.4f}" for value in b_hidden_new.flatten()]))

