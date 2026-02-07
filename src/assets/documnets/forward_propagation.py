import streamlit as st


def forward_propagation_docs_page() -> None:
	st.subheader("Forward Propagation")

	left, center, right = st.columns([1, 2, 1])
	with center:
		st.image(
			"https://miro.medium.com/v2/resize:fit:1200/1*J-v2B6T9RKxdvwThtQ1NVg.png",
			caption="Forward propagation overview",
			use_container_width=True,
			# width=300,
		)

	st.markdown(
		"""
## 1. What is Forward Propagation?

**Forward propagation** is the process by which **input data moves forward through a neural network to produce an output**.

> Forward propagation is simply the **prediction phase** of a neural network.

No learning happens here - only calculation.

---

## 2. Why Forward Propagation is Important

- Every **prediction** uses forward propagation
- Backpropagation depends entirely on its output
- If forward propagation is wrong, training is meaningless

Think of it as: **decision making before correction**.

---

## 3. Basic Flow (High Level)

1. Take input features
2. Multiply by weights
3. Add bias
4. Apply activation function
5. Pass output to next layer
6. Repeat until final output

---

## 4. Forward Propagation in a Single Neuron

**Step 1: Weighted Sum**
"""
	)

	st.latex(r"z = \sum_{i=1}^{n} w_i x_i + b")

	st.markdown(
		"""
Where:

- $x_i$ - input features
- $w_i$ - weights
- $b$ - bias

**Step 2: Activation Function**
"""
	)

	st.latex(r"a = f(z)")

	st.markdown(
		"""
Common activation functions:

- Step (Perceptron)
- Sigmoid
- ReLU
- Tanh

Output $a$ becomes input for the next layer.

---

## 5. Forward Propagation in a Multi-Layer Network (MLP)

**Layer 1 (Hidden Layer)**
"""
	)

	st.latex(r"z^{(1)} = W^{(1)} X + b^{(1)}")
	st.latex(r"a^{(1)} = f(z^{(1)})")

	st.markdown(
		"""
**Layer 2 (Next Hidden or Output Layer)**
"""
	)

	st.latex(r"z^{(2)} = W^{(2)} a^{(1)} + b^{(2)}")
	st.latex(r"a^{(2)} = f(z^{(2)})")

	st.markdown(
		"""
**General Formula for Any Layer $l$**
"""
	)

	st.latex(r"z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}")
	st.latex(r"a^{(l)} = f(z^{(l)})")

	st.markdown(
		"""
Where $a^{(0)} = X$ (input layer).

---

## 6. Activation Functions (Intuition + Math)

**1. Sigmoid**
"""
	)

	st.latex(r"f(z) = \frac{1}{1 + e^{-z}}")

	st.markdown(
		"""
- Output: $(0, 1)$
- Used in binary classification output layer
- Suffers from vanishing gradients

**2. ReLU (Most Used)**
"""
	)

	st.latex(r"f(z) = \max(0, z)")

	st.markdown(
		"""
- Fast
- Sparse activation
- Default choice in hidden layers

**3. Softmax (Multi-Class Output)**
"""
	)

	st.latex(r"\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}")

	st.markdown(
		"""
- Converts outputs into probabilities
- Sum of outputs = 1

---

## 7. Forward Propagation Example (Simple)

Suppose:

- Inputs: $x = [1, 2]$
- Weights: $w = [0.5, -1]$
- Bias: $b = 0.5$

**Step 1:**
"""
	)

	st.latex(r"z = (1)(0.5) + (2)(-1) + 0.5 = -1")

	st.markdown(
		"""
**Step 2 (ReLU):**
"""
	)

	st.latex(r"a = \max(0, -1) = 0")

	st.markdown(
		"""
That is the neuron's output.

---

## 8. Output Layer Forward Propagation

**Binary Classification:**
"""
	)

	st.latex(r"\hat{y} = \sigma(z)")

	st.markdown(
		"""
**Multi-Class Classification:**
"""
	)

	st.latex(r"\hat{y} = \text{Softmax}(z)")

	st.markdown(
		"""
**Regression:**
"""
	)

	st.latex(r"\hat{y} = z")

	st.markdown(
		"""
Choice of activation depends on **problem type**.

---

### 9. Key Observations

- Forward propagation is **pure matrix multiplication**
- Bias prevents the model from being too rigid
- Activation introduces **non-linearity**
- Without activation, the network collapses into a linear model

---

### 10. Advantages of Forward Propagation

- Simple and deterministic
- Fast (especially on GPUs)
- Used during training and inference
- Scales to deep networks

---

### 11. Limitations (By Itself)

- No learning happens
- Errors are not corrected
- Depends entirely on weight quality

Forward propagation **alone** is useless without backpropagation.

---

### 12. Exam / Interview One-Liner

> Forward propagation is the process of passing input data through the neural network layer by layer using weighted sums, bias addition, and activation functions to generate predictions.

---

### 13. Mental Model (Very Important)

> Forward propagation is flying the aircraft; backpropagation is course correction after feedback.
"""
	)

	# st.info(
	# 	"Next steps: loss functions (MSE, cross-entropy), backpropagation intuition, "
	# 	"vanishing gradients, or a complete training loop."
	# )
