import streamlit as st


def perceptron_docs_page() -> None:
	st.subheader("Perceptron - Beginner-Friendly Notes")

	left, center, right = st.columns([1, 2, 1])
	with center:
		st.image(
			"https://miro.medium.com/0%2Am87K4ZlVc1MR_Uvu.jpeg",
			caption="Perceptron intuition",
			use_container_width=True,
		)

	st.markdown(
		"""
### 1. What is a Perceptron?

A **perceptron** is the **simplest neural network model**, inspired by how a biological neuron works.

> A perceptron is **not magic** - it is basically a *smart decision rule* that learns how to classify data using weights.

It is mainly used for **binary classification** (yes/no, 0/1).

---

### 2. Intuition (Before Math)

Imagine you are an air defense officer deciding:

> "Should I raise an alert or not?"

You consider:

- Radar signal strength
- Speed of object
- Altitude

Each factor has **importance** (weight). You add them up, compare with a **threshold**, and decide.
That is exactly what a perceptron does.

---

### 3. Components of a Perceptron

A perceptron has:

1. **Inputs:** $x_1, x_2, x_3, \dots, x_n$
2. **Weights (importance of each input):** $w_1, w_2, w_3, \dots, w_n$
3. **Bias (threshold adjustment):** $b$
4. **Activation Function** (decision maker)

---

### 4. Mathematical Model

**Step 1: Weighted Sum**

$$
z = \sum_{i=1}^{n} w_i x_i + b
$$

**Step 2: Activation Function (Step Function)**

This makes the final **binary decision**.

---

### 5. Perceptron Learning Rule (Training)

Training means **adjusting weights** when the prediction is wrong.

**Weight Update Formula:**

$$
w_i^{new} = w_i^{old} + \eta (y_{true} - y_{pred}) x_i
$$

**Bias Update:**

$$
b^{new} = b^{old} + \eta (y_{true} - y_{pred})
$$

Where:

- $\eta$ = learning rate (small positive value)
- $y_{true}$ = actual label
- $y_{pred}$ = predicted output

**Key Insight:** Perceptron learns **only when it makes a mistake**.

---

### 6. Decision Boundary

Mathematically:

$$
w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b = 0
$$

This equation represents a **straight line (2D)** or **hyperplane (higher dimensions)**.

---

### 7. Advantages of Perceptron

- Simple to understand and implement
- Fast training
- Works well for linearly separable data
- Foundation of all neural networks

---

### 8. Disadvantages of Perceptron

- Only works for **binary classification**
- Uses **step activation** (not differentiable)
- Cannot learn complex patterns
- No probability output

---

### 9. Limitations (Very Important)

**1. Cannot Solve Non-Linear Problems**

Classic example: **XOR problem**. No straight line can separate XOR data.

**2. Single Layer Only**

A single perceptron:

- Has **no hidden layers**
- Cannot learn hierarchical features

**3. No Gradient-Based Optimization**

Because the step function is non-differentiable:

- Backpropagation cannot be applied
- Learning is very limited

---

### 10. Why Perceptron Still Matters (Real Talk)

Every **CNN, RNN, Transformer** is built from the same core idea:
**weighted sum + activation**.

Perceptron teaches:

- Decision boundaries
- Role of weights and bias
- Why deep learning was needed

Think of it like **basic flight training** before flying a fighter jet.

---

### 11. One-Line Exam Summary

> A perceptron is a single-layer neural network used for binary classification that learns a linear decision boundary using weighted inputs, bias, and a step activation function.
"""
	)

	st.latex(
		r"""
y =
\begin{cases}
1 & \text{if } z \ge 0 \\
0 & \text{if } z < 0
\end{cases}
"""
	)

	# st.info(
	# 	"Next topics: XOR problem (with diagram), why sigmoid replaced step function, "
	# 	"how perceptron evolved into MLP, or a code-level walkthrough."
	# )
