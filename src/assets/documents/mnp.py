import streamlit as st


def mnp_docs_page() -> None:
	st.subheader("Multi-Layer Perceptron (MLP)")

	left, center, right = st.columns([1, 2, 1])
	with center:
		st.image(
			"src/assets/image/mnp.png",
			caption="MLP architecture",
			width=450,
		)



	st.markdown(
		"""
### 1. What is an MLP?

A **Multi-Layer Perceptron (MLP)** is a **feedforward neural network** that consists of:

- One **input layer**
- One or more **hidden layers**
- One **output layer**

In simple words: an MLP is a perceptron that grew up, gained depth, and learned to solve **non-linear problems**.

---

### 2. Why Do We Need MLP?

Single perceptron fails when:

- Data is non-linearly separable
- Problem requires feature interactions (XOR, images, text)

MLP solves this by:

- Adding **hidden layers**
- Using **non-linear activation functions**

---

### 3. Architecture of an MLP

**Input Layer**

- Receives raw features
- No computation

**Hidden Layers**

- Perform feature transformation
- Add non-linearity

**Output Layer**

- Produces final prediction

---

### 4. Forward Propagation in MLP (Math)

Let $a^{(0)} = X$ (input).

**For any hidden layer $l$**
"""
	)

	st.latex(r"z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}")
	st.latex(r"a^{(l)} = f(z^{(l)})")

	st.markdown(
		"""
**Output Layer**
"""
	)

	st.latex(r"z^{(L)} = W^{(L)} a^{(L-1)} + b^{(L)}")
	st.latex(r"\hat{y} = g(z^{(L)})")

	st.markdown(
		"""
Where:

- $f(\cdot)$ - hidden layer activation (ReLU, tanh)
- $g(\cdot)$ - output activation (sigmoid, softmax, linear)

---

### 5. Activation Functions Used in MLP

**Hidden Layers (Most Common) - ReLU**
"""
	)

	st.latex(r"f(z) = \max(0, z)")

	st.markdown(
		"""
Why ReLU:

- Fast
- Avoids vanishing gradients
- Sparse activations

**Output Layer (Depends on Task)**

| Task | Activation |
| --- | --- |
| Binary classification | Sigmoid |
| Multi-class classification | Softmax |
| Regression | Linear |

---

### 6. Backpropagation in MLP (Key Equations)

**Output Layer Error**
"""
	)

	st.latex(r"\delta^{(L)} = \frac{\partial L}{\partial z^{(L)}}")

	st.markdown(
		"""
**Hidden Layer Error**
"""
	)

	st.latex(
		r"\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \cdot f'(z^{(l)})"
	)

	st.markdown(
		"""
**Weight and Bias Gradients**
"""
	)

	st.latex(r"\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T")
	st.latex(r"\frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}")

	st.markdown(
		"""
**Update Rule (Gradient Descent)**
"""
	)

	st.latex(r"W = W - \eta \frac{\partial L}{\partial W}")
	st.latex(r"b = b - \eta \frac{\partial L}{\partial b}")

	st.markdown(
		"""
---

### 7. Key Power of MLP

An MLP can approximate any continuous function. This is the **Universal Approximation Theorem**.

Even 1 hidden layer is theoretically enough, but depth improves efficiency and learning.

---

### 8. Decision Boundary

- Perceptron - straight line
- MLP - **curved, complex boundaries**

Hidden layers bend space so classes become separable.

---
### 9. Advantages of MLP

- Solves non-linear problems
- Flexible architecture
- Works for classification and regression
- Foundation of deep learning

---

### 10. Limitations of MLP

- Requires large data
- Computationally expensive
- Prone to overfitting
- Hard to interpret

Also:

- Poor with raw images - CNN needed
- Poor with sequences - RNN/Transformer needed

---

### 11. Common Beginner Mistakes

- Too many layers for small data
- Wrong activation-loss pairing
- Ignoring normalization
- Expecting MLP to beat CNNs on images

---

### 12. When Should You Use MLP?

- Tabular data
- Credit scoring
- Sensor data
- Simple pattern learning

Not ideal for:

- Images
- Long sequences
- NLP tasks

---
### 13. Exam / Interview One-Liner

> A Multi-Layer Perceptron is a feedforward neural network with one or more hidden layers that uses non-linear activation functions and backpropagation to learn complex decision boundaries.

---

### 14. Mental Model

> Perceptron = single decision; MLP = committee of decisions, refined layer by layer.
"""
	)

        # st.info(
        # 	"Next steps: XOR solved using MLP, loss functions in depth, vanishing vs "
        # 	"exploding gradients, training loop from scratch, or MLP vs CNN vs RNN."
        # )
