import streamlit as st


def back_propagation_docs_page() -> None:
	st.subheader("Backward Propagation (Backprop)")

	left, center, right = st.columns([1, 2, 1])
	with center:
		st.image(
			"https://cdn.botpenguin.com/assets/website/image_347c8041fd.png",
			caption="Gradient flow intuition",
			use_container_width=True,
		)

	st.markdown(
		"""
### 1. What is Backward Propagation?

**Backward propagation** is the algorithm used to **train neural networks** by:

- Measuring how wrong the prediction is
- Sending that error **backwards**
- Updating weights to reduce future error

In plain words: backpropagation is **learning from mistakes**.

---

### 2. Why Backpropagation Exists

- Forward propagation only predicts
- Models do not improve unless errors guide updates
- Backprop efficiently tells **which weight caused how much error**

Without backprop, deep learning would not exist.

---

### 3. High-Level Workflow

1. Perform **forward propagation**
2. Compute **loss**
3. Calculate gradients using **chain rule**
4. Update weights using **gradient descent**
5. Repeat until convergence

---

### 4. Loss Function (Starting Point)

**Mean Squared Error (Regression)**
"""
	)

	st.latex(r"L = \frac{1}{2}(y - \hat{y})^2")

	st.markdown(
		"""
**Binary Cross Entropy**
"""
	)

	st.latex(r"L = -[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]")

	st.markdown(
		"""
---

### 5. Core Idea: Chain Rule

If:
"""
	)

	st.latex(r"y = f(g(x))")

	st.markdown(
		"""
Then:
"""
	)

	st.latex(r"\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}")

	st.markdown(
		"""
Backprop applies this **layer by layer, backwards**.

---

### 6. Backpropagation in a Single Neuron

**Step 1: Forward Pass**
"""
	)

	st.latex(r"z = wx + b")
	st.latex(r"\hat{y} = f(z)")

	st.markdown(
		"""
**Step 2: Loss Gradient**
"""
	)

	st.latex(r"\frac{\partial L}{\partial \hat{y}}")

	st.markdown(
		"""
**Step 3: Gradient w.r.t Activation Input**
"""
	)

	st.latex(
		r"\frac{\partial L}{\partial z} = \frac{\partial L}{\partial \hat{y}} \cdot f'(z)"
	)

	st.markdown(
		"""
**Step 4: Gradients of Weights and Bias**
"""
	)

	st.latex(r"\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} \cdot x")
	st.latex(r"\frac{\partial L}{\partial b} = \frac{\partial L}{\partial z}")

	st.markdown(
		"""
**Step 5: Weight Update Rule**
"""
	)

	st.latex(r"w = w - \eta \frac{\partial L}{\partial w}")
	st.latex(r"b = b - \eta \frac{\partial L}{\partial b}")

	st.markdown(
		"""
Where $\eta$ is the learning rate.

---

### 7. Backpropagation in a Multi-Layer Network

**Output Layer Error**
"""
	)

	st.latex(r"\delta^{(L)} = \frac{\partial L}{\partial z^{(L)}}")
	st.latex(r"\delta^{(L)} = (\hat{y} - y) \cdot f'(z^{(L)})")

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
**Gradient Calculation**
"""
	)

	st.latex(r"\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T")
	st.latex(r"\frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}")

	st.markdown(
		"""
---

### 8. Intuition (Very Important)

- Output layer asks: "How wrong am I?"
- Hidden layer asks: "How much did I contribute to that mistake?"
- Earlier layers ask: "Was my influence big or small?"

Backprop answers this **numerically**.

---

### 9. Why Activation Function Matters

Without activation:

- Derivatives become constant
- Network becomes linear
- Learning collapses

**ReLU derivative**
"""
	)

	st.latex(
		r"f'(z) = \begin{cases} 1 & z > 0 \\ 0 & z \le 0 \end{cases}"
	)

	st.markdown(
		"""
**Sigmoid derivative**
"""
	)

	st.latex(r"f'(z) = f(z)(1 - f(z))")

	st.markdown(
		"""
---

### 10. Advantages of Backpropagation

- Efficient computation
- Scales to deep networks
- Uses gradient descent optimally
- Works with millions of parameters

---

### 11. Limitations and Practical Issues

- Vanishing gradients (sigmoid, tanh)
- Exploding gradients
- Sensitive to learning rate
- Requires differentiable functions

This is why ReLU, batch norm, and residual connections exist.

---

### 12. Common Beginner Mistakes

- Forgetting bias gradient
- Wrong matrix dimensions
- Using wrong activation-loss combo
- Learning rate too high or too low

---

### 13. Exam / Interview One-Liner

> Backpropagation is a gradient-based optimization algorithm that uses the chain rule to propagate error backward through the network and update weights to minimize loss.

---

### 14. Final Mental Model

> Forward propagation is **making a decision**; backward propagation is **learning from consequences**.
"""
	)

	# st.info(
	# 	"Next steps: full training loop (FP + BP + GD), XOR solved using MLP, "
	# 	"vanishing vs exploding gradients, why ReLU works, or backprop in code."
	# )
