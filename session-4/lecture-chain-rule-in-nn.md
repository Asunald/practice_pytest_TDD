# Backpropagation and the Chain Rule

## Updating Weights and Biases in Neural Networks

---

# 1. What Happens After the Forward Pass?

In the **forward pass**, a neural network takes an input $x$, passes it through layers, and produces a **prediction** $\hat{y}$.

We then compare the prediction with the **true label** $y$ using a **loss function**:

Example (Mean Squared Error):

$$
L = \frac{1}{2} (\hat{y} - y)^2
$$

Or for classification:

Cross-Entropy (general multi-class):

$$
L = -\sum y \log(\hat{y})
$$

Binary Cross-Entropy (BCE, for 0/1 labels):

$$
L = - \Big( y \log(\hat{y}) + (1-y) \log(1-\hat{y}) \Big)
$$

The goal is:

> **Minimize the loss by adjusting weights and biases.**


---

# 2. What is Backpropagation?

Backpropagation is the process of:

1. Calculating **how much each parameter contributes to the error**
2. Sending this information **backward through the network**
3. Updating parameters using **gradient descent**

Training cycle:

```
Forward Pass
     ↓
Compute Loss
     ↓
Backpropagation
     ↓
Update Weights & Biases
```

---

# 3. Gradient Descent Review

We update parameters using the gradient:

$$
\theta = \theta - \eta \frac{\partial L}{\partial \theta}
$$

Where:

* $\theta$ = parameter (weight or bias)
* $\eta$ = learning rate
* $\frac{\partial L}{\partial \theta}$ = gradient

For neural networks we must compute:

$$
\frac{\partial L}{\partial W}
$$

for **every weight and bias**.

---

# 4. The Problem: Many Layers

Consider a simple neural network:

$$
\text{Input} \rightarrow \text{Hidden Layer} \rightarrow \text{Output}
$$

Hidden layer:

$$
z_1 = W_1 x + b_1
$$

$$
a_1 = f(z_1)
$$

Output layer:

$$
z_2 = W_2 a_1 + b_2
$$

$$
\hat{y} = f(z_2)
$$

Loss:

$$
L(\hat{y}, y)
$$

We want:

$$
\frac{\partial L}{\partial W_1}
$$

But the loss depends on many intermediate variables. This is why we use the **Chain Rule**.

---

# 5. Simple Example: Understanding the Chain Rule

Suppose you have:

$$
y = u^2
$$

$$
u = 3x + 1
$$

You want to find $dy/dx$ (how $y$ changes with respect to $x$).

**Method 1: Direct substitution**

$$
y = (3x + 1)^2
$$

$$
\frac{dy}{dx} = 2(3x + 1) \cdot 3 = 6(3x + 1)
$$

**Method 2: Chain Rule**

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}
$$

$$
\frac{dy}{du} = 2u = 2(3x + 1)
$$

$$
\frac{du}{dx} = 3
$$

$$
\frac{dy}{dx} = 2(3x + 1) \cdot 3 = 6(3x + 1)
$$

**Same answer!** The chain rule breaks complex derivatives into simpler pieces.

---

# 6. Why This Matters for Neural Networks

Neural networks are just **chains of functions**:

$$
\text{Input} \rightarrow \text{Layer 1} \rightarrow \text{Layer 2} \rightarrow ... \rightarrow \text{Output} \rightarrow \text{Loss}
$$

To find how the loss changes with respect to weights in Layer 1, we need to **chain through all the layers in between**. That’s where the **chain rule** comes in!

**General Chain Rule Formula:**

$$
\text{For a chain of functions: } y = f(g(h(x)))
$$

$$
\frac{dy}{dx} = \frac{df}{dg} \cdot \frac{dg}{dh} \cdot \frac{dh}{dx}
$$

Each derivative tells us:

> “If I change this input a little bit, how much does the output change?”

---



# 7. Output Layer Gradient

Let’s compute the gradient for the output layer. Suppose we have a **binary classification problem** with **Binary Cross-Entropy (BCE)** and a **sigmoid output**:

$$
L = - \Big( y \log(\hat{y}) + (1-y) \log(1-\hat{y}) \Big), \quad \hat{y} = \sigma(z_2)
$$

Step 1: Derivative of the Loss w.r.t. $\hat{y}$

$$
\frac{\partial L}{\partial \hat{y}} = - \left( \frac{y}{\hat{y}} - \frac{1-y}{1-\hat{y}} \right)
$$

This tells us **how sensitive the loss is to the predicted probability**.

Step 2: Derivative of $\hat{y}$ w.r.t. $z_2$ (sigmoid activation)

For $\hat{y} = \sigma(z_2)$:

$$
\frac{\partial \hat{y}}{\partial z_2} = \sigma(z_2)(1 - \sigma(z_2)) = \hat{y}(1 - \hat{y})
$$

Step 3: Output Layer Error Term $\delta_2$

Using the **chain rule**:

$$
\delta_2 = \frac{\partial L}{\partial z_2} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_2}
$$

**Interpretation:** $\delta_2$ represents **how much the output of this layer contributed to the final loss**. It’s the “error signal” sent backward.

---

# 8. Gradients for Output Weights and Bias

The output layer equation:

$$
z_2 = W_2 a_1 + b_2
$$

Using calculus:

* **Weight gradients**:

$$
\frac{\partial L}{\partial W_2} = \delta_2 \cdot a_1^T
$$

* **Bias gradients**:

$$
\frac{\partial L}{\partial b_2} = \delta_2
$$

**Interpretation:**

* Each weight gradient shows **how much changing this weight will change the loss**.
* Each bias gradient shows **how much changing this bias will change the loss**.

---

# 9. Backpropagating to the Hidden Layer

The error for the hidden layer is computed by **propagating the output error backward**:

$$
\delta_1 = (W_2^T \delta_2) \odot f'(z_1)
$$

Where:

* $W_2^T \delta_2$ propagates the output error to the hidden layer
* $f'(z_1)$ is the derivative of the hidden activation function
* $\odot$ denotes **element-wise multiplication**

Then, the hidden layer gradients:

* **Weights:**

$$
\frac{\partial L}{\partial W_1} = \delta_1 \cdot x^T
$$

* **Biases:**

$$
\frac{\partial L}{\partial b_1} = \delta_1
$$

**Interpretation:** Each hidden weight receives credit/blame proportional to **how much it contributed to the output error**.

---

# 10. Updating Parameters (Gradient Descent)

Once we have all gradients, update weights and biases:

$$
W \leftarrow W - \eta \frac{\partial L}{\partial W}
$$

$$
b \leftarrow b - \eta \frac{\partial L}{\partial b}
$$

Where $\eta$ is the **learning rate**.

**Intuition:** Move each parameter **slightly in the direction that reduces the loss**.

---

# 11. Key Takeaways

1. Forward pass → compute activations → predict output
2. Compute loss → measure prediction error
3. Backpropagation → use **chain rule** to compute gradients layer by layer
4. Update weights & biases → **gradient descent**

**Core Concept:** Backpropagation is **walking backward through the computation graph**, passing error signals and adjusting parameters **proportionally to their contribution to the error**.
