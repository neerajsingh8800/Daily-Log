# Phase 1: Mathematical Prerequisites for Deep Learning

To understand Deep Learning, you don't need to be a mathematician, but you do need to understand how data is structured and how it "moves" through a network.

---

## 1. Linear Algebra (The Data Structure)
In Deep Learning, data is stored in **Tensors**. A tensor is simply a multi-dimensional array.

* **Scalar:** A single number (0D Tensor).
* **Vector:** A list of numbers (1D Tensor).
* **Matrix:** A grid of numbers (2D Tensor).
* **Tensor:** $n$-dimensional arrays (3D and above).

### Key Operation: Dot Product
The Dot Product is how a neuron combines inputs ($x$) and weights ($w$).
If $A = [a_1, a_2]$ and $B = [b_1, b_2]$, the dot product is:
$$A \cdot B = \sum_{i=1}^{n} a_i b_i = a_1b_1 + a_2b_2$$

---

## 2. Calculus (The Engine of Learning)
Calculus tells us how to change the weights of a model to reduce the error.

### The Derivative
The derivative $f'(x)$ tells us the **slope** or the rate of change. If the slope is positive, the error is increasing; if negative, it's decreasing.

### The Chain Rule
This is the most important rule in Deep Learning. It allows us to calculate the gradient of a complex function by breaking it into pieces. This is the foundation of **Backpropagation**.
$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

---

## 3. Probability (The Uncertainty)
Models don't usually say "This is a cat." They say "I am 95% sure this is a cat."

### Softmax Function
Used in the output layer of a neural network to turn raw numbers (logits) into probabilities that sum up to 1.
$$\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

---

## 4. Summary Table for Quick Reference

| Concept | Simple Definition | Why it matters in DL? |
| :--- | :--- | :--- |
| **Tensors** | Multi-dim arrays | How we represent images/text. |
| **Matrix Mult** | Combining grids | How layers connect to each other. |
| **Gradient** | Direction of steepest ascent | Tells the model which way to "walk" to find the minimum error. |
| **Cross-Entropy** | Distance between two distributions | The standard loss function for classification. |

---
