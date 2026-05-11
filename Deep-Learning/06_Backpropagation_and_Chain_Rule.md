# 06. Backpropagation and the Chain Rule

Backpropagation is the shorthand for "backward propagation of errors." it is the algorithm used to calculate the gradient of the loss function with respect to the weights of the network.

---

## 1. The Core Logic: The Chain Rule
To update a weight in an early layer, we need to know how a change in that weight affects the final loss. Since the weight is nested deep inside functions (activations and summations), we use the **Chain Rule** from calculus.

If $y = f(g(x))$, then the derivative of $y$ with respect to $x$ is:
$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$$

In a Neural Network, to find the gradient of the Loss ($L$) with respect to a weight ($w$):
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}$$

---

## 2. The Four Fundamental Equations
To perform backpropagation, we calculate these gradients in reverse order (from output to input):

1.  **Error in Output Layer ($\delta^L$):**
    $$\delta^L = \nabla_a L \odot \sigma'(z^L)$$
2.  **Error in Hidden Layer ($\delta^l$):**
    $$\delta^l = ((w^{l+1})^T \delta^{l+1}) \odot \sigma'(z^l)$$
3.  **Gradient of Loss w.r.t Bias:**
    $$\frac{\partial L}{\partial b^l} = \delta^l$$
4.  **Gradient of Loss w.r.t Weight:**
    $$\frac{\partial L}{\partial w^l} = \delta^l (a^{l-1})^T$$

---

## 3. The Backpropagation Steps
1.  **Forward Pass:** Compute the activations for every layer until you get the output $\hat{y}$.
2.  **Compute Output Error:** Calculate how much the output layer's prediction differs from the target.
3.  **Backpropagate Error:** Move backward through the layers, calculating the "responsibility" each neuron has for the error.
4.  **Update Weights:** Use an Optimizer (like SGD) to adjust the weights using the calculated gradients.

---

## 4. Implementation in Python (NumPy)

This script demonstrates a single backpropagation step for a 2-layer network.

```python
import numpy as np

# Activation and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize data
X = np.array([[0.5, 0.1]]) # Input
y = np.array([[1.0]])      # Target

# Weights
W1 = np.random.randn(2, 3)
W2 = np.random.randn(3, 1)

# --- FORWARD PASS ---
z1 = np.dot(X, W1)
a1 = sigmoid(z1)
z2 = np.dot(a1, W2)
a2 = sigmoid(z2) # Final Prediction

# --- BACKWARD PASS ---
# 1. Error at output
error = y - a2
d_output = error * sigmoid_derivative(a2)

# 2. Error at hidden layer
error_hidden = d_output.dot(W2.T)
d_hidden = error_hidden * sigmoid_derivative(a1)

# --- WEIGHT UPDATES ---
learning_rate = 0.1
W2 += a1.T.dot(d_output) * learning_rate
W1 += X.T.dot(d_hidden) * learning_rate

print(f"Prediction after one step: {a2}")
```
