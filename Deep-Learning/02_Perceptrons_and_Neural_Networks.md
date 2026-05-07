# 02. Perceptrons and Neural Networks

This module explains the transition from the simplest unit of deep learning—the Perceptron—to the complex Multi-Layer Perceptron (MLP) architectures that power modern AI.

---

## 1. The Perceptron: The Fundamental Building Block
The Perceptron is a linear classifier. It is the simplest artificial neural network, consisting of a single neuron that makes predictions by calculating a weighted sum of inputs.

### Mathematical Model
For an input vector $\mathbf{x}$, the perceptron computes:

$$z = \sum_{i=1}^{n} w_i x_i + b$$

Where:
* $w$: **Weights** (represent the importance of the input).
* $b$: **Bias** (shifts the decision boundary).
* $z$: **Logit** (the raw output before activation).

### The Decision Rule
The original perceptron used a **Heaviside Step Function**:
$$\hat{y} = \begin{cases} 1 & \text{if } z \ge 0 \\ 0 & \text{if } z < 0 \end{cases}$$

> **Limitation:** A single perceptron can only solve **Linearly Separable** problems. It famously fails the **XOR Gate** problem because a single straight line cannot separate the classes.

---

## 2. Multi-Layer Perceptron (MLP)
To solve non-linear problems, we stack perceptrons into layers, creating a **Neural Network**.

### Network Architecture
1.  **Input Layer:** Passes the features to the hidden layers. No computation happens here.
2.  **Hidden Layers:** The "engine" of the network where feature extraction occurs.
3.  **Output Layer:** Maps the hidden representations to the desired output format (e.g., a probability).

### The Forward Pass Formula
In a network, the output of one layer becomes the input of the next:
$$\mathbf{a}^{(l)} = \sigma(\mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)})$$
Where $\sigma$ is a non-linear activation function (like ReLU or Sigmoid) that allows the network to learn complex patterns.

---

## 3. Comparison Table

| Feature | Perceptron | Neural Network (MLP) |
| :--- | :--- | :--- |
| **Layers** | Single | Multiple (Input, Hidden, Output) |
| **Decision Boundary** | Linear (Straight line) | Non-linear (Complex curves) |
| **Activation Function** | Step Function | ReLU, Sigmoid, Tanh, etc. |
| **Applications** | Simple Logic Gates (AND, OR) | Image Recognition, NLP, Regression |

---

## 4. Implementation: Perceptron vs. MLP Logic
The following code demonstrates a manual implementation of a 2-layer Neural Network logic using NumPy.

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Weights for Layer 1 (Input to Hidden)
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        
        # Weights for Layer 2 (Hidden to Output)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def relu(self, z):
        return np.max(0, z)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        # Layer 1: Linear + ReLU Activation
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # Layer 2: Linear + Sigmoid Activation
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2

# Simulation
nn = NeuralNetwork(input_size=3, hidden_size=4, output_size=1)
sample_input = np.array([[0.1, 0.5, -0.2]])
output = nn.forward(sample_input)

print(f"Input: {sample_input}")
print(f"Network Output (Probability): {output}")
```
