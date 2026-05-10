# 05. Optimizers

Optimizers are algorithms or methods used to change the attributes of your neural network, such as weights and learning rate, to reduce the losses. Optimization is the process of finding the global minimum of the cost function.

---

## 1. Gradient Descent (The Foundation)
The goal is to update the weights $w$ in the direction that decreases the Loss Function $L$.

### Standard Weight Update Rule:
$$w_{t+1} = w_t - \eta \cdot \nabla L(w_t)$$

Where:
* $\eta$ (Eta): The **Learning Rate** (step size).
* $\nabla L(w_t)$: The **Gradient** (direction of steepest ascent).

---

## 2. Common Optimization Algorithms

### A. Stochastic Gradient Descent (SGD)
Updates weights after seeing every single training example.
* **Pros:** Fast, can escape local minima due to noise.
* **Cons:** High variance in updates causes the loss to fluctuate heavily.

### B. SGD with Momentum
Helps accelerate SGD in the relevant direction and dampens oscillations by adding a fraction $\gamma$ of the previous update to the current one.
* **Formula:** $$v_t = \gamma v_{t-1} + \eta \nabla L(w_t)$$
$$w_{t+1} = w_t - v_t$$

### C. RMSProp (Root Mean Squared Propagation)
Adapts the learning rate for each parameter. It divides the learning rate by an exponentially decaying average of squared gradients.
* **Formula:**
$$E[g^2]_t = 0.9 E[g^2]_{t-1} + 0.1 g_t^2$$
$$w_{t+1} = w_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t$$

### D. Adam (Adaptive Moment Estimation)
The "Gold Standard" in 2026. Adam combines the benefits of **Momentum** and **RMSProp**. It calculates an exponential moving average of the gradient (1st moment) and the squared gradient (2nd moment).
* **Pros:** Requires little tuning, handles sparse gradients, and is very computationally efficient.

---

## 3. Comparison of Optimizers

| Optimizer | Best For | Key Characteristic |
| :--- | :--- | :--- |
| **SGD** | Simple models | High noise, slow convergence. |
| **Momentum** | Deep networks | Overcomes plateaus and oscillations. |
| **RMSProp** | Recurrent Neural Networks | Normalizes gradients by magnitude. |
| **Adam** | General Deep Learning | Adaptive learning rates for every parameter. |

---

## 4. Implementation in Python (NumPy)

```python
import numpy as np

class Optimizers:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
        # For Momentum
        self.v = 0
        # For Adam/RMSProp
        self.m = 0
        self.s = 0
        self.t = 0

    def sgd(self, w, dw):
        return w - self.lr * dw

    def momentum(self, w, dw, gamma=0.9):
        self.v = gamma * self.v + self.lr * dw
        return w - self.v

    def adam(self, w, dw, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.t += 1
        # 1st moment (Momentum)
        self.m = beta1 * self.m + (1 - beta1) * dw
        # 2nd moment (RMSProp)
        self.s = beta2 * self.s + (1 - beta2) * (dw**2)
        
        # Bias correction
        m_hat = self.m / (1 - beta1**self.t)
        s_hat = self.s / (1 - beta2**self.t)
        
        return w - self.lr * m_hat / (np.sqrt(s_hat) + epsilon)

# Example usage
optimizer = Optimizers(learning_rate=0.1)
weight = 0.5
gradient = 0.05

new_weight = optimizer.adam(weight, gradient)
print(f"Old weight: {weight} -> New weight (Adam): {new_weight}")
```
