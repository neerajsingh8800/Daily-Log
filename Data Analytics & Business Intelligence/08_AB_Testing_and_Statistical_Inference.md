# Module 08: A/B Testing and Statistical Inference

While Business Intelligence (BI) dashboards excel at showing *what* happened, Statistical Inference is required to prove *why* it happened and whether a change was actually effective. 

In product analytics, **A/B Testing** is the gold standard. It allows companies to test two versions of a product against each other to determine which performs better, using statistical mathematics to separate genuine business impact from random variance.

---

## 1. The Core Theory of Hypothesis Testing

Before running a test, we must define our assumptions. In statistics, we always assume that any change we make will have *no effect* until proven otherwise.

### The Hypotheses
* **Null Hypothesis ($H_0$):** The assumption that there is no statistical difference between Variant A (Control) and Variant B (Treatment). Any observed difference is just random luck.
* **Alternative Hypothesis ($H_A$):** The assumption that there *is* a statistically significant difference between the variants caused by the changes made.

### The P-Value and Significance Level ($\alpha$)
* **Significance Level ($\alpha$):** The threshold for proof, typically set at $0.05$ (5%). This means we are willing to accept a 5% risk of concluding a difference exists when there actually isn't one (a False Positive).
* **P-Value:** The probability of observing the test results if the Null Hypothesis ($H_0$) is true. 
    * If $P \leq \alpha$: We **reject** the Null Hypothesis. The result is statistically significant.
    * If $P > \alpha$: We **fail to reject** the Null Hypothesis. The result is just noise.

---

## 2. The Mathematics: Two-Sample Z-Test for Proportions

The most common A/B test in e-commerce is testing **Conversion Rates** (e.g., clicking a "Buy Now" button). Because a user either converts (1) or doesn't (0), we use a Two-Sample Z-Test for Proportions.

### The Formulas

First, calculate the **Pooled Proportion** ($\hat{p}$), which is the overall conversion rate across both groups:

$$
\hat{p} = \frac{X_{A} + X_{B}}{N_{A} + N_{B}}
$$

*(Where $X$ is the number of conversions, and $N$ is the total number of visitors in each group).*

Next, calculate the **Standard Error** (SE) of the difference between the two groups:

$$
SE = \sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{N_A} + \frac{1}{N_B}\right)}
$$

Finally, calculate the **Z-Score**, which tells us how many standard deviations the difference is from zero:

$$
Z = \frac{\hat{p}_B - \hat{p}_A}{SE}
$$

If the resulting $Z$-score corresponds to a P-value less than our $\alpha$ (0.05), we have a statistically significant winner!

---

## 3. Real-World Use Case: E-Commerce Checkout Optimization

**The Scenario:** Your product team wants to change the checkout button from Blue (Variant A - Control) to Green (Variant B - Treatment). 
* **Variant A** had 10,000 visitors and 500 conversions (5.0% rate).
* **Variant B** had 10,000 visitors and 560 conversions (5.6% rate).

A BI dashboard shows a 0.6% lift. But is it real, or just a lucky week for the Green button? We must run the test.

---

## 4. Implementation Example: Python

Instead of calculating this by hand, Data Scientists and Product Analysts use Python libraries like `statsmodels` to compute the Z-test and P-value instantly.

```python
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest

# 1. Define the test data
# Conversions for Variant A (Control) and Variant B (Treatment)
conversions = np.array([500, 560]) 

# Total visitors for Variant A and Variant B
visitors = np.array([10000, 10000])

# 2. Run the Two-Sample Z-Test
z_score, p_value = proportions_ztest(count=conversions, nobs=visitors, alternative='two-sided')

# 3. Output the Results
print(f"Z-Score: {z_score:.4f}")
print(f"P-Value: {p_value:.4f}")

# 4. Business Logic Interpretation
alpha = 0.05
if p_value < alpha:
    print("\nResult: Reject the Null Hypothesis.")
    print("Conclusion: The Green button generated a statistically significant increase in conversions. Deploy to production.")
else:
    print("\nResult: Fail to reject the Null Hypothesis.")
    print("Conclusion: The difference is likely due to random chance. Keep the Blue button.")
```
