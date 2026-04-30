# Feature Engineering for Machine Learning

Feature engineering is the process of using domain knowledge to extract features (characteristics, properties, attributes) from raw data. These features can be used to improve the performance of machine learning algorithms.

> "Coming up with features is difficult, time-consuming, requires expert knowledge. 'Applied machine learning' is basically feature engineering." — *Andrew Ng*

---

## 1. Handling Missing Values
Missing data can bias results or cause errors in many algorithms (like SVM or Linear Regression).

### Numerical Imputation
```python
import pandas as pd
from sklearn.impute import SimpleImputer

# Mean/Median Imputation
imputer = SimpleImputer(strategy='median')
df['age'] = imputer.fit_transform(df[['age']])
```
### Categorical Imputation
# Imputing with the most frequent value (Mode)
```python
imputer = SimpleImputer(strategy='most_frequent')
df['city'] = imputer.fit_transform(df[['city']])
```

## 2.Categorical Encoding
Machine Learning models only understand numbers. We must convert text categories into numerical format.

### One-Hot Encoding (Nominal Data)
Use this for data with no inherent order (e.g., Colors, Countries).
```python
df = pd.get_dummies(df, columns=['color'], prefix='color')
```

### Label/Ordinal Encoding (Ordinal Data)
Use this when order matters (e.g., "Small", "Medium", "Large").
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['education_level'] = le.fit_transform(df['education_level'])
```

## 3.Numerical Variable Transformation
Changing the distribution of data to help models converge faster.

### Log Transformation
Helps handle skewed data and reduces the impact of outliers.
```python
import numpy as np
df['fare_log'] = np.log1p(df['fare']) # log1p handles log(0) by adding 1
```

### Feature Scaling
* Standardization (Z-score): Centers data around mean 0 and standard deviation 1.

* Normalization (Min-Max): Scales data between 0 and 1.
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
```

# Standardization
```python
scaler = StandardScaler()
df['salary_scaled'] = scaler.fit_transform(df[['salary']])
```


## 4.Feature Creation (Domain Specific)
Creating new features from existing ones to capture complex relationships.

### Date/Time Features
```python
df['date'] = pd.to_datetime(df['date'])
df['hour'] = df['date'].dt.hour
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
```

### Polynomial Features
Creating interaction terms (e.g., $x_1 \times x_2$).
```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, interaction_only=True)
interactions = poly.fit_transform(df[['feature_A', 'feature_B']])
```

## 5. Outlier Handling
Outliers can significantly skew models like Linear Regression.

*Trimming: Removing the outliers.

*Capping (Winsorization): Setting outliers to a maximum/minimum threshold (e.g., the 99th percentile).
# Capping using IQR
```python
upper_limit = df['revenue'].quantile(0.95)
lower_limit = df['revenue'].quantile(0.05)

df['revenue'] = np.where(df['revenue'] > upper_limit, upper_limit,
                np.where(df['revenue'] < lower_limit, lower_limit, df['revenue']))
```

## 6. Feature Selection
Not all features are useful. Selecting the best ones reduces overfitting and training time.

*Correlation Heatmaps: Remove highly correlated features (redundancy).

*Feature Importance: Using Random Forest or XGBoost to see which features contribute most.

*Recursive Feature Elimination (RFE): Automatically removing the least important features.
```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
rfe = RFE(model, n_features_to_select=5)
fit = rfe.fit(X, y)
print("Selected Features: %s" % fit.support_)
```







