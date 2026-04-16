from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load the dataset (Features = X, Labels = y)
data = load_iris()
X, y = data.data, data.target

# 2. Split the data (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Choose the model and train it
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 4. Make predictions on the test set
predictions = model.predict(X_test)

# 5. Check how accurate the model is
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
