import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# 1. Generate Synthetic Numerical Database (for binary classification)
np.random.seed(42)
N_samples = 200
# Feature 1 and Feature 2
X1 = np.random.normal(loc=10, scale=3, size=N_samples)
X2 = np.random.normal(loc=50, scale=10, size=N_samples)

# Target (y): Binary outcome (0 or 1) based on a linear boundary with noise.
linear_combination = X1 + X2 / 10 + np.random.normal(0, 1, N_samples)
y = (linear_combination > 15).astype(int)

# Create DataFrame
data = pd.DataFrame({'Feature1': X1, 'Feature2': X2, 'Target': y})

# 2. Prepare Data and Split
X = data[['Feature1', 'Feature2']]
y = data['Target']

# Split data into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Train the Logistic Regression Model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# 4. Predict on Test Set
y_pred = model.predict(X_test)

# 5. Analyze Performance

# Calculate Classification Metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, zero_division=0)

# Retrieve Model Parameters
intercept = model.intercept_[0]
coefficients = model.coef_[0]

# Print Results
print("--- Logistic Regression Performance Analysis ---")
print("\n--- Model Parameters ---")
print(f"Intercept (Bias): {intercept:.4f}")
print(f"Coefficient (Feature1): {coefficients[0]:.4f}")
print(f"Coefficient (Feature2): {coefficients[1]:.4f}")
print("\n--- Confusion Matrix (Testing Data) ---")
print("           Predicted 0   Predicted 1")
print(f"Actual 0:  {conf_matrix[0, 0]:<12}{conf_matrix[0, 1]:<12}")
print(f"Actual 1:  {conf_matrix[1, 0]:<12}{conf_matrix[1, 1]:<12}")
print("\n--- Classification Report (Testing Data) ---")
print(class_report)
print(f"\nOverall Accuracy: {accuracy:.4f}")
