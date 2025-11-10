import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

# 1. Generate Synthetic Numerical Data (Simulating a Chosen Database)
# We use a non-linear function (sine wave) plus noise to ensure a polynomial model can overfit.
np.random.seed(42)
N_samples = 100
X = np.sort(5 * np.random.rand(N_samples, 1), axis=0)
y_true = np.sin(X).ravel() + np.random.normal(0, 0.2, N_samples) # Target variable with noise

# 2. Split Data into Training and Testing Sets (Essential for verification)
X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.3, random_state=42)

# 3. Model Training and Evaluation (Verification)

# --- a) Low Complexity Model (Degree 1: Linear Regression) ---
# Used as a baseline for comparison.
degree_low = 1
model_low = make_pipeline(PolynomialFeatures(degree_low), LinearRegression())
model_low.fit(X_train, y_train)

# Calculate Errors
train_error_low = mean_squared_error(y_train, model_low.predict(X_train))
test_error_low = mean_squared_error(y_test, model_low.predict(X_test))

# --- b) High Complexity Model (Degree 15: High-Order Polynomial) ---
# Used to intentionally induce and verify overfitting.
degree_high = 15
model_high = make_pipeline(PolynomialFeatures(degree_high), LinearRegression())
model_high.fit(X_train, y_train)

# Calculate Errors
train_error_high = mean_squared_error(y_train, model_high.predict(X_train))
test_error_high = mean_squared_error(y_test, model_high.predict(X_test))

# 4. Output Results for Overfitting Verification
print("--- Overfitting Verification (Polynomial Regression) ---")
print("\nLow Complexity Model (Degree 1):")
print(f"Training Mean Squared Error (MSE): {train_error_low:.4f}")
print(f"Testing Mean Squared Error (MSE):  {test_error_low:.4f}")
print("-" * 30)
print("\nHigh Complexity Model (Degree 15 - Overfit Test):")
print(f"Training Mean Squared Error (MSE): {train_error_high:.4f}")
print(f"Testing Mean Squared Error (MSE):  {test_error_high:.4f}")
print("-" * 30)

# Verification Logic
if train_error_high < train_error_low and test_error_high > test_error_low:
    print("\n✅ Verification Successful: **Overfitting Detected**.")
    print("The high-complexity model has much lower training error but significantly higher testing error.")
elif train_error_high < test_error_high and abs(train_error_high / test_error_high) < 0.5:
    print("\n⚠️ Potential Overfitting: Training error is much lower than Testing error.")
else:
    print("\nℹ️ No significant overfitting detected for the chosen seed/data split.")
