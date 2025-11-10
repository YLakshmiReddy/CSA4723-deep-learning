import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Choose/Generate a Numerical Database (Synthetic Data for demonstration)
np.random.seed(42)
N_samples = 100
# Feature (X): Random values from 0 to 10
X = 10 * np.random.rand(N_samples, 1)
# Target (y): Linear relationship y = 2*X + 5 + noise
y = 2 * X.ravel() + 5 + np.random.normal(0, 2, N_samples)

# Convert to DataFrame for standard practice
data = pd.DataFrame({'Feature': X.ravel(), 'Target': y})

# 2. Prepare Data and Split
X = data[['Feature']]
y = data['Target']

# Split data into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Predict on Test Set
y_pred = model.predict(X_test)

# 5. Analyze Performance and Output Results

# Retrieve Model Parameters
intercept = model.intercept_
coefficient = model.coef_[0]

# Calculate Key Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
y_train_pred = model.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)

print("--- Linear Regression Performance Analysis ---")
print(f"\nModel Equation: Y = {coefficient:.2f} * X + {intercept:.2f}")
print("\n--- Performance Metrics (Testing Data) ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2 Score): {r2:.4f}")
print(f"R-squared (Training Data): {train_r2:.4f}")
