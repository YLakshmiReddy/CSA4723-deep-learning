import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 1. Load and Prepare Data
data = load_iris()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Define pipeline for consistent scaling and classification
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(max_iter=500, random_state=42))
])

# --- 2. Define Hyperparameter Search Space ---
# Parameters must be specified for the steps in the pipeline (e.g., 'mlp__param')
param_grid = {
    'mlp__hidden_layer_sizes': [(10,), (50,), (10, 10), (100,)], # Number of neurons/layers
    'mlp__activation': ['relu', 'tanh'],                        # Activation function
    'mlp__learning_rate_init': [0.001, 0.01],                   # Learning rate
    'mlp__solver': ['adam']                                     # Optimization algorithm
}

# 3. Perform Hyperparameter Tuning using GridSearchCV
print("--- Starting GridSearchCV for Hyperparameter Tuning ---")
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5, # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1, # Use all available cores
    verbose=1
)

grid_search.fit(X_train, y_train)

# --- 4. Verification: Compare Baseline Model vs. Tuned Model ---

# A. Baseline Model (Default parameters)
baseline_model = MLPClassifier(random_state=42, max_iter=500)
baseline_pipeline = Pipeline([('scaler', StandardScaler()), ('mlp', baseline_model)])
baseline_pipeline.fit(X_train, y_train)
baseline_pred = baseline_pipeline.predict(X_test)
baseline_accuracy = accuracy_score(y_test, baseline_pred)

# B. Tuned Model (Best parameters from Grid Search)
tuned_pipeline = grid_search.best_estimator_
tuned_pred = tuned_pipeline.predict(X_test)
tuned_accuracy = accuracy_score(y_test, tuned_pred)

# 5. Output Results
print("\n--- Hyperparameter Tuning Results ---")
print(f"Best Parameters Found: {grid_search.best_params_}")
print(f"Cross-Validation Accuracy (Best Model): {grid_search.best_score_:.4f}")

print("\n--- Performance Verification (Change in Output) ---")
print(f"Baseline Model Test Accuracy: {baseline_accuracy:.4f}")
print(f"Tuned Model Test Accuracy: {tuned_accuracy:.4f}")
print(f"Accuracy Change: {(tuned_accuracy - baseline_accuracy):.4f}")

# Optional: Full report for the best model
print("\nClassification Report for Tuned Model:")
print(classification_report(y_test, tuned_pred, target_names=data.target_names))