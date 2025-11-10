import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# --- 1. Bi-Level (Binary) Confusion Matrix ---

# Sample Data (Binary Classification)
y_true_binary = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0])
y_pred_binary = np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0])

# Construct the Bi-Level Confusion Matrix
cm_binary = confusion_matrix(y_true_binary, y_pred_binary)

# Convert to DataFrame for better visualization
cm_df_binary = pd.DataFrame(
    cm_binary,
    index=[f'Actual {i}' for i in [0, 1]],
    columns=[f'Predicted {i}' for i in [0, 1]]
)

# Verify Performance (Classification Report)
report_binary = classification_report(y_true_binary, y_pred_binary, zero_division=0)

print("--- Bi-Level (Binary) Results ---")
print("\nConfusion Matrix (DataFrame):")
print(cm_df_binary)
print("\nPerformance Metrics (Classification Report):")
print(report_binary)

# --- 2. Multi-Level (Multi-Class) Confusion Matrix ---

# Sample Data (Multi-Class Classification with 3 classes: 0, 1, 2)
y_true_multi = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
y_pred_multi = np.array([0, 1, 1, 0, 1, 2, 0, 2, 2, 1, 1, 2])

# Construct the Multi-Level Confusion Matrix
cm_multi = confusion_matrix(y_true_multi, y_pred_multi)

# Convert to DataFrame for better visualization
cm_df_multi = pd.DataFrame(
    cm_multi,
    index=[f'Actual {i}' for i in [0, 1, 2]],
    columns=[f'Predicted {i}' for i in [0, 1, 2]]
)

# Verify Performance (Classification Report)
report_multi = classification_report(y_true_multi, y_pred_multi, zero_division=0)

print("\n--- Multi-Level (Multi-Class) Results ---")
print("\nConfusion Matrix (DataFrame):")
print(cm_df_multi)
print("\nPerformance Metrics (Classification Report):")
print(report_multi)
