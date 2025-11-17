import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# 1. Load Data
data = load_iris()
X = data.data
y = data.target
target_names = data.target_names

# 2. Preprocessing and Splitting
# Scaling is optional for tree-based models like Random Forest, but included as good practice
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Model Training
# RandomForestClassifier is used with 100 trees (n_estimators)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Prediction
y_pred = model.predict(X_test)

# 5. Performance Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=target_names)

# 6. Output Results
print("--- Performance Evaluation of Random Forest Classifier (on Iris) ---")
print(f"\nOverall Test Accuracy: {accuracy:.4f}")
print("\n--- Confusion Matrix (Testing Data) ---")

# Create a DataFrame for clear matrix visualization
conf_df = pd.DataFrame(
    conf_matrix,
    index=[f'Actual {name}' for name in target_names],
    columns=[f'Predicted {name}' for name in target_names]
)
print(conf_df)
print("\n--- Classification Report (Testing Data) ---")
print(class_report)