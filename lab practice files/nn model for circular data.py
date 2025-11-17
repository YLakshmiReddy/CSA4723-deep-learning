import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import make_moons # Generates circular/non-linear data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- 1. Prepare Circular (Two Moons) Data ---
# n_samples=300, noise=0.1 for clear but non-linearly separable data
X, Y = make_moons(n_samples=300, noise=0.1, random_state=42)
N_features = X.shape[1]

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42, stratify=Y
)

# 2. Preprocessing: Scale the features (crucial for NNs)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 3. Construct and Compile NN Model (MLP) ---
# A deep network is necessary to learn the non-linear boundary

model = Sequential([
    # Input layer and first hidden layer (16 neurons)
    Dense(16, activation='relu', input_shape=(N_features,)),
    # Second hidden layer (8 neurons)
    Dense(8, activation='relu'),
    # Output layer for binary classification using SIGMOID
    Dense(1, activation='sigmoid')
])

# Compile the model using binary_crossentropy loss
model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# --- 4. Train the Model ---

print("--- Starting NN Model Training on Moons Data ---")

history = model.fit(
    X_train_scaled, Y_train,
    epochs=50,
    batch_size=8,
    validation_split=0.1, # Use 10% of training data for validation
    verbose=0
)

# --- 5. Performance Verification ---

# Predict probabilities on the test set
Y_prob = model.predict(X_test_scaled)

# Convert probabilities to binary class predictions (0 or 1)
Y_pred = (Y_prob > 0.5).astype(int)

# Calculate Key Metrics
accuracy = accuracy_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)
class_report = classification_report(Y_test, Y_pred, target_names=['Class 0', 'Class 1'])

# 6. Output Results
print("\n--- NN Model Performance Verification (Circular Data) ---")
print(f"\nOverall Test Accuracy: {accuracy:.4f}")

print("\n--- Confusion Matrix (Testing Data) ---")
conf_df = pd.DataFrame(
    conf_matrix,
    index=['Actual Class 0', 'Actual Class 1'],
    columns=['Predicted Class 0', 'Predicted Class 1']
)
print(conf_df)

print("\n--- Classification Report (Testing Data) ---")
print(class_report)

# Optional: Plotting the Decision Boundary for visual verification
def plot_decision_boundary(X, y, model, scaler):
    h = .02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Scale the grid points before prediction
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_scaled = scaler.transform(grid_points)

    Z = model.predict(grid_points_scaled)
    Z = (Z > 0.5).astype(int).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=plt.cm.RdYlBu)
    plt.title('NN Decision Boundary on Two Moons Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Use the test set for plotting
plot_decision_boundary(X_test, Y_test, model, scaler)