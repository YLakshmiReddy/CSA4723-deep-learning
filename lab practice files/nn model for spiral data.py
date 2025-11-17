import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- 1. Prepare Spiral Data Generator Function ---
def generate_spiral_data(samples_per_class, num_classes):
    """Generates the spiral dataset."""
    N = samples_per_class
    K = num_classes
    X = np.zeros((N * K, 2)) # data matrix (features)
    y = np.zeros(N * K, dtype='uint8') # class labels
    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N) # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2 # theta
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = j
    return X, y

# Generate 3-class spiral data (300 samples total)
X, Y = generate_spiral_data(samples_per_class=100, num_classes=3)
N_features = X.shape[1]
NUM_CLASSES = 3

# Convert integer labels to one-hot encoding (required for Softmax output)
Y_cat = tf.keras.utils.to_categorical(Y, num_classes=NUM_CLASSES)

# Split data
X_train, X_test, Y_train_cat, Y_test_cat = train_test_split(
    X, Y_cat, test_size=0.3, random_state=42, stratify=Y
)
# Keep integer test labels for final metric calculation
_, _, _, Y_test_int = train_test_split(
    X, Y, test_size=0.3, random_state=42, stratify=Y
)

# 2. Preprocessing: Scale the features (crucial for NNs)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 3. Construct and Compile NN Model (MLP) ---
# A deep, wide network is required to solve the spiral data effectively

model = Sequential([
    # Input layer and first hidden layer (128 neurons)
    Dense(128, activation='relu', input_shape=(N_features,)),
    # Second hidden layer
    Dense(128, activation='relu'),
    # Third hidden layer
    Dense(64, activation='relu'),
    # Output layer for multi-class classification using SOFTMAX
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model using categorical_crossentropy loss
model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- 4. Train the Model ---

print("--- Starting NN Model Training on Spiral Data ---")

history = model.fit(
    X_train_scaled, Y_train_cat,
    epochs=100, # Increased epochs needed for complex data
    batch_size=32,
    validation_split=0.1,
    verbose=0
)

# --- 5. Performance Verification ---

# Predict probabilities on the test set
Y_prob = model.predict(X_test_scaled)

# Convert Softmax probabilities to class predictions
Y_pred_int = np.argmax(Y_prob, axis=1)

# Calculate Key Metrics
accuracy = accuracy_score(Y_test_int, Y_pred_int)
conf_matrix = confusion_matrix(Y_test_int, Y_pred_int)
class_report = classification_report(Y_test_int, Y_pred_int, zero_division=0)

# 6. Output Results
print("\n--- NN Model Performance Verification (Spiral Data) ---")
print(f"\nOverall Test Accuracy: {accuracy:.4f}")

print("\n--- Confusion Matrix (Testing Data) ---")
conf_df = pd.DataFrame(
    conf_matrix,
    index=[f'Actual Class {i}' for i in range(NUM_CLASSES)],
    columns=[f'Predicted Class {i}' for i in range(NUM_CLASSES)]
)
print(conf_df)

print("\n--- Classification Report (Testing Data) ---")
print(class_report)

# Optional: Plotting the Decision Boundary for visual verification
def plot_decision_boundary(X, Y_true, model, scaler):
    h = .02
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_scaled = scaler.transform(grid_points)

    Z = model.predict(grid_points_scaled)
    Z = np.argmax(Z, axis=1).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.6, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=Y_true, edgecolors='k', marker='o', cmap=plt.cm.RdYlBu)
    plt.title('NN Decision Boundary on Spiral Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Use the test set for plotting
plot_decision_boundary(X_test, Y_test_int, model, scaler)