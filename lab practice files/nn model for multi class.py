import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- 1. Prepare Multi-Class Data (Iris Dataset) ---
data = load_iris()
X = data.data
y = data.target
target_names = data.target_names
NUM_CLASSES = len(target_names)

# Convert integer labels to one-hot encoding (required for Softmax output)
Y_cat = to_categorical(y, num_classes=NUM_CLASSES)

# Split data
X_train, X_test, Y_train_cat, Y_test_cat = train_test_split(
    X, Y_cat, test_size=0.3, random_state=42, stratify=y
)
# Keep integer test labels for final metric calculation
_, _, _, Y_test_int = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 2. Preprocessing: Scale the features (crucial for NNs)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 3. Construct and Compile NN Model (MLP) ---

model = Sequential([
    # Input layer and first hidden layer
    Dense(10, activation='relu', input_shape=(X.shape[1],)),
    # Second hidden layer
    Dense(10, activation='relu'),
    # Output layer for multi-class classification using SOFTMAX
    # Softmax outputs a probability distribution over the NUM_CLASSES
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model using categorical_crossentropy loss
model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- 4. Train the Model ---

print("--- Starting NN Model Training ---")

history = model.fit(
    X_train_scaled, Y_train_cat,
    epochs=50,
    batch_size=8,
    validation_split=0.1, # Use 10% of training data for validation
    verbose=0
)

# --- 5. Performance Verification ---

# Predict probabilities on the test set
Y_prob = model.predict(X_test_scaled)

# Convert Softmax probabilities to class predictions (the index with the highest probability)
Y_pred_int = np.argmax(Y_prob, axis=1)

# Calculate Key Metrics
accuracy = accuracy_score(Y_test_int, Y_pred_int)
conf_matrix = confusion_matrix(Y_test_int, Y_pred_int)
class_report = classification_report(Y_test_int, Y_pred_int, target_names=target_names)

# 6. Output Results
print("\n--- NN Model Performance Verification (Multi-Class Data) ---")
print(f"\nOverall Test Accuracy: {accuracy:.4f}")

print("\n--- Confusion Matrix (Testing Data) ---")
conf_df = pd.DataFrame(
    conf_matrix,
    index=[f'Actual {name}' for name in target_names],
    columns=[f'Predicted {name}' for name in target_names]
)
print(conf_df)

print("\n--- Classification Report (Testing Data) ---")
print(class_report)

# Optional: Plotting convergence
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Convergence')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()