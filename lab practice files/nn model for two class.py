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

# --- 1. Prepare Two-Class (Binary) Data (Synthetic for demonstration) ---
N_samples = 200
N_features = 5

# Create input features (X)
X = np.random.randn(N_samples, N_features)

# Create binary target labels (Y) with a non-linear relationship + noise
# Target Y = 1 if the sum of squares of first two features is large
Y = ((X[:, 0]**2 + X[:, 1]**2 + np.random.rand(N_samples)) > 1.5).astype(int)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42, stratify=Y
)

# 2. Preprocessing: Scale the features (crucial for NNs)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 3. Construct and Compile NN Model (MLP) ---

model = Sequential([
    # Input layer and first hidden layer
    Dense(10, activation='relu', input_shape=(N_features,)),
    # Second hidden layer
    Dense(5, activation='relu'),
    # Output layer for binary classification using SIGMOID
    # Sigmoid outputs a probability between 0 and 1
    Dense(1, activation='sigmoid')
])

# Compile the model using binary_crossentropy loss
model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# --- 4. Train the Model ---

print("--- Starting NN Model Training ---")

history = model.fit(
    X_train_scaled, Y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1, # Use 10% of training data for validation
    verbose=0
)

# --- 5. Performance Verification ---

# Predict probabilities on the test set
Y_prob = model.predict(X_test_scaled)

# Convert probabilities to binary class predictions (0 or 1) using a 0.5 threshold
Y_pred = (Y_prob > 0.5).astype(int)

# Calculate Key Metrics
accuracy = accuracy_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)
class_report = classification_report(Y_test, Y_pred, target_names=['Class 0', 'Class 1'])

# 6. Output Results
print("\n--- NN Model Performance Verification (Two-Class Data) ---")
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

# Optional: Plotting convergence
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Convergence')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()