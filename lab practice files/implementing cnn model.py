import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import os
import random

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = str(42)
random.seed(42)

# --- 1. Prepare Synthetic Data (Replace with your actual data loading) ---

# Define image parameters
IMG_WIDTH = 64
IMG_HEIGHT = 64
CHANNELS = 3  # RGB
NUM_SAMPLES = 200

# Create random input images (200 samples of 64x64x3)
X = np.random.rand(NUM_SAMPLES, IMG_HEIGHT, IMG_WIDTH, CHANNELS).astype('float32')

# Create binary labels (0 or 1)
y_binary = np.random.randint(0, 2, NUM_SAMPLES)

# Convert binary labels to one-hot encoding for Softmax (2 output neurons)
# Class 0 -> [1, 0], Class 1 -> [0, 1]
y = tf.keras.utils.to_categorical(y_binary, num_classes=2)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_binary
)

# --- 2. Construct the CNN Model ---

model = Sequential([
    # Convolutional Block 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
    MaxPooling2D((2, 2)),

    # Convolutional Block 2
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Flattening Layer
    Flatten(),

    # Dense Layer 1
    Dense(64, activation='relu'),

    # Output Layer with SOFTMAX
    # Output MUST have 2 neurons for Softmax-based binary classification
    Dense(2, activation='softmax')
])

# --- 3. Compile the Model ---

# Use categorical_crossentropy for a Softmax output layer
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print a summary of the model structure
model.summary()

# --- 4. Train the Model ---

print("\n--- Starting Model Training ---")

history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# --- 5. Verify Performance ---

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n--- Model Verification ---")
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Demonstrate prediction and output format
predictions = model.predict(X_test[:5])
print("\n--- Sample Predictions (Softmax Output) ---")
print("Raw Softmax Output (Probabilities for Class 0, Class 1):")
print(predictions)

# Convert Softmax probabilities back to class labels
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test[:5], axis=1)

print("\nPredicted Classes:", predicted_classes)
print("True Classes:     ", true_classes)