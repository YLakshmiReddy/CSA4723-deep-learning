import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import random
import os

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = str(42)
random.seed(42)

# --- 1. Data Preparation (Synthetic Binary Image Data) ---

IMG_WIDTH = 32
IMG_HEIGHT = 32
CHANNELS = 3
NUM_SAMPLES = 500
NUM_CLASSES = 2

# Create random input images
X = np.random.rand(NUM_SAMPLES, IMG_HEIGHT, IMG_WIDTH, CHANNELS).astype('float32')
# Create binary labels (0 or 1)
y_binary = np.random.randint(0, NUM_CLASSES, NUM_SAMPLES)
# Use sigmoid for the final layer, so keep labels as single column
y = y_binary

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# --- 2. Define CNN Construction Function ---

def create_cnn_model(activation_func='relu', learning_rate=0.001, optimizer_name='Adam'):
    """Constructs a basic CNN model with configurable parameters."""

    # Select Optimizer
    if optimizer_name == 'Adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == 'SGD':
        optimizer = SGD(learning_rate=learning_rate)
    elif optimizer_name == 'RMSprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError("Invalid optimizer name.")

    model = Sequential([
        Conv2D(32, (3, 3), activation=activation_func, input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation=activation_func),
        MaxPooling2D((2, 2)),

        Flatten(),

        Dense(64, activation=activation_func),

        # Output layer for binary classification using SIGMOID
        Dense(1, activation='sigmoid')
    ])

    # Use binary_crossentropy for sigmoid output layer
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# --- 3. Define Configurations to Test ---

# List of architectures to test (simulating parameter experimentation)
test_configurations = [
    # Baseline (Good defaults)
    {'lr': 0.001, 'opt': 'Adam', 'act': 'relu', 'bs': 32},

    # High Learning Rate
    {'lr': 0.1, 'opt': 'Adam', 'act': 'relu', 'bs': 32},

    # Different Optimizer (SGD)
    {'lr': 0.01, 'opt': 'SGD', 'act': 'relu', 'bs': 64},

    # Different Activation (Tanh)
    {'lr': 0.001, 'opt': 'Adam', 'act': 'tanh', 'bs': 32},

    # Different Batch Size (Small)
    {'lr': 0.001, 'opt': 'RMSprop', 'act': 'relu', 'bs': 16},
]

results = []

# --- 4. Train and Evaluate Models ---

for i, config in enumerate(test_configurations):
    print(f"\n--- Running Configuration {i + 1} ---")

    # Create and compile the model
    model = create_cnn_model(
        activation_func=config['act'],
        learning_rate=config['lr'],
        optimizer_name=config['opt']
    )

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=3,  # Use few epochs for demonstration speed
        batch_size=config['bs'],
        validation_data=(X_test, y_test),
        verbose=0
    )

    # Evaluate performance
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Record results
    results.append({
        'Config_ID': i + 1,
        'Batch_Size': config['bs'],
        'Optimizer': config['opt'],
        'Activation': config['act'],
        'Learning_Rate': config['lr'],
        'Test_Accuracy': accuracy
    })

    print(
        f"Config {i + 1} (LR: {config['lr']}, Opt: {config['opt']}, BS: {config['bs']}) -> Test Accuracy: {accuracy:.4f}")

# --- 5. Output Final Evaluation ---

results_df = pd.DataFrame(results)

print("\n--- Final Performance Evaluation Summary ---")
print(results_df.sort_values(by='Test_Accuracy', ascending=False).to_string(index=False))