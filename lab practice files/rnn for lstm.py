import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM
from tensorflow.keras.datasets import mnist

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- 1. Load and Prepare Data (MNIST) ---

# Load MNIST data
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

# Use a smaller subset for faster demonstration
X_train_full = X_train_full[:5000]
y_train_full = y_train_full[:5000]
X_test = X_test[:1000]
y_test = y_test[:1000]

# Normalize pixel values
X_train_full = X_train_full.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Split training set for validation
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

# One-hot encode labels for Keras (10 classes)
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_valid_cat = tf.keras.utils.to_categorical(y_valid, num_classes=10)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Input shape for sequence models: (timesteps, features) = (28, 28)
INPUT_SHAPE = (28, 28)
OUTPUT_UNITS = 10

# --- 2. RNN Performance Evaluation ---

rnn_model = Sequential([
    # SimpleRNN processes the 28 time steps, each with 28 features
    SimpleRNN(units=128, activation='relu', input_shape=INPUT_SHAPE),
    Dense(OUTPUT_UNITS, activation='softmax')
])

rnn_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

rnn_history = rnn_model.fit(
    X_train, y_train_cat,
    epochs=5,
    validation_data=(X_valid, y_valid_cat),
    verbose=0
)

rnn_loss, rnn_accuracy = rnn_model.evaluate(X_test, y_test_cat, verbose=0)

# --- 3. LSTM Performance Evaluation ---

lstm_model = Sequential([
    # LSTM layer for handling long-term dependencies
    LSTM(units=128, activation='tanh', input_shape=INPUT_SHAPE),
    Dense(OUTPUT_UNITS, activation='softmax')
])

lstm_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

lstm_history = lstm_model.fit(
    X_train, y_train_cat,
    epochs=5,
    validation_data=(X_valid, y_valid_cat),
    verbose=0
)

lstm_loss, lstm_accuracy = lstm_model.evaluate(X_test, y_test_cat, verbose=0)

# --- 4. Boltzmann Machine (RBM) Performance Verification ---

# RBM performance is verified by using it for feature extraction
# The output is then fed into a standard classifier (Logistic Regression).

# Reshape data for RBM/Logistic Regression: (samples, 784 features)
X_train_flat = X_train.reshape(-1, 28 * 28)
X_test_flat = X_test.reshape(-1, 28 * 28)

# Pre-train RBM for feature extraction
rbm = BernoulliRBM(
    n_components=128, # Number of hidden units (extracted features)
    learning_rate=0.01,
    n_iter=20,
    random_state=42,
    verbose=False
)
rbm.fit(X_train_flat)

# Transform data using RBM features
X_train_rbm = rbm.transform(X_train_flat)
X_test_rbm = rbm.transform(X_test_flat)

# Train a Logistic Regression Classifier on RBM features
rbm_classifier = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=200, random_state=42)
rbm_classifier.fit(X_train_rbm, y_train)

# Predict and verify performance
rbm_pred = rbm_classifier.predict(X_test_rbm)
rbm_accuracy = accuracy_score(y_test, rbm_pred)

# --- 5. Output Final Results ---

print("--- Performance Verification on MNIST Database ---")
print("\n1. Recurrent Neural Network (SimpleRNN):")
print(f"Test Accuracy: {rnn_accuracy:.4f}")

print("\n2. Long Short-Term Memory (LSTM):")
print(f"Test Accuracy: {lstm_accuracy:.4f}")

print("\n3. Boltzmann Machine (RBM Pre-training + Logistic Regression):")
print(f"RBM Features (128 components) Test Accuracy: {rbm_accuracy:.4f}")

# Verification of LSTM vs RNN performance
if lstm_accuracy > rnn_accuracy:
    print("\nObservation: LSTM outperformed SimpleRNN, suggesting better handling of sequential data.")
else:
    print("\nObservation: SimpleRNN performed comparably or better than LSTM on this small subset.")