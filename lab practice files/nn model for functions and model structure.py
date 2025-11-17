import numpy as np
import pandas as pd
import tensorflow as tf  # <-- FIX: This was missing and caused the NameError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_moons
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- 1. Prepare Data (Two Moons - Non-linear) ---
X, Y = make_moons(n_samples=300, noise=0.15, random_state=42)
N_features = X.shape[1]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42, stratify=Y
)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- 2. Define Model Construction and Evaluation Function ---

def build_and_evaluate_mlp(activation_func, hidden_layer_config):
    """Constructs, trains, and evaluates an MLP with specified architecture and activation."""

    # 3. Construct the Model Structure
    model = Sequential()

    # Input and Hidden Layers
    model.add(Dense(hidden_layer_config[0], activation=activation_func, input_shape=(N_features,)))
    for num_neurons in hidden_layer_config[1:]:
        model.add(Dense(num_neurons, activation=activation_func))

    # Output Layer (Binary Classification)
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    model.fit(
        X_train_scaled, Y_train,
        epochs=30,
        batch_size=8,
        verbose=0
    )

    # Evaluate performance
    loss, accuracy = model.evaluate(X_test_scaled, Y_test, verbose=0)

    return accuracy


# --- 4. Define Configurations to Test ---

# Activation Functions to test
activations = ['relu', 'tanh', 'sigmoid']

# Model Structures to test (Hidden Layers)
structures = {
    'Shallow (1x10)': (10,),
    'Medium (2x8)': (8, 8),
    'Deep (3x16)': (16, 16, 16)
}

results = []

# --- 5. Run Verification Loop ---

print("--- Starting NN Performance Verification ---")

for act_name in activations:
    for struct_name, struct_config in structures.items():
        accuracy = build_and_evaluate_mlp(act_name, struct_config)

        results.append({
            'Activation': act_name,
            'Structure': struct_name,
            'Layers': len(struct_config),
            'Test_Accuracy': accuracy
        })

# --- 6. Output Final Evaluation ---

results_df = pd.DataFrame(results)
results_df_sorted = results_df.sort_values(by='Test_Accuracy', ascending=False)

print("\n--- Final Performance Evaluation Summary ---")
print(results_df_sorted.to_string(index=False))

print("\n--- Best Performing Configuration ---")
best_config = results_df_sorted.iloc[0]
print(f"Accuracy: {best_config['Test_Accuracy']:.4f}")
print(f"Details: Activation='{best_config['Activation']}', Structure='{best_config['Structure']}'")