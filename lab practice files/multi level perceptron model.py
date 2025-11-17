import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

# --- 1. Load and Prepare Data (Using Iris dataset for simplicity) ---
data = load_iris(as_frame=True)
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define preprocessing (scaling is crucial for MLPs)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X.columns)
    ])


# --- 2. Define MLP Model Construction and Verification Function ---
def build_and_verify_mlp(learning_rate_init, activation_func, hidden_layer_config):
    """
    Constructs an MLPClassifier with specified parameters and verifies its performance.
    """
    # 3. Construct the MLP Model
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_config,  # E.g., (100,) or (10, 10)
        activation=activation_func,  # E.g., 'relu', 'logistic', 'tanh'
        solver='adam',  # Optimization algorithm
        learning_rate_init=learning_rate_init,  # Learning rate (alpha)
        max_iter=500,  # Number of training epochs
        random_state=42,
        n_iter_no_change=20  # For early stopping
    )

    # Create a full pipeline with preprocessing and the model
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', mlp)])

    # 4. Train the Model
    model_pipeline.fit(X_train, y_train)

    # 5. Verify Performance
    y_pred = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\n--- MLP Configuration ---")
    print(f"Learning Rate: {learning_rate_init}, Activation: '{activation_func}', Hidden Layers: {hidden_layer_config}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", report)
    print("--------------------------------------------------")

    return model_pipeline, accuracy


# --- 6. Demonstrate Verification with Different Configurations (as requested) ---

print("--- Multi-Layer Perceptron Verification ---")

# A. Configuration 1: Baseline (Default/Good Parameters)
build_and_verify_mlp(
    learning_rate_init=0.001,
    activation_func='relu',
    hidden_layer_config=(10, 10)  # Two hidden layers, 10 neurons each
)

# B. Configuration 2: High Learning Rate (May oscillate/fail to converge)
build_and_verify_mlp(
    learning_rate_init=0.1,
    activation_func='relu',
    hidden_layer_config=(10, 10)
)

# C. Configuration 3: Different Activation Function (Tanh)
build_and_verify_mlp(
    learning_rate_init=0.001,
    activation_func='tanh',
    hidden_layer_config=(50,)  # One hidden layer, 50 neurons
)