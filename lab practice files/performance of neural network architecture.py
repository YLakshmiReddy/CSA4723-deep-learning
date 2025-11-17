import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

# --- 1. Load and Prepare Data (Using Iris dataset) ---
data = load_iris(as_frame=True)
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define preprocessing (StandardScaler is crucial for MLPs)
preprocessor = StandardScaler()


# --- 2. Define the Evaluation Function ---
def evaluate_mlp_architecture(learning_rate_init, activation_func, hidden_layer_config):
    """
    Constructs an MLPClassifier with specified parameters, trains it, and returns
    the test accuracy.
    """

    # 3. Construct the MLP Model
    # The hidden_layer_config determines the number of layers and neurons.
    # e.g., (100,) means 1 layer of 100 neurons.
    # e.g., (10, 10) means 2 layers of 10 neurons each.
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_config,
        activation=activation_func,
        solver='adam',
        learning_rate_init=learning_rate_init,
        max_iter=500,
        random_state=42,
        n_iter_no_change=20,
        verbose=False  # Set to True to see convergence messages
    )

    # Create a full pipeline with scaling and the classifier
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', mlp)])

    # 4. Train the Model
    try:
        model_pipeline.fit(X_train, y_train)
    except Exception as e:
        # Catch potential errors like non-convergence for high learning rates
        return 0.0, str(e)

    # 5. Verify Performance
    y_pred = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Return performance metric and the configuration string
    config_str = (
        f"Layers: {len(hidden_layer_config)}, Neurons: {hidden_layer_config}, "
        f"LR: {learning_rate_init}, Act: '{activation_func}'"
    )
    return accuracy, config_str


# --- 6. Define Configurations to Test ---

# List of architectures to test (simulating the Playground experimentation)
test_configurations = [
    # Baseline
    (0.001, 'relu', (10,)),

    # High Learning Rate (Testing convergence/overshooting)
    (0.1, 'relu', (10,)),

    # Low Learning Rate (Testing slow convergence)
    (0.0001, 'relu', (10,)),

    # Shallow Network (Testing underfitting/simpler model)
    (0.001, 'relu', (5,)),

    # Deep Network (Testing complexity)
    (0.001, 'relu', (20, 20, 20)),

    # Different Activation Function (Tanh)
    (0.001, 'tanh', (10,)),

    # Different Activation Function (Logistic/Sigmoid)
    (0.001, 'logistic', (10,)),
]

results = []
for lr, act, hlc in test_configurations:
    accuracy, config_str = evaluate_mlp_architecture(lr, act, hlc)
    results.append({'Config': config_str, 'Accuracy': accuracy})

# --- 7. Output Performance Evaluation ---

results_df = pd.DataFrame(results)
results_df_sorted = results_df.sort_values(by='Accuracy', ascending=False)

print("--- Performance Evaluation of Neural Network Architectures ---")
print(results_df_sorted.to_string(index=False))

# Optional: Print best performing configuration
best_config = results_df_sorted.iloc[0]
print("\n--- Best Performing Configuration ---")
print(f"Accuracy: {best_config['Accuracy']:.4f}")
print(f"Details: {best_config['Config']}")