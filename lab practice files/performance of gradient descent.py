import numpy as np
import matplotlib.pyplot as plt

# 1. Define the Objective Function (Cost Function)
# We will use a simple Quadratic function: J(w) = w^2 - 4w + 5
def cost_function(w):
    return w**2 - 4*w + 5

# 2. Define the Gradient (Derivative of the Cost Function)
# J'(w) = 2w - 4
def gradient(w):
    return 2*w - 4

# 3. Implement the Gradient Descent Algorithm
def gradient_descent(start_w, learning_rate, n_iterations):
    w_history = [start_w]
    cost_history = [cost_function(start_w)]
    w = start_w
    
    for i in range(n_iterations):
        # Calculate the gradient (slope) at the current weight 'w'
        grad = gradient(w)
        
        # Update the weight: w = w - learning_rate * gradient
        w = w - learning_rate * grad
        
        # Record the new weight and cost
        w_history.append(w)
        cost_history.append(cost_function(w))
        
        # Check for convergence (optional, for demonstration purposes)
        if abs(grad) < 1e-6:
            print(f"Converged at iteration {i+1}.")
            break
            
    return w, w_history, cost_history

# --- Execution Parameters ---
start_w = 0       # Initial guess for the weight
learning_rate = 0.1 # Step size (hyperparameter)
n_iterations = 20 # Number of steps

# 4. Run Gradient Descent
final_w, w_history, cost_history = gradient_descent(start_w, learning_rate, n_iterations)

# 5. Output Results and Plot Performance

print("--- Gradient Descent Performance Demonstration ---")
print(f"Optimal Weight (Analytic Minimum is w=2): {final_w:.4f}")
print(f"Minimum Cost (Analytic Minimum is J(w)=1): {cost_function(final_w):.4f}")
print(f"Total Iterations Run: {len(w_history) - 1}")

# Plotting the Cost Function and the Descent Path
w_range = np.linspace(-1, 5, 100)
cost_range = cost_function(w_range)

plt.figure(figsize=(10, 5))

# Plot 1: Cost Function vs. Weight (w) and the Path
plt.subplot(1, 2, 1)
plt.plot(w_range, cost_range, label='Cost Function $J(w)$', color='blue')
plt.plot(w_history, cost_history, 'ro--', label='Descent Path') # 'ro--' for red circles, dashed line
plt.scatter([final_w], [cost_function(final_w)], color='green', marker='o', s=100, label='Final Minimum')
plt.title('Gradient Descent Path')
plt.xlabel('Weight (w)')
plt.ylabel('Cost $J(w)$')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Plot 2: Cost vs. Iteration (Convergence Curve)
plt.subplot(1, 2, 2)
plt.plot(range(len(cost_history)), cost_history, 'm-')
plt.title('Cost vs. Iteration (Convergence)')
plt.xlabel('Iteration')
plt.ylabel('Cost $J(w)$')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
