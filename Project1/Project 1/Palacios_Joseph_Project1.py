import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load in the "carbig" dataset and clean any rows.
df = pd.read_excel('proj1Dataset.xlsx')
df = df.dropna()

# Select 'Weight' as predictor and 'Horsepower' as target variable
predictor = np.array(df['Weight'])
target = np.array(df['Horsepower'])

# Add a column of ones to X for the bias term
X = np.c_[predictor, np.ones_like(predictor)]

# Closed-form solution
w = np.linalg.pinv(X) @ target

# Normalize the predictor using z-score normalization
predictor_mean = np.mean(predictor)
predictor_std = np.std(predictor)
predictor_norm = (predictor - predictor_mean) / predictor_std

# Initialize parameters
w0 = 0  # intercept
w1 = 0  # slope

# Set learning rate and iterations
rho = 0.1
iterations = 1000

# Grab the number of samples
N = len(predictor)

# Set a threshold for termination criterion
threshold = 1e-6

# Perform gradient descent
for _ in range(iterations):

    # Calculate predictions
    predictions = w0 + w1 * predictor_norm

    # Calculate errors
    errors = predictions - target

    # Update parameters using the transpose of the gradient
    gradient0 = np.sum(errors)
    gradient1 = np.sum(errors * predictor_norm)

    # Update parameters
    w0_update = w0 - rho * gradient0 / N
    w1_update = w1 - rho * gradient1 / N

    # Check whether the change in parameters w0 and w1 are below the threshold.
    # If below the threshold, the loop breaks, indicating convergence.
    # Also prints the number of iterations it took to converge.

    if np.abs(w0_update - w0) < threshold and np.abs(w1_update - w1) < threshold:
        print(f"Converged after {_+1} iterations.")
        break

    # Update parameters for the next iteration
    w0 = w0_update
    w1 = w1_update

    # Could additionally include a clause where we diverge
    # If after all the iterations, we still haven't converged then we diverged...
    # Any potential problems with this?

# Plot linear regression using closed-form solution
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.scatter(predictor, target, marker='x', color='red')
plt.plot(predictor, X @ w, label='Closed Form', color='blue')
plt.xlabel('Weight')
plt.ylabel('Horsepower')
plt.legend()

# Plot linear regression using gradient descent
plt.subplot(1,2,2)
plt.scatter(predictor, target, marker='x', color='red')
plt.plot(predictor, w0 + w1 * predictor_norm, label='Gradient Descent', color='lime')
plt.xlabel('Weight')
plt.ylabel('Horsepower')
plt.legend()

plt.tight_layout()
plt.show()