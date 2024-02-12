import numpy as np
import matplotlib.pyplot as plt

def generate_data(n):
    # Select seed
    np.random.seed(11585932)

    # Obtain X from a uniform distribution
    X = np.random.uniform(0, 1, n)
    # Obtain t from a sinusoid with noise from a Gaussian distribution
    t = np.sin(2*np.pi*X) + np.random.normal(0, 0.3, n)
    
    # Normalize X using z-score normalization
    X_mean = np.mean(X)
    X_std = np.std(X)
    X_normalized = (X - X_mean) / X_std
    
    # Normalize t using z-score normalization
    t_mean = np.mean(t)
    t_std = np.std(t)
    t_normalized = (t - t_mean) / t_std
    
    return X_normalized, t_normalized

def construct_phi(X, M):
    n = len(X)
    # Generate empty matrix for phi
    phi = np.ones((n, M+1))
    # Generates mapping for X_test based on polynomial degree M
    for i in range(n):
        for j in range(1, M+1):
            phi[i, j] = X[i]**(M-j)
    return phi

def calculate_error(phi, t, w):
    n = len(t)
    # Objective function J(w) = ||I(w)-t||^2, <-- L2 Norm!
    return np.sqrt(np.linalg.norm(phi.dot(w) - t, ord=2)**2 / n)

def plot_errors(train_errors, test_errors):
    x_error = np.arange(len(train_errors))
    plt.plot(x_error, train_errors, fillstyle='none', marker='o', c='blue', label='Training')
    plt.plot(x_error, test_errors, fillstyle='none', marker='o', c='red', label='Test')
    plt.xlim(0, len(train_errors)-1)
    plt.ylim(0, max(test_errors) + 0.1)
    plt.title(f'Linear Regression for Nonlinear Models, N_Train = {samples}')
    plt.xlabel("M")
    plt.ylabel("$E_{RMS}$")
    plt.legend()
    plt.show()

# Number of training samples
sample_set = [10, 100]

# Runs the training/testing process for 10 and 100 training samples.
for samples in sample_set:
    # Generates train set based on N_train
    X_train, t_train = generate_data(samples)
    # Generates test set for N_test = 100
    X_test, t_test = generate_data(100)

    # Create empty arrays for errors
    train_errors = []
    test_errors = []

    # Runs through each polynomial and maps the 
    for M in range(10):
        phi_train = construct_phi(X_train, M)
        w = np.linalg.pinv(phi_train).dot(t_train)

        phi_test = construct_phi(X_test, M)

        train_errors.append(calculate_error(phi_train, t_train, w))
        test_errors.append(calculate_error(phi_test, t_test, w))

    plot_errors(train_errors, test_errors)

    # Print the recorded errors
    print(f"Training Errors for N_train = {samples}:")
    print(train_errors)
    print(f"Testing Errors: for N_train = {samples}")
    print(test_errors)