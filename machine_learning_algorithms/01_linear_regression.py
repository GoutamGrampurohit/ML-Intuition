import httpx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def collect_dataset():
    """Download and prepare dataset"""
    response = httpx.get(
        "https://raw.githubusercontent.com/yashLadha/The_Math_of_Intelligence/master/Week1/ADRvsRating.csv",
        timeout=10,
    )

    lines = response.text.splitlines()
    data = []

    for item in lines[1:]:  # skip header
        data.append(item.split(","))

    dataset = np.array(data, dtype=float)
    return dataset


def feature_scaling(data_x):
    """Normalize features (excluding bias column)"""
    data_x[:, 1:] = (data_x[:, 1:] - np.mean(data_x[:, 1:])) / np.std(data_x[:, 1:])
    return data_x


def compute_error(data_x, data_y, theta):
    """Mean Squared Error"""
    n = len(data_y)
    predictions = np.dot(theta, data_x.T)
    error = np.sum((predictions - data_y) ** 2) / (2 * n)
    return error


def gradient_descent(data_x, data_y, theta, alpha, iterations):
    """Train model using gradient descent"""
    n = len(data_y)
    errors = []

    for i in range(iterations):
        predictions = np.dot(theta, data_x.T)
        gradient = np.dot((predictions - data_y), data_x)
        theta = theta - (alpha / n) * gradient

        error = compute_error(data_x, data_y, theta)
        errors.append(error)

        if i % 1000 == 0:
            print(f"Iteration {i} - Error {error:.5f}")

    return theta, errors


def predict(data_x, theta):
    """Make predictions"""
    return np.dot(theta, data_x.T)


def plot_regression(data_x, data_y, theta):
    """Plot regression line"""
    plt.scatter(data_x[:, 1], data_y, label="Data")
    predictions = predict(data_x, theta)
    plt.plot(data_x[:, 1], predictions.flatten(), label="Model", color="red")
    plt.xlabel("ADR")
    plt.ylabel("Rating")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.show()


def sklearn_comparison(data_x, data_y):
    """Compare with sklearn"""
    model = LinearRegression()
    model.fit(data_x[:, 1:].reshape(-1, 1), data_y)

    print("\n--- Sklearn Comparison ---")
    print("Intercept:", model.intercept_)
    print("Coefficient:", model.coef_)


def main():
    # Load data
    data = collect_dataset()

    # Prepare features and labels
    data_x = data[:, :-1]
    data_y = data[:, -1]

    # Add bias column
    data_x = np.c_[np.ones(len(data_x)), data_x]

    # Normalize features
    data_x = feature_scaling(data_x)

    # Initialize parameters
    theta = np.zeros((1, data_x.shape[1]))

    # Train model
    theta, errors = gradient_descent(
        data_x, data_y, theta, alpha=0.001, iterations=10000
    )

    print("\nFinal Parameters (Theta):")
    print(theta)

    # Save model
    np.save("theta.npy", theta)

    # Plot result
    plot_regression(data_x, data_y, theta)

    # Compare with sklearn
    sklearn_comparison(data_x, data_y)


if __name__ == "__main__":
    main()