"""
Polynomial Regression using Ordinary Least Squares (OLS)
with numerical stability improvements and feature scaling.
"""

import matplotlib.pyplot as plt
import numpy as np


class PolynomialRegression:
    __slots__ = (
        "degree",
        "params",
        "mean",
        "std",
        "regularization",
    )

    def __init__(
        self,
        degree: int,
        regularization: float = 1e-8,
    ) -> None:
        """
        Initialize Polynomial Regression model.

        Args:
            degree: Polynomial degree
            regularization: Ridge regularization strength

        Raises:
            ValueError: If degree is negative
        """

        if degree < 0:
            raise ValueError(
                "Polynomial degree must be non-negative"
            )

        self.degree = degree
        self.params = None
        self.mean = None
        self.std = None
        self.regularization = regularization

    @staticmethod
    def _design_matrix(
        data: np.ndarray,
        degree: int,
    ) -> np.ndarray:
        """
        Construct Vandermonde design matrix.
        """
        data = np.asarray(data)
        if data.ndim != 1:
            raise ValueError(
                "Data must have dimensions N x 1"
            )

        return np.vander(
            data,
            N=degree + 1,
            increasing=True,
        )

    def _normalize(
        self,
        data: np.ndarray,
    ) -> np.ndarray:
        """
        Normalize input data using training statistics.
        """

        return (data - self.mean) / self.std

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        """
        Fit polynomial regression model using
        Ridge-Regularized Least Squares.
        """

        x_train = np.asarray(x_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)

        if x_train.ndim != 1:
            raise ValueError(
                "x_train must be one-dimensional"
            )

        if y_train.ndim != 1:
            raise ValueError(
                "y_train must be one-dimensional"
            )

        if len(x_train) != len(y_train):
            raise ValueError(
                "x_train and y_train must have same length"
            )

        # Feature scaling
        self.mean = np.mean(x_train)
        self.std = np.std(x_train)

        if self.std == 0:
            raise ValueError(
                "Standard deviation cannot be zero"
            )

        x_train = self._normalize(x_train)

        # Design matrix
        X = self._design_matrix(
            x_train,
            self.degree,
        )

        # Ridge regularization
        identity = np.eye(X.shape[1])

        self.params = np.linalg.inv(
            X.T @ X
            + self.regularization * identity
        ) @ X.T @ y_train

    def predict(
        self,
        data: np.ndarray,
    ) -> np.ndarray:
        """
        Predict output values for input data.
        """

        if self.params is None:
            raise ArithmeticError(
                "Predictor hasn't been fit yet"
            )

        data = np.asarray(data, dtype=float)

        if data.ndim != 1:
            raise ValueError(
                "Prediction data must be one-dimensional"
            )

        data = self._normalize(data)

        X = self._design_matrix(
            data,
            self.degree,
        )

        return X @ self.params

    def mean_squared_error(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """
        Compute Mean Squared Error.
        """

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if y_true.shape != y_pred.shape:
            raise ValueError(
                "Shapes of y_true and y_pred must match"
            )

        return np.mean((y_true - y_pred) ** 2)

    def r2_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """
        Compute R² Score.
        """

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        ss_total = np.sum(
            (y_true - np.mean(y_true)) ** 2
        )

        ss_residual = np.sum(
            (y_true - y_pred) ** 2
        )

        return 1 - (ss_residual / ss_total)

    def fit_predict(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
    ) -> np.ndarray:
        """
        Fit model and return predictions.
        """

        self.fit(x_train, y_train)

        return self.predict(x_train)


def main() -> None:
    """
    Demonstration of Polynomial Regression.
    """

    import seaborn as sns

    mpg_data = sns.load_dataset("mpg").dropna()

    x = mpg_data.weight.values
    y = mpg_data.mpg.values

    poly_reg = PolynomialRegression(
        degree=2,
        regularization=1e-6,
    )

    poly_reg.fit(x, y)

    predictions = poly_reg.predict(x)

    mse = poly_reg.mean_squared_error(
        y,
        predictions,
    )

    r2 = poly_reg.r2_score(
        y,
        predictions,
    )

    print("\nModel Parameters:")
    print(poly_reg.params)

    print(f"\nMSE : {mse:.4f}")
    print(f"R²  : {r2:.4f}")

    # Visualization
    sorted_indices = np.argsort(x)

    x_sorted = x[sorted_indices]
    y_sorted_pred = predictions[sorted_indices]

    plt.figure(figsize=(10, 6))

    plt.scatter(
        x,
        y,
        alpha=0.5,
        label="Actual Data",
    )

    plt.plot(
        x_sorted,
        y_sorted_pred,
        linewidth=3,
        label="Polynomial Fit",
    )

    plt.title(
        "Polynomial Regression"
    )

    plt.xlabel("Weight (lbs)")
    plt.ylabel("MPG")

    plt.legend()

    plt.show()


if __name__ == "__main__":

    main()