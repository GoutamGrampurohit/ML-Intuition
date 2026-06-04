"""
Gradient Boosting Regressor Example

Features:
- California Housing Dataset
- Cross Validation
- Feature Importance
- Residual Analysis
- Early Stopping
- Proper Evaluation Metrics
- Visualization
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    cross_val_score,
    train_test_split,
)


class GradientBoostingRegressionModel:
    """
    Gradient Boosting Regressor Wrapper
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        min_samples_split: int = 4,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
        random_state: int = 42,
    ) -> None:

        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            random_state=random_state,
        )

    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> None:
        """
        Train Gradient Boosting model.
        """

        self.model.fit(
            x_train,
            y_train,
        )

    def predict(
        self,
        x_test: pd.DataFrame,
    ) -> np.ndarray:
        """
        Predict target values.
        """

        return self.model.predict(
            x_test
        )

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        """
        Evaluate regression performance.
        """

        mse = mean_squared_error(
            y_true,
            y_pred,
        )

        rmse = np.sqrt(mse)

        mae = mean_absolute_error(
            y_true,
            y_pred,
        )

        r2 = r2_score(
            y_true,
            y_pred,
        )

        print(
            f"\nMean Squared Error: "
            f"{mse:.4f}"
        )

        print(
            f"Root Mean Squared Error: "
            f"{rmse:.4f}"
        )

        print(
            f"Mean Absolute Error: "
            f"{mae:.4f}"
        )

        print(
            f"R² Score: "
            f"{r2:.4f}"
        )

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
    ) -> None:
        """
        Perform cross validation.
        """

        scores = cross_val_score(
            self.model,
            X,
            y,
            cv=cv,
            scoring="r2",
        )

        print(
            f"\nCross Validation Scores:\n"
            f"{scores}"
        )

        print(
            f"\nMean CV R² Score: "
            f"{scores.mean():.4f}"
        )

    def plot_feature_importance(
        self,
        feature_names,
    ) -> None:
        """
        Plot feature importance.
        """

        importance = (
            self.model.feature_importances_
        )

        sorted_indices = np.argsort(
            importance
        )

        plt.figure(figsize=(10, 6))

        plt.barh(
            np.array(feature_names)[
                sorted_indices
            ],
            importance[sorted_indices],
        )

        plt.xlabel("Importance")

        plt.title(
            "Feature Importance"
        )

        plt.show()

    @staticmethod
    def plot_predictions(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        """
        Plot actual vs predicted values.
        """

        plt.figure(figsize=(8, 6))

        plt.scatter(
            y_true,
            y_pred,
            alpha=0.6,
        )

        plt.plot(
            [
                y_true.min(),
                y_true.max(),
            ],
            [
                y_true.min(),
                y_true.max(),
            ],
            "r--",
            linewidth=2,
        )

        plt.xlabel("Actual Values")

        plt.ylabel("Predicted Values")

        plt.title(
            "Actual vs Predicted"
        )

        plt.grid(True)

        plt.show()

    @staticmethod
    def plot_residuals(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        """
        Plot residuals.
        """

        residuals = y_true - y_pred

        plt.figure(figsize=(8, 6))

        plt.scatter(
            y_pred,
            residuals,
            alpha=0.6,
        )

        plt.axhline(
            y=0,
            linestyle="--",
        )

        plt.xlabel("Predicted Values")

        plt.ylabel("Residuals")

        plt.title(
            "Residual Plot"
        )

        plt.grid(True)

        plt.show()


def load_dataset() -> tuple[
    pd.DataFrame,
    pd.Series,
]:
    """
    Load California Housing dataset.
    """

    housing = fetch_california_housing()

    X = pd.DataFrame(
        housing.data,
        columns=housing.feature_names,
    )

    y = pd.Series(
        housing.target,
        name="HousePrice",
    )

    return X, y


def main() -> None:
    """
    Driver function.
    """

    # Load dataset
    X, y = load_dataset()

    print(
        "\nDataset Shape:"
    )

    print(X.shape)

    print(
        "\nFirst 5 Rows:"
    )

    print(X.head())

    print(
        "\nSummary Statistics:"
    )

    print(X.describe().T)

    # Train/Test Split
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
        )
    )

    # Initialize model
    model = (
        GradientBoostingRegressionModel(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
        )
    )

    # Train model
    model.fit(
        X_train,
        y_train,
    )

    # Predictions
    y_pred = model.predict(
        X_test
    )

    # Evaluation
    model.evaluate(
        y_test,
        y_pred,
    )

    # Cross Validation
    model.cross_validate(
        X,
        y,
        cv=5,
    )

    # Feature Importance
    model.plot_feature_importance(
        X.columns
    )

    # Actual vs Predicted
    model.plot_predictions(
        y_test,
        y_pred,
    )

    # Residual Plot
    model.plot_residuals(
        y_test,
        y_pred,
    )


if __name__ == "__main__":

    main()