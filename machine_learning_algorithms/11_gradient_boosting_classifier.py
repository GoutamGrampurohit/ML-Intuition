"""
Gradient Boosting Binary Classifier


- Probability prediction
- Loss tracking
- Accuracy evaluation
- Input validation
- Binary classification support
"""

import numpy as np

from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    log_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from matplotlib import pyplot as plt


class GradientBoostingClassifier:
    """
    Binary Gradient Boosting Classifier
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 1,
    ) -> None:

        if n_estimators <= 0:
            raise ValueError(
                "n_estimators must be positive"
            )

        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive"
            )

        self.n_estimators = n_estimators

        self.learning_rate = learning_rate

        self.max_depth = max_depth

        self.models = []

        self.loss_history = []

        self.initial_prediction = 0.0

    @staticmethod
    def sigmoid(
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Sigmoid activation function.
        """

        return 1 / (
            1 + np.exp(-np.clip(x, -500, 500))
        )

    def fit(
        self,
        features: np.ndarray,
        target: np.ndarray,
    ) -> None:
        """
        Train Gradient Boosting Classifier.

        NOTE:
        Target must contain binary labels:
        {0,1}
        """

        features = np.asarray(
            features,
            dtype=float,
        )

        target = np.asarray(target)

        unique_classes = np.unique(target)

        if len(unique_classes) != 2:
            raise ValueError(
                "This implementation supports "
                "binary classification only."
            )

        # Initial prediction (log odds)
        positive_ratio = np.clip(
            np.mean(target),
            1e-10,
            1 - 1e-10,
        )

        self.initial_prediction = np.log(
            positive_ratio
            / (1 - positive_ratio)
        )

        predictions = np.full(
            target.shape,
            self.initial_prediction,
            dtype=float,
        )

        # Gradient boosting iterations
        for _ in range(self.n_estimators):

            probabilities = self.sigmoid(
                predictions
            )

            # Negative gradient
            residuals = target - probabilities

            # Weak learner
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
            )

            tree.fit(
                features,
                residuals,
            )

            update = tree.predict(features)

            predictions += (
                self.learning_rate * update
            )

            self.models.append(tree)

            # Track loss
            loss = log_loss(
                target,
                self.sigmoid(predictions),
            )

            self.loss_history.append(loss)

    def predict_proba(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        """
        Predict probabilities.
        """

        features = np.asarray(
            features,
            dtype=float,
        )

        predictions = np.full(
            features.shape[0],
            self.initial_prediction,
            dtype=float,
        )

        for tree in self.models:

            predictions += (
                self.learning_rate
                * tree.predict(features)
            )

        probabilities = self.sigmoid(
            predictions
        )

        return np.column_stack(
            [
                1 - probabilities,
                probabilities,
            ]
        )

    def predict(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        """
        Predict class labels.
        """

        probabilities = self.predict_proba(
            features
        )[:, 1]

        return (
            probabilities >= 0.5
        ).astype(int)

    def plot_loss(self) -> None:
        """
        Plot training loss curve.
        """

        plt.figure(figsize=(8, 5))

        plt.plot(self.loss_history)

        plt.xlabel("Iterations")

        plt.ylabel("Log Loss")

        plt.title(
            "Gradient Boosting Training Loss"
        )

        plt.grid(True)

        plt.show()


def main() -> None:
    """
    Demonstration using binary Iris dataset.
    """

    iris = load_iris()

    X = iris.data

    y = iris.target

    # Convert to binary classification
    # class 0 vs non-class 0
    y_binary = (
        y == 0
    ).astype(int)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y_binary,
            test_size=0.2,
            random_state=42,
            stratify=y_binary,
        )
    )

    classifier = (
        GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=1,
        )
    )

    classifier.fit(
        X_train,
        y_train,
    )

    predictions = classifier.predict(
        X_test
    )

    probabilities = (
        classifier.predict_proba(X_test)
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print(
        f"\nAccuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        "\nClassification Report:\n"
    )

    print(
        classification_report(
            y_test,
            predictions,
        )
    )

    # Confusion Matrix
    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "Not Class 0",
            "Class 0",
        ],
    )

    display.plot(
        cmap="Blues",
    )

    plt.title(
        "Gradient Boosting Confusion Matrix"
    )

    plt.show()

    # Loss curve
    classifier.plot_loss()

    # Example probability predictions
    print(
        "\nFirst 5 Prediction Probabilities:\n"
    )

    print(
        probabilities[:5]
    )


if __name__ == "__main__":

    main()