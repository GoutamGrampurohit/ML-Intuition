"""
Gaussian Naive Bayes Classification Example

Features:
- Standardization
- Train/Test Split
- Cross Validation
- Confusion Matrix
- Classification Report
- Probability Prediction
- Accuracy Evaluation
"""

from matplotlib import pyplot as plt
import numpy as np

from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import (
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


class GaussianNaiveBayesClassifier:
    """
    Gaussian Naive Bayes Classifier Wrapper
    """

    def __init__(self) -> None:

        self.scaler = StandardScaler()

        self.model = GaussianNB()

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        """
        Train Gaussian Naive Bayes model.
        """

        x_train_scaled = self.scaler.fit_transform(
            x_train
        )

        self.model.fit(
            x_train_scaled,
            y_train,
        )

    def predict(
        self,
        x_test: np.ndarray,
    ) -> np.ndarray:
        """
        Predict class labels.
        """

        x_test_scaled = self.scaler.transform(
            x_test
        )

        return self.model.predict(
            x_test_scaled
        )

    def predict_probabilities(
        self,
        x_test: np.ndarray,
    ) -> np.ndarray:
        """
        Predict class probabilities.
        """

        x_test_scaled = self.scaler.transform(
            x_test
        )

        return self.model.predict_proba(
            x_test_scaled
        )

    def evaluate(
        self,
        x_test: np.ndarray,
        y_test: np.ndarray,
        target_names: list[str],
    ) -> None:
        """
        Evaluate classifier performance.
        """

        predictions = self.predict(x_test)

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        print(
            f"\nModel Accuracy: "
            f"{accuracy * 100:.2f}%"
        )

        print(
            "\nClassification Report:\n"
        )

        print(
            classification_report(
                y_test,
                predictions,
                target_names=target_names,
            )
        )

        # Confusion Matrix
        matrix = confusion_matrix(
            y_test,
            predictions,
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=target_names,
        )

        display.plot(
            cmap="Blues",
        )

        plt.title(
            "Confusion Matrix - Iris Dataset"
        )

        plt.show()

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
    ) -> None:
        """
        Perform cross validation.
        """

        X_scaled = self.scaler.fit_transform(X)

        scores = cross_val_score(
            self.model,
            X_scaled,
            y,
            cv=cv,
        )

        print(
            f"\nCross Validation Scores: "
            f"{scores}"
        )

        print(
            f"Mean CV Accuracy: "
            f"{scores.mean() * 100:.2f}%"
        )


def main() -> None:
    """
    Driver function.
    """

    # Load dataset
    iris = load_iris()

    X = np.array(
        iris["data"]
    )

    y = np.array(
        iris["target"]
    )

    target_names = list(
        iris["target_names"]
    )

    # Train/Test Split
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42,
            stratify=y,
        )
    )

    # Initialize classifier
    classifier = (
        GaussianNaiveBayesClassifier()
    )

    # Train model
    classifier.fit(
        X_train,
        y_train,
    )

    # Evaluate model
    classifier.evaluate(
        X_test,
        y_test,
        target_names,
    )

    # Cross Validation
    classifier.cross_validate(
        X,
        y,
        cv=5,
    )

    # Example prediction
    sample = np.array([
        [5.1, 3.5, 1.4, 0.2]
    ])

    prediction = classifier.predict(
        sample
    )[0]

    probabilities = (
        classifier.predict_probabilities(
            sample
        )[0]
    )

    print("\nSample Prediction:")

    print(
        f"Predicted Class: "
        f"{target_names[prediction]}"
    )

    print(
        "\nClass Probabilities:"
    )

    for class_name, prob in zip(
        target_names,
        probabilities,
    ):

        print(
            f"{class_name}: "
            f"{prob:.4f}"
        )


if __name__ == "__main__":

    main()