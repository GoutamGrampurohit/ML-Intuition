"""
k-Nearest Neighbours (kNN) Classification Algorithm
"""

from collections import Counter
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class KNN:
    """
    k-Nearest Neighbours Classifier
    """

    def __init__(
        self,
        train_data: np.ndarray,
        train_target: np.ndarray,
        class_labels: list[str],
    ) -> None:
        """
        Initialize KNN classifier.
        """

        train_data = np.asarray(train_data, dtype=float)
        train_target = np.asarray(train_target)

        if len(train_data) != len(train_target):
            raise ValueError(
                "train_data and train_target must have same length"
            )

        if len(class_labels) == 0:
            raise ValueError(
                "class_labels cannot be empty"
            )

        self.scaler = StandardScaler()

        # Feature scaling
        self.train_data = self.scaler.fit_transform(
            train_data
        )

        self.train_target = train_target
        self.labels = class_labels

    @staticmethod
    def _euclidean_distance(
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
        """
        Compute Euclidean distance.
        """

        return float(np.linalg.norm(a - b))

    def classify(
        self,
        pred_point: np.ndarray,
        k: int = 5,
    ) -> str:
        """
        Classify a single data point.
        """

        if k <= 0:
            raise ValueError(
                "k must be greater than 0"
            )

        if k > len(self.train_data):
            raise ValueError(
                "k cannot exceed number of training samples"
            )

        pred_point = np.asarray(
            pred_point,
            dtype=float,
        )

        # Scale prediction point
        pred_point = self.scaler.transform(
            [pred_point]
        )[0]

        # Vectorized distance calculation
        distances = np.linalg.norm(
            self.train_data - pred_point,
            axis=1,
        )

        # Get k nearest indices
        nearest_indices = np.argsort(distances)[:k]

        # Get neighbour labels
        nearest_labels = self.train_target[
            nearest_indices
        ]

        # Majority voting
        most_common = Counter(
            nearest_labels
        ).most_common(1)[0][0]

        return self.labels[most_common]

    def predict(
        self,
        X_test: np.ndarray,
        k: int = 5,
    ) -> np.ndarray:
        """
        Predict labels for multiple samples.
        """

        X_test = np.asarray(
            X_test,
            dtype=float,
        )

        predictions = [
            self.classify(point, k)
            for point in X_test
        ]

        return np.array(predictions)

    def accuracy(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        k: int = 5,
    ) -> float:
        """
        Compute classification accuracy.
        """

        predictions = self.predict(
            X_test,
            k,
        )

        predicted_indices = np.array([
            np.where(
                self.labels == pred
            )[0][0]
            if isinstance(self.labels, np.ndarray)
            else self.labels.index(pred)
            for pred in predictions
        ])

        return float(
            np.mean(predicted_indices == y_test)
        )


if __name__ == "__main__":

    iris = datasets.load_iris()

    X = np.array(iris["data"])

    y = np.array(iris["target"])

    iris_classes = list(
        iris["target_names"]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    classifier = KNN(
        X_train,
        y_train,
        iris_classes,
    )

    iris_point = np.array(
        [4.4, 3.1, 1.3, 1.4]
    )

    prediction = classifier.classify(
        iris_point,
        k=3,
    )

    print("\nPrediction:")
    print(prediction)

    accuracy = classifier.accuracy(
        X_test,
        y_test,
        k=3,
    )

    print(f"\nAccuracy: {accuracy:.4f}")