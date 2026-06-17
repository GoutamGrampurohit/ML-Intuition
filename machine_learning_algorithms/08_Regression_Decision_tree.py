"""
Implementation of a basic Regression Decision Tree.

Features:
- Correct MSE computation
- Sorted data handling
- Better split boundaries
- Batch prediction support
- Input validation
- Recursive tree construction
"""

import numpy as np


class DecisionTree:
    def __init__(
        self,
        depth: int = 5,
        min_leaf_size: int = 5,
    ) -> None:

        self.depth = depth
        self.min_leaf_size = min_leaf_size

        self.decision_boundary = None

        self.left = None
        self.right = None

        self.prediction = None

    @staticmethod
    def mean_squared_error(
        labels: np.ndarray,
        prediction: float,
    ) -> float:
        """
        Compute Mean Squared Error.
        """

        labels = np.asarray(labels)

        if labels.ndim != 1:
            raise ValueError(
                "Labels must be one-dimensional"
            )

        return np.mean(
            (labels - prediction) ** 2
        )

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """
        Train regression decision tree.
        """

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        # Validation
        if x.ndim != 1:
            raise ValueError(
                "Input data set must be one-dimensional"
            )

        if y.ndim != 1:
            raise ValueError(
                "Data set labels must be one-dimensional"
            )

        if len(x) != len(y):
            raise ValueError(
                "x and y must have same length"
            )

        # Sort data
        sorted_indices = np.argsort(x)

        x = x[sorted_indices]
        y = y[sorted_indices]

        # Stopping conditions
        if (
            len(x) < 2 * self.min_leaf_size
            or self.depth <= 1
        ):
            self.prediction = np.mean(y)
            return

        best_split = None

        min_error = float("inf")

        # Try all valid splits
        for i in range(
            self.min_leaf_size,
            len(x) - self.min_leaf_size + 1,
        ):

            left_y = y[:i]
            right_y = y[i:]

            left_prediction = np.mean(left_y)
            right_prediction = np.mean(right_y)

            # CORRECTED MSE
            left_error = self.mean_squared_error(
                left_y,
                left_prediction,
            )

            right_error = self.mean_squared_error(
                right_y,
                right_prediction,
            )

            total_error = (
                left_error * len(left_y)
                + right_error * len(right_y)
            ) / len(y)

            if total_error < min_error:

                min_error = total_error

                best_split = i

        # No valid split found
        if best_split is None:
            self.prediction = np.mean(y)
            return

        # Create split
        left_x = x[:best_split]
        left_y = y[:best_split]

        right_x = x[best_split:]
        right_y = y[best_split:]

        # Better split boundary
        self.decision_boundary = (
            x[best_split - 1]
            + x[best_split]
        ) / 2

        # Create child trees
        self.left = DecisionTree(
            depth=self.depth - 1,
            min_leaf_size=self.min_leaf_size,
        )

        self.right = DecisionTree(
            depth=self.depth - 1,
            min_leaf_size=self.min_leaf_size,
        )

        # Recursive training
        self.left.train(
            left_x,
            left_y,
        )

        self.right.train(
            right_x,
            right_y,
        )

    def predict(
        self,
        x,
    ):
        """
        Predict single value or array of values.
        """

        # Batch prediction
        if isinstance(x, np.ndarray):

            return np.array([
                self.predict(value)
                for value in x
            ])

        # Leaf node
        if self.prediction is not None:
            return self.prediction

        # Traverse tree
        if x < self.decision_boundary:
            return self.left.predict(x)

        return self.right.predict(x)

    def print_tree(
        self,
        level: int = 0,
    ) -> None:
        """
        Print tree structure.
        """

        indent = "  " * level

        if self.prediction is not None:

            print(
                f"{indent}Leaf: "
                f"prediction={self.prediction:.4f}"
            )

            return

        print(
            f"{indent}Node: "
            f"x < {self.decision_boundary:.4f}"
        )

        self.left.print_tree(level + 1)

        self.right.print_tree(level + 1)


def main() -> None:
    """
    Demonstration using sin(x).
    """

    x = np.arange(
        -1.0,
        1.0,
        0.01,
    )

    y = np.sin(x)

    tree = DecisionTree(
        depth=6,
        min_leaf_size=5,
    )

    tree.train(x, y)

    predictions = tree.predict(x)

    mse = np.mean(
        (predictions - y) ** 2
    )

    print("\nRegression Tree Structure:\n")

    tree.print_tree()

    print(f"\nMean Squared Error: {mse:.6f}")

    # Example predictions
    test_values = np.array([
        -0.75,
        -0.25,
        0.0,
        0.5,
        0.9,
    ])

    test_predictions = tree.predict(
        test_values
    )

    print("\nPredictions:\n")

    for value, pred in zip(
        test_values,
        test_predictions,
    ):

        print(
            f"x = {value:.2f} "
            f"-> prediction = {pred:.4f}"
        )


if __name__ == "__main__":

    main()