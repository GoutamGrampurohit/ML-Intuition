"""
Principal Component Analysis (PCA)

Features:
- Dataset standardization
- Manual PCA implementation using SVD
- Dimensionality reduction
- Explained variance ratio
- Cumulative variance
- Reconstruction capability
- Visualization
"""

import doctest

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


class PCA:
    """
    Principal Component Analysis using Singular Value Decomposition.
    """

    def __init__(
        self,
        n_components: int,
    ) -> None:

        if n_components <= 0:
            raise ValueError(
                "n_components must be greater than 0"
            )

        self.n_components = n_components

        self.components = None

        self.mean = None

        self.explained_variance = None

        self.explained_variance_ratio = None

        self.cumulative_variance_ratio = None

    def fit(
        self,
        X: np.ndarray,
    ) -> None:
        """
        Fit PCA model using SVD.
        """

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                "Input data must be two-dimensional"
            )

        n_samples, n_features = X.shape

        if self.n_components > n_features:
            raise ValueError(
                "n_components cannot exceed number of features"
            )

        # Mean centering
        self.mean = np.mean(X, axis=0)

        X_centered = X - self.mean

        # Singular Value Decomposition
        U, S, Vt = np.linalg.svd(
            X_centered,
            full_matrices=False,
        )

        # Principal components
        self.components = Vt[
            : self.n_components
        ]

        # Explained variance
        explained_variance = (
            S**2
        ) / (n_samples - 1)

        total_variance = np.sum(
            explained_variance
        )

        self.explained_variance = (
            explained_variance[
                : self.n_components
            ]
        )

        self.explained_variance_ratio = (
            self.explained_variance
            / total_variance
        )

        self.cumulative_variance_ratio = (
            np.cumsum(
                self.explained_variance_ratio
            )
        )

    def transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Transform data into lower-dimensional space.
        """
        if self.components is None:
            raise ValueError(
                "PCA model has not been fitted"
            )

        X = np.asarray(X, dtype=float)

        X_centered = X - self.mean

        return np.dot(
            X_centered,
            self.components.T,
        )
    def fit_transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Fit PCA and transform data.
        """

        self.fit(X)

        return self.transform(X)

    def inverse_transform(
        self,
        X_transformed: np.ndarray,
    ) -> np.ndarray:
        """
        Reconstruct original data.
        """

        return np.dot(
            X_transformed,
            self.components,
        ) + self.mean


def collect_dataset() -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Load Iris dataset.
    """

    data = load_iris()

    return (
        np.array(data.data),
        np.array(data.target),
    )


def standardize_data(
    X: np.ndarray,
) -> np.ndarray:
    """
    Standardize dataset.
    """

    scaler = StandardScaler()

    return scaler.fit_transform(X)


def visualize_pca(
    transformed_data: np.ndarray,
    labels: np.ndarray,
) -> None:
    """
    Visualize PCA-transformed data.
    """

    plt.figure(figsize=(8, 6))

    for class_value in np.unique(labels):

        indices = labels == class_value

        plt.scatter(
            transformed_data[indices, 0],
            transformed_data[indices, 1],
            label=f"Class {class_value}",
            alpha=0.7,
        )

    plt.xlabel("Principal Component 1")

    plt.ylabel("Principal Component 2")

    plt.title(
        "PCA Projection of Iris Dataset"
    )

    plt.legend()

    plt.grid(True)

    plt.show()


def main() -> None:
    """
    Driver function.
    """

    # Load dataset
    X, y = collect_dataset()

    # Standardize dataset
    X_scaled = standardize_data(X)

    # Apply PCA
    pca = PCA(n_components=2)

    transformed_data = pca.fit_transform(
        X_scaled
    )

    # Reconstruction
    reconstructed_data = (
        pca.inverse_transform(
            transformed_data
        )
    )

    reconstruction_error = np.mean(
        (X_scaled - reconstructed_data) ** 2
    )

    # Results
    print(
        "\nTransformed Dataset "
        "(First 5 rows):"
    )

    print(transformed_data[:5])

    print(
        "\nExplained Variance Ratio:"
    )

    print(
        pca.explained_variance_ratio
    )

    print(
        "\nCumulative Variance Ratio:"
    )

    print(
        pca.cumulative_variance_ratio
    )

    print(
        f"\nReconstruction Error: "
        f"{reconstruction_error:.6f}"
    )

    # Visualization
    visualize_pca(
        transformed_data,
        y,
    )


if __name__ == "__main__":

    doctest.testmod()

    main()