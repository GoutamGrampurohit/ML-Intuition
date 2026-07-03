from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np


def main():
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    y = np.array([0, 1, 0, 0])

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.25,
        random_state=42,
    )

    clf = MLPClassifier(
        hidden_layer_sizes=(5, 2),
        solver="lbfgs",
        alpha=1e-5,
        max_iter=1000,
        random_state=1,
    )

    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print(f"Accuracy: {accuracy:.4f}")

    test_samples = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )

    test_samples = scaler.transform(
        test_samples
    )

    print(
        "\nPredictions:",
        clf.predict(test_samples),
    )

    print(
        "\nProbabilities:"
    )

    print(
        clf.predict_proba(
            test_samples
        )
    )


if __name__ == "__main__":
    main()