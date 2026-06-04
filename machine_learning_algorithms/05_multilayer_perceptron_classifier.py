"""
Simple Neural Network using sklearn MLPClassifier

This example demonstrates a basic neural network
for binary classification using sklearn.
"""

from sklearn.neural_network import MLPClassifier


def train_model():
    """
    Train MLP classifier on simple dataset
    """

    X = [
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    y = [0, 1, 0, 0]

    clf = MLPClassifier(
        solver="lbfgs",
        alpha=1e-5,
        hidden_layer_sizes=(5, 2),
        random_state=1,
    )

    clf.fit(X, y)

    return clf


def predict(model, test_data):
    """
    Predict output for test data
    """
    return model.predict(test_data)


def wrapper(predictions):
    """
    Convert predictions to list

    >>> model = train_model()
    >>> test = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    >>> preds = predict(model, test)
    >>> [int(x) for x in wrapper(preds)]
    [0, 0, 1]
    """
    return list(predictions)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    model = train_model()

    test = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]

    predictions = predict(model, test)

    print("Predictions:", wrapper(predictions))