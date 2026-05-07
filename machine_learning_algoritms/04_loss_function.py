"""
Loss functions commonly used in Machine Learning and Deep Learning.
"""

import numpy as np


EPSILON = 1e-15


def validate_same_shape(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Validate that two arrays have the same shape.
    """

    if not isinstance(y_true, np.ndarray):
        raise TypeError("y_true must be a NumPy array.")

    if not isinstance(y_pred, np.ndarray):
        raise TypeError("y_pred must be a NumPy array.")

    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays must have the same shape.")


def binary_cross_entropy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = EPSILON,
) -> float:
    """
    Compute Binary Cross Entropy Loss.
    """

    validate_same_shape(y_true, y_pred)

    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -(
        y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    )

    return np.mean(loss)


def binary_focal_cross_entropy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gamma: float = 2.0,
    alpha: float = 0.25,
    epsilon: float = EPSILON,
) -> float:
    """
    Compute Binary Focal Cross Entropy Loss.
    """

    validate_same_shape(y_true, y_pred)

    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -(
        alpha
        * (1 - y_pred) ** gamma
        * y_true
        * np.log(y_pred)
        + (1 - alpha)
        * y_pred**gamma
        * (1 - y_true)
        * np.log(1 - y_pred)
    )

    return np.mean(loss)


def categorical_cross_entropy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = EPSILON,
) -> float:
    """
    Compute Categorical Cross Entropy Loss.
    """

    validate_same_shape(y_true, y_pred)

    if np.any((y_true != 0) & (y_true != 1)):
        raise ValueError("y_true must contain only 0 or 1.")

    if not np.allclose(y_true.sum(axis=1), 1):
        raise ValueError("y_true must be one-hot encoded.")

    if not np.allclose(y_pred.sum(axis=1), 1, atol=epsilon):
        raise ValueError(
            "Predicted probabilities must sum to 1."
        )

    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -np.sum(y_true * np.log(y_pred), axis=1)

    return np.mean(loss)


def categorical_focal_cross_entropy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: np.ndarray = None,
    gamma: float = 2.0,
    epsilon: float = EPSILON,
) -> float:
    """
    Compute Categorical Focal Cross Entropy Loss.
    """

    validate_same_shape(y_true, y_pred)

    num_classes = y_true.shape[1]

    if alpha is None:
        alpha = np.ones(num_classes)

    if len(alpha) != num_classes:
        raise ValueError(
            "Length of alpha must match number of classes."
        )

    if not np.allclose(y_true.sum(axis=1), 1):
        raise ValueError("y_true must be one-hot encoded.")

    if not np.allclose(y_pred.sum(axis=1), 1, atol=epsilon):
        raise ValueError(
            "Predicted probabilities must sum to 1."
        )

    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -np.sum(
        alpha
        * ((1 - y_pred) ** gamma)
        * y_true
        * np.log(y_pred),
        axis=1,
    )

    return np.mean(loss)


def hinge_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute Hinge Loss.
    """

    validate_same_shape(y_true, y_pred)

    if np.any((y_true != -1) & (y_true != 1)):
        raise ValueError(
            "y_true must contain only -1 or 1."
        )

    loss = np.maximum(0, 1 - (y_true * y_pred))

    return np.mean(loss)


def huber_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    delta: float = 1.0,
) -> float:
    """
    Compute Huber Loss.
    """

    validate_same_shape(y_true, y_pred)

    error = y_true - y_pred

    quadratic = 0.5 * error**2

    linear = delta * (
        np.abs(error) - 0.5 * delta
    )

    loss = np.where(
        np.abs(error) <= delta,
        quadratic,
        linear,
    )

    return np.mean(loss)


def mean_squared_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute Mean Squared Error.
    """

    validate_same_shape(y_true, y_pred)

    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute Mean Absolute Error.
    """

    validate_same_shape(y_true, y_pred)

    return np.mean(np.abs(y_true - y_pred))


def mean_squared_logarithmic_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute Mean Squared Logarithmic Error.
    """

    validate_same_shape(y_true, y_pred)

    return np.mean(
        (np.log1p(y_true) - np.log1p(y_pred)) ** 2
    )


def mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = EPSILON,
) -> float:
    """
    Compute Mean Absolute Percentage Error.
    """

    validate_same_shape(y_true, y_pred)

    y_true = np.where(y_true == 0, epsilon, y_true)

    return np.mean(
        np.abs((y_true - y_pred) / y_true)
    )


def smooth_l1_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    beta: float = 1.0,
) -> float:
    """
    Compute Smooth L1 Loss.
    """

    validate_same_shape(y_true, y_pred)

    diff = np.abs(y_true - y_pred)

    loss = np.where(
        diff < beta,
        0.5 * diff**2 / beta,
        diff - 0.5 * beta,
    )

    return np.mean(loss)


def kullback_leibler_divergence(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = EPSILON,
) -> float:
    """
    Compute Kullback-Leibler Divergence.
    """

    validate_same_shape(y_true, y_pred)

    y_true = np.clip(y_true, epsilon, 1)

    y_pred = np.clip(y_pred, epsilon, 1)

    loss = y_true * np.log(y_true / y_pred)

    return np.sum(loss)


def perplexity_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-7,
) -> float:
    """
    Compute Perplexity Loss.
    """

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            "Batch size must be equal."
        )

    if y_true.shape[1] != y_pred.shape[1]:
        raise ValueError(
            "Sentence lengths must be equal."
        )

    vocab_size = y_pred.shape[2]

    if np.max(y_true) >= vocab_size:
        raise ValueError(
            "Label exceeds vocabulary size."
        )

    y_pred = np.clip(y_pred, epsilon, 1)

    batch_indices = np.arange(y_true.shape[0])[:, None]

    sequence_indices = np.arange(y_true.shape[1])

    true_class_probs = y_pred[
        batch_indices,
        sequence_indices,
        y_true,
    ]

    perplexity = np.exp(
        -np.mean(np.log(true_class_probs), axis=1)
    )

    return np.mean(perplexity)


__all__ = [
    "binary_cross_entropy",
    "binary_focal_cross_entropy",
    "categorical_cross_entropy",
    "categorical_focal_cross_entropy",
    "hinge_loss",
    "huber_loss",
    "mean_squared_error",
    "mean_absolute_error",
    "mean_squared_logarithmic_error",
    "mean_absolute_percentage_error",
    "smooth_l1_loss",
    "kullback_leibler_divergence",
    "perplexity_loss",
]


if __name__ == "__main__":

    y_true = np.array([1, 0, 1, 1])

    y_pred = np.array([0.9, 0.2, 0.8, 0.7])

    print(
        "Binary Cross Entropy:",
        binary_cross_entropy(y_true, y_pred),
    )

    print(
        "Mean Squared Error:",
        mean_squared_error(y_true, y_pred),
    )