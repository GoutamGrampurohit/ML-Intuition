import math
import random


LEARNING_RATE = 0.1
INPUT_VALUE = 0.02


def sigmoid_function(
    value: float,
    deriv: bool = False
) -> float:

    sigmoid = 1 / (1 + math.exp(-value))

    if deriv:
        return sigmoid * (1 - sigmoid)

    return sigmoid


def forward_propagation(
    expected: int,
    number_propagations: int
) -> float:

    random.seed(42)

    weight = random.uniform(-1, 1)

    target = expected / 100

    for _ in range(number_propagations):

        # Forward pass
        weighted_sum = INPUT_VALUE * weight

        output = sigmoid_function(
            weighted_sum
        )

        # Error
        error = target - output

        # Gradient
        gradient = (
            error
            * sigmoid_function(
                weighted_sum,
                deriv=True
            )
        )

        # Weight update
        weight += (
            LEARNING_RATE
            * gradient
            * INPUT_VALUE
        )

    return output * 100


if __name__ == "__main__":

    expected = int(
        input("Expected value: ")
    )

    number_propagations = int(
        input(
            "Number of propagations: "
        )
    )

    result = forward_propagation(
        expected,
        number_propagations,
    )

    print(
        f"\nPredicted value: "
        f"{result:.4f}"
    )