"""
Implementation of gradient descent algorithm for minimizing cost of a linear hypothesis
function.
"""

import numpy as np

# List of input, output pairs
train_data = (
    ((5, 2, 3), 15),
    ((6, 5, 9), 25),
    ((11, 12, 13), 41),
    ((1, 1, 1), 8),
)

test_data = (
    ((515, 22, 13), 555),
    ((61, 35, 49), 150),
)

# Bias + 3 feature weights
parameter_vector = np.array([2.0, 4.0, 1.0, 5.0])

m = len(train_data)

LEARNING_RATE = 0.0001
MAX_ITERATIONS = 100000
TOLERANCE = 1e-6


def _error(example_no, data_set="train"):
    """
    :param data_set: train data or test data
    :param example_no: example number whose error has to be checked
    :return: error in example pointed by example number.
    """
    return calculate_hypothesis_value(example_no, data_set) - output(
        example_no, data_set
    )


def _hypothesis_value(data_input_tuple):
    """
    Calculates hypothesis function value for a given input
    :param data_input_tuple: Input tuple of a particular example
    :return: Value of hypothesis function at that point.
    """

    hyp_val = parameter_vector[0]  # bias term

    for i in range(len(data_input_tuple)):
        hyp_val += data_input_tuple[i] * parameter_vector[i + 1]

    return hyp_val


def output(example_no, data_set):
    """
    :param data_set: test data or train data
    :param example_no: example whose output is to be fetched
    :return: output for that example
    """

    if data_set == "train":
        return train_data[example_no][1]

    elif data_set == "test":
        return test_data[example_no][1]

    return None


def calculate_hypothesis_value(example_no, data_set):
    """
    Calculates hypothesis value for a given example
    :param data_set: test data or train_data
    :param example_no: example whose hypothesis value is to be calculated
    :return: hypothesis value for that example
    """

    if data_set == "train":
        return _hypothesis_value(train_data[example_no][0])

    elif data_set == "test":
        return _hypothesis_value(test_data[example_no][0])

    return None


def summation_of_cost_derivative(index, end=m):
    """
    Calculates the sum of cost function derivative

    :param index: index wrt derivative is being calculated
    :param end: value where summation ends

    :return: summation of cost derivative

    Note:
    If index is -1, derivative is calculated wrt bias parameter.
    """

    summation_value = 0

    for i in range(end):

        if index == -1:
            summation_value += _error(i)

        else:
            summation_value += _error(i) * train_data[i][0][index]

    return summation_value


def get_cost_derivative(index):
    """
    :param index: parameter index wrt derivative is calculated
    :return: derivative wrt that parameter
    """

    return summation_of_cost_derivative(index, m) / m


def compute_cost():
    """
    Computes Mean Squared Error Cost Function
    """

    total_error = 0

    for i in range(m):
        total_error += _error(i) ** 2

    return total_error / (2 * m)


def run_gradient_descent():
    """
    Runs gradient descent until convergence
    """

    global parameter_vector

    iteration = 0

    while iteration < MAX_ITERATIONS:

        iteration += 1

        temp_parameter_vector = np.zeros(len(parameter_vector))

        for i in range(len(parameter_vector)):

            cost_derivative = get_cost_derivative(i - 1)

            temp_parameter_vector[i] = (
                parameter_vector[i]
                - LEARNING_RATE * cost_derivative
            )

        current_cost = compute_cost()

        if np.allclose(
            parameter_vector,
            temp_parameter_vector,
            atol=TOLERANCE,
            rtol=0,
        ):
            print(f"Converged after {iteration} iterations")
            break

        parameter_vector = temp_parameter_vector

        if iteration % 1000 == 0:
            print(f"Iteration {iteration} | Cost: {current_cost:.6f}")

    print("\nFinal Parameters:")
    print(parameter_vector)


def test_gradient_descent():
    """
    Tests model on test data
    """

    print("\nTesting gradient descent for linear regression.\n")

    for i in range(len(test_data)):

        print(f"Actual output value : {output(i, 'test')}")

        print(
            f"Hypothesis output   : "
            f"{calculate_hypothesis_value(i, 'test'):.4f}"
        )

        print()


if __name__ == "__main__":

    run_gradient_descent()

    test_gradient_descent()