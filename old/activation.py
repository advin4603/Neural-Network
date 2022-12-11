from math import exp


def activation(weighted_input: float) -> float:
    return 1 / (1 + exp(-weighted_input))


def activation_derivative(weighted_input: float) -> float:
    activation_value = activation(weighted_input)
    return activation_value * (1 - activation_value)
