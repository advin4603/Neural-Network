def node_cost(output_activation: float, expected_activation: float):
    return (output_activation - expected_activation) ** 2


def node_cost_derivative(output_activation: float, expected_activation: float):
    return (output_activation - expected_activation) * 2
