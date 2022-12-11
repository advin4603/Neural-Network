from typing import *
from activation import activation, activation_derivative
from random import uniform
from cost import node_cost_derivative


def generate_random_weight(scale: float) -> float:
    return uniform(-1, 1) * scale


class Layer:
    def __init__(self, nodes_in_count: int, nodes_out_count: int, weights: List[List[float]] = None,
                 biases: List[float] = None):
        self.nodes_in_count = nodes_in_count
        self.nodes_out_count = nodes_out_count

        inv_sqrt_nodes_in = 1 / (self.nodes_in_count ** 0.5)

        self.cost_gradient_w = [[0 for _ in range(self.nodes_out_count)] for _ in range(self.nodes_in_count)]
        self.cost_gradient_b = [0 for _ in range(self.nodes_out_count)]

        self.activations: List[float] = []
        self.weighted_inputs: List[float] = []
        self.inputs: List[float] = []

        self.weights = weights
        if self.weights is None:
            self.weights = [[generate_random_weight(inv_sqrt_nodes_in) for _ in range(self.nodes_out_count)] for _ in
                            range(self.nodes_in_count)]
        elif len(self.weights) != self.nodes_in_count or len(self.weights[0]) != self.nodes_out_count:
            raise ValueError("Incompatible weight dimensions.")
        self.biases = biases
        if self.biases is None:
            self.biases = [0 for _ in range(self.nodes_out_count)]
        elif len(self.biases) != self.nodes_out_count:
            raise ValueError("Incompatible biases dimensions.")

    def calculate_outputs(self, inputs: List[float]) -> List[float]:
        self.activations: List[float] = []
        self.weighted_inputs: List[float] = []
        self.inputs: List[float] = inputs.copy()
        if len(inputs) != self.nodes_in_count:
            raise ValueError("Incompatible input dimensions")
        for i in range(self.nodes_out_count):
            self.weighted_inputs.append(self.biases[i] + sum(
                inputs[j] * self.weights[j][i] for j in range(self.nodes_in_count)))
            self.activations.append(activation(self.weighted_inputs[i]))

        return self.activations

    def update_gradients(self, node_values: List[float]):

        if len(node_values) != self.nodes_out_count:
            raise ValueError("Incompatible node values dimensions")

        for i in range(self.nodes_out_count):
            for j in range(self.nodes_in_count):
                derivative_cost_wrt_wt = self.inputs[j] * node_values[i]
                self.cost_gradient_w[j][i] += derivative_cost_wrt_wt

            derivative_cost_wrt_bias = 1 * node_values[i]
            self.cost_gradient_b[i] += derivative_cost_wrt_bias

    def apply_gradients(self, learn_rate: float):
        for i in range(self.nodes_out_count):
            self.biases[i] -= self.cost_gradient_b[i] * learn_rate
            for j in range(self.nodes_in_count):
                self.weights[j][i] -= self.cost_gradient_w[j][i] * learn_rate

    def clear_gradients(self):
        self.cost_gradient_w = [[0 for _ in range(self.nodes_out_count)] for _ in range(self.nodes_in_count)]
        self.cost_gradient_b = [0 for _ in range(self.nodes_out_count)]

    def calculate_output_layer_node_values(self, expected_outputs: List[float]) -> List[float]:
        node_values = []

        if len(expected_outputs) != self.nodes_out_count:
            raise ValueError("Incompatible expected outputs dimensions")

        for i in range(len(expected_outputs)):
            cost_derivative = node_cost_derivative(self.activations[i], expected_outputs[i])
            activation_derivative_value = activation_derivative(self.weighted_inputs[i])
            node_values.append(activation_derivative_value * cost_derivative)

        return node_values

    def calculate_hidden_layer_node_values(self, old_layer: "Layer", old_node_values: List[float]) -> List[float]:
        new_node_values: List[float] = []

        for i in range(self.nodes_out_count):
            new_node_value = 0
            for j in range(len(old_node_values)):
                weighted_input_derivative = old_layer.weights[i][j]
                new_node_value += weighted_input_derivative * old_node_values[j]
            new_node_value *= activation_derivative(self.weighted_inputs[i])
            new_node_values.append(new_node_value)

        return new_node_values
