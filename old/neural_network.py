from layer import Layer
from typing import *
from cost import node_cost

Datapoint = Tuple[List[float], List[float]]


class NeuralNetwork:
    def __init__(self, layer_sizes: List[int]):
        self.layer_sizes = layer_sizes
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1]))

    def calculate_outputs(self, inputs: List[float]) -> List[float]:
        for layer in self.layers:
            inputs = layer.calculate_outputs(inputs)

        return inputs

    def classify(self, inputs: List[float]):
        outputs = self.calculate_outputs(inputs)
        return max(range(len(outputs)), key=lambda i: outputs[i])

    def cost(self, datapoint: Datapoint):
        outputs = self.calculate_outputs(datapoint[0])
        cost = sum(node_cost(output, expected) for output, expected in zip(outputs, datapoint[1]))

        return cost

    def avg_cost(self, datapoints: List[Datapoint]):
        total_cost = sum(self.cost(datapoint) for datapoint in datapoints) / len(datapoints)
        return total_cost

    def apply_all_gradients(self, learn_rate: float):
        for layer in self.layers:
            layer.apply_gradients(learn_rate)

    def clear_all_gradients(self):
        for layer in self.layers:
            layer.clear_gradients()

    def update_all_gradients(self, datapoint: Datapoint):
        self.calculate_outputs(datapoint[0])
        output_layer = self.layers[-1]

        node_values = output_layer.calculate_output_layer_node_values(datapoint[1])
        output_layer.update_gradients(node_values)
        prev_layer = output_layer
        for hidden_layer in self.layers[-2::-1]:
            node_values = hidden_layer.calculate_hidden_layer_node_values(prev_layer, node_values)
            hidden_layer.update_gradients(node_values)
            prev_layer = hidden_layer

    def learn(self, training_batch: List[Datapoint], learn_rate: float):
        for datapoint in training_batch:
            self.update_all_gradients(datapoint)

        self.apply_all_gradients(learn_rate / len(training_batch))
        self.clear_all_gradients()
