import numpy as np
from typing import *
import numpy.typing as npt
import random


def sigmoid(z: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    s = sigmoid(z)
    return s * (1 - s)


class Network:
    def __init__(self, sizes: Sequence[int]):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases: Sequence[npt.NDArray[np.float_]] = [np.random.randn(size, 1) for size in sizes[1:]]
        self.weights: Sequence[npt.NDArray[np.float_]] = [np.random.randn(to_layer, from_layer) for from_layer, to_layer
                                                          in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        """Return the output of the network if "a" is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def sgd(self,
            training_data: MutableSequence[Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]],
            epochs: int,
            mini_batch_size: int,
            eta: float,
            test_data: Union[None, Sequence[Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]]] = None,
            epoch_callback: Callable[[int, Union[None, Tuple[int, int]]], None] = lambda n, t: print(
                f"Epoch {n} complete") if t is None else print(f"Epoch {n}: {t[0]} / {t[1]}")
            ):
        """Train the neural network using mini-batch stochastic
            gradient descent.  The "training_data" is a list of tuples
            "(x, y)" representing the training inputs and the desired
            outputs.  The other non-optional parameters are
            self-explanatory.  If "test_data" is provided then the
            network will be evaluated against the test data after each
            epoch, and partial progress printed out.  This is useful for
            tracking progress, but slows things down substantially."""
        n = len(training_data)
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = (training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size))
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data is not None:
                epoch_callback(epoch, (self.evaluate(test_data), len(test_data)))
            else:
                epoch_callback(epoch, None)

    def update_mini_batch(self, mini_batch: Sequence[Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]],
                          eta: float):
        """Update the network's weights and biases by applying
                gradient descent using backpropagation to a single mini batch.
                The "mini_batch" is a list of tuples "(x, y)", and "eta"
                is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for inp, out in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(inp, out)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, inp: npt.NDArray[np.float_], out: npt.NDArray[np.float_]) -> Tuple[
        Sequence[npt.NDArray[np.float_]], Sequence[npt.NDArray[np.float_]]]:
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
                gradient for the cost function C_x.  ``nabla_b`` and
                ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
                to ``self.biases`` and ``self.weights``."""

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation: npt.NDArray[np.float_] = inp
        activations: List[npt.NDArray[np.float_]] = [inp]  # list to store all the activations, layer by layer

        zs: List[npt.NDArray[np.float_]] = []  # list to store all the z vectors, layer by layer

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], out) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return nabla_b, nabla_w

    def evaluate(self, test_data: Sequence[Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]]):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""

        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for x, y in test_data]
        print(test_results[0])
        return sum(1 for (x, y) in test_results if x == y)

    @staticmethod
    def cost_derivative(output_activations: npt.NDArray[np.float_], y: npt.NDArray[np.float_]):
        """Return the vector of partial derivatives \partial C_x /
                \partial a for the output activations."""
        return output_activations - y