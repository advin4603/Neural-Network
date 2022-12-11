import sys

import numpy as np
from typing import *
from pathlib import Path
import numpy.typing as npt
import random
import json


def sigmoid(z: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    s = sigmoid(z)
    return s * (1 - s)


def vectorized_result(j) -> npt.NDArray[np.float_]:
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


class Cost:
    @staticmethod
    def fn(a: npt.NDArray[np.float_], y: npt.NDArray[np.float_]) -> float:
        raise NotImplementedError()

    @staticmethod
    def delta(z: npt.NDArray[np.float_], a: npt.NDArray[np.float_], y: npt.NDArray[np.float_]) -> npt.NDArray[
        np.float_]:
        raise NotImplementedError()


class QuadraticCost(Cost):

    @staticmethod
    def fn(a: npt.NDArray[np.float_], y: npt.NDArray[np.float_]) -> float:
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z: npt.NDArray[np.float_], a: npt.NDArray[np.float_], y: npt.NDArray[np.float_]) -> npt.NDArray[
        np.float_]:
        return (a - y) * sigmoid_prime(z)


class CrossEntropyCost(Cost):

    @staticmethod
    def fn(a: npt.NDArray[np.float_], y: npt.NDArray[np.float_]):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z: npt.NDArray[np.float_], a: npt.NDArray[np.float_], y: npt.NDArray[np.float_]):
        return a - y


class Network:
    def __init__(self, sizes: Sequence[int], cost: Type[Cost] = CrossEntropyCost):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method).
        """
        self.biases: Sequence[npt.NDArray[np.float_]] = []
        self.weights: Sequence[npt.NDArray[np.float_]] = []
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost: Type[Cost] = cost
        self.default_weight_initializer()

    def large_weight_initializer(self):
        """Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        This weight and bias initializer uses the same approach as in
        Chapter 1, and is included for purposes of comparison.  It
        will usually be better to use the default weight initializer
        instead.

        """
        self.biases = [np.random.randn(size, 1) for size in self.sizes[1:]]
        self.weights = [np.random.randn(to_layer, from_layer) for from_layer, to_layer
                        in zip(self.sizes[:-1], self.sizes[1:])]

    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        """
        self.biases = [np.random.randn(size, 1) for size in self.sizes[1:]]
        self.weights = [np.random.randn(to_layer, from_layer) / np.sqrt(from_layer) for from_layer, to_layer
                        in zip(self.sizes[:-1], self.sizes[1:])]

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
            lmbda: float = 0.0,
            evaluation_data: Union[None, Sequence[Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]]] = None,
            monitor_evaluation_cost: bool = False,
            monitor_evaluation_accuracy: bool = False,
            monitor_training_cost: bool = False,
            monitor_training_accuracy: bool = False,
            epoch_callback: Callable[[int], None] = lambda n, t: print(f"Epoch {n} complete")
            ) -> Tuple[List[float], List[int], List[float], List[int]]:
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set."""
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = (training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size))
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, n)

            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print(f"Cost on training data: {cost}")

            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print(f"Accuracy on training data: {accuracy} / {n}")

            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print(f"Cost on evaluation data: {cost}")

            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print(f"Accuracy on evaluation data: {self.accuracy(evaluation_data)} / {len(evaluation_data)}")

            epoch_callback(epoch)
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch: Sequence[Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]],
                          eta: float, lmbda: float, n: int):
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

        self.weights = [(1 - eta * (lmbda / n)) * w - (eta / len(mini_batch)) * nw for w, nw in
                        zip(self.weights, nabla_w)]
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
        delta = self.cost.delta(zs[-1], activations[-1], out)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return nabla_b, nabla_w

    def accuracy(self, data: Sequence[Tuple[npt.NDArray[np.float_], Union[npt.NDArray[np.float_], float]]],
                 convert: bool = False) -> int:
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data.

        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for x, y in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for x, y in data]

        return sum(1 for (x, y) in results if x == y)

    def total_cost(self, data: Sequence[Tuple[npt.NDArray[np.float_], Union[npt.NDArray[np.float_], float]]],
                   lmbda: float, convert: bool = False) -> float:
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """

        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert:
                y = vectorized_result(y)
            cost += self.cost.fn(a, y) / len(data)
        cost += 0.5 * (lmbda / len(data)) * sum(np.linalg.norm(w) ** 2 for w in self.weights)
        return cost

    def save(self, filepath: Union[Path, str]):
        """Save the neural network to the file pointed to by ``filepath``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}

        with open(filepath, "w") as f:
            json.dump(data, f)


def load(filepath: Union[Path, str]) -> Network:
    """Load a neural network from the file at ``filepath``.  Returns an
    instance of Network.
    """
    with open(filepath) as f:
        data = json.load(f)

    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]

    return net
