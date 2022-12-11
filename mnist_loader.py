import csv
import numpy as np
from alive_progress import alive_bar


def load_data():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 30,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarray containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    with open("./train.csv") as file, alive_bar(42000, title="Loading") as bar:
        reader = csv.reader(file)
        next(reader)
        training_data = []
        validation_data = []
        test_data = []
        i = 0
        for row in reader:
            label = vectorized_result(int(row[0]))
            pixels = np.array(tuple((float(int(i) / 255),) for i in row[1:]))
            if i < 30_000:
                training_data.append((pixels, label))
            elif 30_000 <= i < 40_000:
                validation_data.append((pixels, label))
            else:
                test_data.append((pixels, label))
            i += 1
            bar()
    return training_data, validation_data, test_data


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
