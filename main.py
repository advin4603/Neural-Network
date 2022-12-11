from network import Network
import mnist_loader as mn
from alive_progress import alive_bar

training_data, validation_data, test_data = mn.load_data()

net = Network((784, 30, 10))


with alive_bar(30, title="Training") as bar:
    def update_bar(n, t):
        bar()
        if t is not None:
            print(f"{t[0]}/{t[1]}")
    net.sgd(training_data, 30, 10, 3.0, test_data=test_data, epoch_callback=update_bar)
