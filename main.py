from network import Network
import mnist_loader as mn
from alive_progress import alive_bar

training_data, validation_data, test_data = mn.load_data()

net = Network((784, 30, 10))

with alive_bar(30, title="Training") as bar:
    def update_bar(n):
        bar()


    net.sgd(training_data, 30, 10, 0.5, lmbda=5.0,
            evaluation_data=test_data,
            epoch_callback=update_bar,
            monitor_training_cost=True,
            monitor_training_accuracy=True,
            monitor_evaluation_cost=True,
            monitor_evaluation_accuracy=True
            )

    net.save("model.json")
