import csv
import math
import turtle
from neural_network import NeuralNetwork


def display(data):
    length = int(math.sqrt(len(data)))
    grid_data = [[data[length * j + i] for j in range(length)] for i in range(length)]
    pixel_size = 15
    setwindowsize(length * pixel_size, length * pixel_size)
    for i in range(length):
        for j in range(length):
            drawpixel(i, j, (grid_data[i][j], 0, 0), pixel_size)
    showimage()


def setwindowsize(x=640, y=640):
    turtle.setup(x, y)
    turtle.setworldcoordinates(0, 0, x, y)


def drawpixel(x, y, color, pixelsize=1):
    turtle.tracer(0, 0)
    turtle.colormode(255)
    turtle.penup()
    turtle.setpos(x * pixelsize, y * pixelsize)
    turtle.color(color)
    turtle.pendown()
    turtle.begin_fill()
    for i in range(4):
        turtle.forward(pixelsize)
        turtle.right(90)
    turtle.end_fill()


def showimage():
    turtle.hideturtle()
    turtle.update()
    turtle.exitonclick()


print("loading...")

with open("../train.csv") as file:
    reader = csv.reader(file)
    next(reader)
    train_data = []
    for row in reader:
        row = [int(n) for n in row]
        label = [0 for _ in range(10)]
        label[row[0]] = 1
        row = [i / 255 for i in row]
        train_data.append((row[1:], label))

test_data = train_data[-100:]
train_data = train_data[:-100]


def accuracy():
    correct = 0
    for datapoint in test_data:
        classification = nn.classify(datapoint[0])
        if max(range(10), key=lambda i: datapoint[1][i]) == classification:
            correct += 1

    print(correct / len(test_data) * 100)


nn = NeuralNetwork([len(train_data[0][0]), 100, 10])
batch_size = 30


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


print("training...")
for training_batch in chunks(train_data, batch_size):
    nn.learn(training_batch, 0.1)
    accuracy()
