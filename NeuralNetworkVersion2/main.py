import numpy as np
from neuralnetwork import NeuralNetwork
from layers import *

from tensorflow.keras.datasets import mnist

(xTrain, yTrain), (xTest, yTest) = mnist.load_data()

xTrain, xTest = np.array(xTrain[:1000]) / 255.0, np.array(xTest[:100]) / 255.0
xTrain, xTest = [x.flatten() for x in xTrain], [x1.flatten() for x1 in xTest]
yTrain, yTest = np.array(yTrain[:1000]), np.array(yTest[:100])
yTrain, yTest = [np.eye(10)[y] for y in yTrain], [np.eye(10)[y1] for y1 in yTest]

model = NeuralNetwork(Input(784))
model.add(Dense(16, "ReLU"))
model.add(Dropout(0.5))
model.add(Dense(16, "ReLU"))
model.add(Dropout(0.5))
model.add(Dense(10, "Softmax"))
model.initialize("HeNormal")
model.compile(["SGDMomentum", 0.9, 0.001], "CategoricalCrossEntropy", ["Accuracy", "Loss"])

model.train(xTrain, yTrain, 20, 10)
model.test(xTest, yTest)