import numpy as np
from tensorflow.keras.datasets import mnist

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def SigmoidGradient(x):
    return Sigmoid(x) * (1 - Sigmoid(x))

def Tanh(x):
    return np.tanh(x)

def TanhGradient(x):
    return (1 - Tanh(x) ** 2)

def ReLU(x):
    return np.maximum(x, np.zeros(x.shape))

def ReLUGradient(x):
    return np.where(x > 0, 1, 0)

def Softmax(x):
    return (1 / np.sum(np.exp(x))) * np.exp(x)

def SoftmaxGradient(x):
    return Softmax(x) * (1 - Softmax(x))

def SimpleError(actual, predicted):
    return actual - predicted

def SimpleErrorGradient(actual, predicted):
    return (-1) * np.ones(len(actual))

def SquaredError(actual, predicted):
    return (1/2) * np.square(actual - predicted)

def SquaredErrorGradient(actual, predicted):
    return actual - predicted

def AbsoluteError(actual, predicted):
    return np.absolute(actual - predicted)

def AbsoluteErrorGradient(actual, predicted):
    return np.sign(actual - predicted)

def CrossEntropyError(actual, predicted):
    return (-1) * actual * np.log10(predicted)

def CrossEntropyErrorGradient(actual, predicted):
    return (-1) * actual / predicted

def SGD(*params):
    return params[0] - params[1] * params[2]

def SGDMomentum(*params):
    velocity = params[1] * params[2] + (1 - params[1]) * params[4]
    return params[0] - params[1] * velocity - params[3] * params[4], velocity

def RMSProp(*params):
    decay = params[1] * params[2] + (1 - params[1]) * (params[4] ** 2)
    return params[0] - (params[3] * params[4]) / (np.sqrt(decay + 1e-8)), decay

def Adam(*params):
    velocity = (params[2] * params[3] + (1 - params[2]) * params[7])
    decay = (params[4] * params[5] + (1 - params[4]) * (params[7] ** 2))
    return params[0] - (params[6] * (velocity / (1 - params[2] ** (params[1] + 1)))) / (np.sqrt((decay / (1 - params[4] ** (params[1] + 1)))) + 1e-8), params[1] + 1, velocity, decay

class InputLayer:
    def __init__(self, numElements):
        self.numElements = numElements

activationFunctions = { "SIGMOID": Sigmoid, "TANH": Tanh, "RELU": ReLU, "SOFTMAX": Softmax }
activationGradients = { "SIGMOID": SigmoidGradient, "TANH": TanhGradient, "RELU": ReLUGradient, "SOFTMAX": SoftmaxGradient }
lossFunctions = { "SIMPLE": SimpleError, "SQUARED": SquaredError, "ABSOLUTE": AbsoluteError, "CROSSENTROPY": CrossEntropyError }
lossGradients = { "SIMPLE": SimpleErrorGradient, "SQUARED": SquaredErrorGradient, "ABSOLUTE": AbsoluteErrorGradient, "CROSSENTROPY": CrossEntropyErrorGradient }
optimizers = { "SGD": SGD, "SGDMOMENTUM": SGDMomentum, "RMSPROP": RMSProp, "ADAM": Adam }

class FeedForwardLayer:
    def __init__(self, numElements, activation):
        if isinstance(numElements, int):
            self.numElements = numElements
        else:
            raise Exception("The number of elements in this feed forward layer must be an integer")
        if (activation.upper() in ["SIGMOID", "TANH", "RELU", "SOFTMAX"]):
            self.activation = activation.upper()
        else:
            raise Exception("This neural network model does not support that activation function")

class FeedForwardNetwork:
    def __init__(self, *layers):
        if len(layers) < 1:
            raise Exception("This feed forward neural network must have at least one layer")
        if not isinstance(layers[0], InputLayer):
            raise Exception("The first layer in this feed forward layer must be an input layer")
        for layer in layers:
            if layers.index(layer) != 0 and not isinstance(layer, FeedForwardLayer): 
                raise Exception("This feed forward neural network must contain only feed forward layers")
        self.layers = list(layers)
        self.nums = [layr.numElements for layr in layers]
        self.weights = [np.random.randn(self.nums[a], self.nums[a + 1]) for a in range(len(self.nums) - 1)]
        self.biases = [np.random.randn(self.nums[b]) for b in range(1, len(self.nums))]
        self.loss = None
        self.optimizer = None

    def add(self, layer):
        if not isinstance(layer, FeedForwardLayer):
            raise Exception("This neural network only supports feed forward layers")
        self.nums.append(layer.numElements)
        self.layers.append(layer)
        self.weights.append(np.random.randn(self.nums[-2], self.nums[-1]))
        self.biases.append(np.random.randn(self.nums[-1]))

    def compute(self, inputs):
        for a in range(len(self.weights)):
            inputs = inputs @ self.weights[a]
            inputs = inputs + self.biases[a]
            inputs = activationFunctions[self.layers[a + 1].activation](inputs)
        return inputs
    
    def compile(self, optimizer, loss):
        if not (loss.upper() in ["SIMPLE", "SQUARED", "ABSOLUTE", "CROSSENTROPY"]):
            raise Exception("This neural network model does not support that loss function")
        if not isinstance(optimizer, list):
            raise Exception("The selected optimizer must be a list consisting of the name and parameters")
        if len(optimizer) < 2:
            raise Exception("The selected optimizer must be a list consisting of the name and parameters")
        if not isinstance(optimizer[0], str):
            raise Exception("The first element of the optimizer array must be a string with the name of the optimizer")
        if not optimizer[0].upper() in ["SGD", "SGDMOMENTUM", "RMSPROP", "ADAM"]:
            raise Exception("This neural network model does not support that optimizer")
        if (optimizer[0].upper() == "SGD" and len(optimizer) != 2):
            raise Exception("The SGD optimizer requires one and only one other parameter for the learning rate")
        if (optimizer[0].upper() == "SGD" and not (isinstance(optimizer[1], float) or isinstance(optimizer[1], int))):
            raise Exception("The SGD learning rate parameter must be an integer or float")
        if (optimizer[0].upper() == "SGDMOMENTUM" and len(optimizer) != 3):
            raise Exception("The SGD+Momentum optimizer requires exactly two parameters for the momentum decay rate and learning rate")
        if (optimizer[0].upper() == "SGDMOMENTUM" and not (isinstance(optimizer[2], float) or isinstance(optimizer[2], int))):
            raise Exception("The SGD+Momentum learning rate parameter must be an integer or float")
        if (optimizer[0].upper() == "SGDMOMENTUM" and not (isinstance(optimizer[1], float) and optimizer[1] > 0 and optimizer[1] < 1)):
            raise Exception("The SGD+Momentum momentum decay rate must be a float strictly between zero and one")
        if (optimizer[0].upper() == "RMSPROP" and len(optimizer) != 3):
            raise Exception("The RMSProp optimizer requires exactly two parameters for the square gradient decay rate and learning rate")
        if (optimizer[0].upper() == "RMSPROP" and not (isinstance(optimizer[2], float) or isinstance(optimizer[2], int))):
            raise Exception("The RMSProp learning rate parameter must be an integer or float")
        if (optimizer[0].upper() == "RMSPROP" and not (isinstance(optimizer[1], float) and optimizer[1] > 0 and optimizer[1] < 1)):
            raise Exception("The RMSProp square gradient decay rate must be a float strictly between zero and one")
        if (optimizer[0].upper() == "ADAM" and len(optimizer) != 4):
            raise Exception("The Adam optimizer requires exactly three parameters for the momentum decay rate, square gradient decay rate, and learning rate")
        if (optimizer[0].upper() == "ADAM" and not (isinstance(optimizer[3], float) or isinstance(optimizer[3], int))):
            raise Exception("The Adam learning rate parameter must be an integer or float")
        if (optimizer[0].upper() == "ADAM" and not (isinstance(optimizer[1], float) and optimizer[1] > 0 and optimizer[1] < 1)):
            raise Exception("The Adam momentum decay rate must be a float strictly between zero and one")
        if (optimizer[0].upper() == "ADAM" and not (isinstance(optimizer[2], float) and optimizer[2] > 0 and optimizer[2] < 1)):
            raise Exception("The Adam square gradient decay rate must be a float strictly between zero and one")
        self.loss = loss.upper()
        self.optimizer = optimizer
        self.optimizer[0] = self.optimizer[0].upper()
    
    def train(self, inputs, outputs, batchSize, epochs):
        if not (isinstance(inputs, np.ndarray) or isinstance(outputs, np.ndarray)):
            raise Exception("The input and output parameters must be arrays of data")
        if len(inputs) < 1 or len(outputs) < 1:
            raise Exception("The input and output arrays must contain more than one element")
        if len(inputs) != len(outputs):
            raise Exception("The input and output arrays must contain the same number of data points")
        for input in inputs:
            if len(input) != self.layers[0].numElements:
                raise Exception("The number of elements in each of the input data points must be equal to the number of elements in the input layer of this network")
        for output in outputs:
            if len(output) != self.layers[-1].numElements:
                raise Exception("The number of elements in each of the output data points must be equal to the number of elements in the last layer of this network")
        if not (isinstance(batchSize, int) and batchSize > 0):
            raise Exception("The batch size of this network must be an integer greater than zero")
        if not (isinstance(epochs, int) and epochs > 0):
            raise Exception("The number of epochs of this network must be an integer greater than zero")
        if self.loss is None or self.optimizer is None:
            raise Exception("The loss function and optimizer are not defined, call the compile(optimizer, loss) function on the model object to define them")

        time, velocity, decay = 0, 0, 0

        for a in range(epochs):
            rounds = len(inputs) // batchSize
            for b in range(rounds):
                actual = np.stack([outputs[d] for d in range(b * batchSize, (b + 1) * batchSize)])
                inputBatch = np.stack([inputs[c] for c in range(b * batchSize, (b + 1) * batchSize)])
                totalData = []
                totalData.append(inputBatch)
                totalData.append(inputBatch @ self.weights[0])
                totalData.append(totalData[1] + self.biases[0])
                totalData.append(activationFunctions[self.layers[1].activation](totalData[2]))
                for e in range(1, len(self.layers) - 1):
                    totalData.append(totalData[3 * e] @ self.weights[e])
                    totalData.append(totalData[3 * e + 1] + self.biases[e])
                    totalData.append(activationFunctions[self.layers[e + 1].activation](totalData[3 * e + 2]))
                predicted = totalData[-1]
                totalData[-1] = lossGradients[self.loss](actual, predicted)
                totalData[-2] = activationGradients[self.layers[-1].activation](totalData[-2]) * totalData[-1]
                totalData[-3] = totalData[-2]
                if self.optimizer == "SGD":
                    self.biases[-1] = SGD(self.biases[-1], self.optimizer[1], totalData[-2])
                    self.weights[-1] = SGD(self.weights[-1], self.optimizer[1], totalData[-3] @ totalData[-4].T)
                elif self.optimizer == "SGDMOMENTUM":
                    self.biases[-1], velocity = SGDMomentum(self.biases[-1], self.optimizer[1], velocity, self.optimizer[2], totalData[-2])
                    self.weights[-1], velocity = SGD(self.weights[-1], self.optimizer[1], velocity, self.optimizer[2], totalData[-3] @ totalData[-4].T)
                elif self.optimizer == "RMSPROP":
                    self.biases[-1], decay = RMSProp(self.biases[-1], self.optimizer[1], decay, self.optimizer[2], totalData[-2])
                    self.weights[-1], decay = RMSProp(self.weights[-1], self.optimizer[1], decay, self.optimizer[2], totalData[-3] @ totalData[-4].T)
                elif self.optimizer == "ADAM":
                    self.biases[-1], time, velocity, decay = Adam(self.biases[-1], time, self.optimizer[1], velocity, self.optimizer[2], decay, self.optimizer[3], totalData[-2])
                    self.weights[-1], time, velocity, decay = Adam(self.weights[-1], time, self.optimizer[1], velocity, self.optimizer[2], decay, self.optimizer[3], totalData[-3] @ totalData[-4].T)
                for f in range(1, len(self.layers) - 1):
                    totalData[-3 * f - 1] = totalData[-3 * f] @ self.weights[-1].T
                    totalData[-3 * f - 2] = activationGradients[self.layers[-f - 1].activation](totalData[-3 * f - 2]) * totalData[-3 * f - 1]
                    totalData[-3 * f - 3] = totalData[-3 * f - 2]
                    if self.optimizer == "SGD":
                        self.biases[-f] = SGD(self.biases[-f], self.optimizer[1], totalData[-3 * f - 2])
                        self.weights[-f] = SGD(self.weights[-f], self.optimizer[1], totalData[-3 * f - 3] @ totalData[-3 * f - 4].T)
                    elif self.optimizer == "SGDMOMENTUM":
                        self.biases[-f], velocity = SGDMomentum(self.biases[-f], self.optimizer[1], velocity, self.optimizer[2], totalData[-3 * f - 2])
                        self.weights[-f], velocity = SGD(self.weights[-f], self.optimizer[1], velocity, self.optimizer[2], totalData[-3 * f - 3] @ totalData[-3 * f - 4].T)
                    elif self.optimizer == "RMSPROP":
                        self.biases[-f], decay = RMSProp(self.biases[-f], self.optimizer[1], decay, self.optimizer[2], totalData[-3 * f - 2])
                        self.weights[-f], decay = RMSProp(self.weights[-f], self.optimizer[1], decay, self.optimizer[2], totalData[-3 * f - 3] @ totalData[-3 * f - 4].T)
                    elif self.optimizer == "ADAM":
                        self.biases[-f], time, velocity, decay = Adam(self.biases[-f], time, self.optimizer[1], velocity, self.optimizer[2], decay, self.optimizer[3], totalData[-3 * f - 2])
                        self.weights[-f], time, velocity, decay = Adam(self.weights[-f], time, self.optimizer[1], velocity, self.optimizer[2], decay, self.optimizer[3], totalData[-3 * f - 3] @ totalData[-3 * f - 4].T)

model = FeedForwardNetwork(InputLayer(784), FeedForwardLayer(10, "ReLU"), FeedForwardLayer(10, "Softmax"))
model.compile(["Adam", 0.9, 0.999, 0.01], "CrossEntropy")

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train, x_test, y_test = x_train[:1000] / 255, y_train[:1000], x_test[:100] / 255, y_test[:100]

x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

model.train(x_train, y_train, 5, 5)

correct, total = 0, 0
for g in range(len(x_test)):
    predicted = model.compute(x_test[g])
    if (np.where(predicted == max(predicted)) == np.where(y_test[g] == max(y_test[g]))):
        correct += 1
    total += 1

print(f"Accuracy: {correct / total}")