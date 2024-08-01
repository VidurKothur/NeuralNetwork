import numpy as np
import activations
import initializations
import layers
import losses
import metrics
import optimizers
import errors

class NeuralNetwork:
    def __init__(self, *lays) -> None:
        layerOptions = layers.layers.values()
        if len(lays) < 1:
            errors.printError(lays, "Please provide at least one layer for this network")
            raise Exception("The number of layers in a neural network must be at least an input layer")
        for layer in lays:
            if lays.index(layer) == 0:
                continue
            if not (type(layer) in layerOptions):
                errors.printError(lays, "Please provide valid layers to this network")
                raise Exception(f'The layer "{layer}" is not a valid layer')
        if not (isinstance(lays[0], layers.Input)):
            errors.printError(lays, "Please provide an input layer as the first layer")
            raise Exception("The first layer in a neural network must always be an input layer")
        
        self._layers = list(lays)
        self._loss = None
        self._metrics = []
        self._batchSize = None

    def add(self, layer) -> None:
        layerOptions = layers.layers.values()
        if not (type(layer) in layerOptions):
            errors.printError(layer, "Please provide a valid layer for this network")
            raise Exception(f'The layer "{layer}" is not a valid layer')
        self._layers.append(layer)

    def initialize(self, initialization: str) -> None:
        if not (isinstance(initialization, str)):
            errors.printError(initialization, "Please provide a string for the initialization parameter")
            raise Exception("The initialization function provided must be a string")
        if not (initialization.upper() in initializations.initializationFunctions.keys()):
            errors.printError(initialization, "Please provide a valid initialization function for this network")
            raise Exception(f'This neural network framework does not support the weight initialization "{initialization}." \n\nHere is a list of weight initializations that this framework does support (case-insensitive): \n\n{initializations.initializationFunctions.keys()}')
        
        for layer in self._layers:
            if layer._initializable:
                layer._initialize(initialization)

    def activate(self, activation: str) -> None:
        if not (isinstance(activation, str)):
            errors.printError(activation, "Please provide a string for the activation parameter")
            raise Exception("The initialization function provided must be a string")
        if not (activation.upper() in activations.activationFunctions.keys()):
            errors.printError(activation, "Please provide a valid activation function for this network")
            raise Exception(f'This neural network framework does not support the activation function "{activation}." \n\nHere is a list of activation functions that this framework does support (case-insensitive): \n\n{activations.activationFunctions.keys()}')
        
        for layer in self._layers:
            if layer._activatable:
                layer._activate(activation)

    def compile(self, optimizer: list, loss: str, mets: list | None = []) -> None:
        if not (isinstance(loss, str)):
            errors.printError(loss, "Please provide a valid loss function for this network")
            raise Exception("Please enter a string for the loss function with the function's name")
        if not (loss.upper() in losses.lossFunctions.keys()):
            errors.printError(loss, "Please provide a valid loss function for this network")
            raise Exception(f'This neural network framework does not support the loss function "{loss}." \n\nHere is a list of loss functions that this framework does support (case-insensitive): \n\n{losses.lossFunctions.keys()}')
        if not (isinstance(optimizer, list) and len(optimizer) > 0 and isinstance(optimizer[0], str)):
            errors.printError(optimizer, "Please provide a list containing valid parameters")
            raise Exception("Please enter a list consisting of the name of the optimizer plus its parameters")
        if not (optimizer[0].upper() in optimizers.optimizers.keys()):
            errors.printError(optimizer, "Please provide a valid optimizer for this network")
            raise Exception(f'This neural network framework does not support the optimizer "{optimizer}." \n\nHere is a list of optimizers that this framework does support (case-insensitive): \n\n{optimizers.optimizers.keys()}')
        if not (len(optimizer) == 1 or len(optimizer) == 1 + len(optimizers.optimizerParams[optimizer[0].upper()][1])):
            errors.printError(optimizer, "Please provide either all or no parameters for this optimizer")
            raise Exception(f'This network does not accept anything other than all parameters or no parameters to avoid parameter confusion. \n\nFor the {optimizer} optimizer, the template should look like this: \n\n{optimizers.optimizerTemplates[optimizer.upper()]}')
        if not (isinstance(mets, list)):
            errors.printError(mets, "Please provide a list of metrics")
            raise Exception("Please enter a list of strings containing the metrics that you want to view during training")
        for metric in mets:
            if not (metric.upper() in metrics.metrics.keys() or metric.upper() == "LOSS"):
                errors.printError(metric, "Please provide valid metrics for this network")
                raise Exception(f'This neural network framework does not support the metric "{metric}." \n\nHere is a list of metrics that this framework does support (case-insensitive): \n\n{metrics.metrics.keys()}')
            self._metrics.append(metric.upper())

        for layer in self._layers:
            if layer._optimizable:
                if len(optimizer) == 1:
                    layer._optimize(optimizer[0], optimizers.optimizerParams[optimizer.upper()][1])
                else:
                    layer._optimize(optimizer[0], optimizer[1:])

        self._loss = loss.upper()

    def train(self, inputs, outputs, batchSize: int, epochs: int) -> None:
        try:
            inputArr = np.array(inputs, dtype = float)
            outputArr = np.array(outputs, dtype = float)
        except Exception:
            errors.printError([inputArr, outputArr], "Please input an iterable that is rectangular and consists of only numbers")
            raise
        else:
            if not (inputArr.ndim == 2 and outputArr.ndim == 2):
                errors.printError([inputArr, outputArr], "Please input an array of vectors, nothing else")
                raise Exception("The input to the train function must be a stack of vectors, no more than 2 dimensions")
            if not (inputArr.shape[1] == self._layers[0]._number):
                errors.printError([inputArr, outputArr], "Please provide input data that matches with the shape of the first layer in the network")
                raise Exception("The input training data must consist of the same number of columns as the first layer in the network")
            if not (outputArr.shape[1] == self._layers[-1]._number):
                errors.printError([inputArr, outputArr], "Please provide label data that matches with the shape of the last layer in the network")
                raise Exception("The label training data must consist of the same number of columns as the last layer in the network")    
            if not (isinstance(batchSize, int) and batchSize > 0):
                errors.printError(batchSize,"Please provide a positive integer for the batch size")
                raise Exception("The batch size of this network must be an integer greater than zero")
            if not (isinstance(epochs, int) and epochs > 0):
                errors.printError(epochs,"Please provide a positive integer for the number of epochs")
                raise Exception("The number of epochs of this network must be an integer greater than zero")
            if self._loss is None:
                errors.printError([inputArr, outputArr],"Please run the compile function before training the network")
                raise Exception("The loss function and optimizer are not defined, call the compile function on the model object to define them")
            self._batchSize = batchSize

            for a in range(epochs):
                rounds = len(inputArr) // batchSize
                predicted = None

                print(f"\nEpoch {a + 1}:\n")

                for b in range(rounds):
                    actual = np.stack([outputArr[d] for d in range(b * batchSize, (b + 1) * batchSize)])
                    inputBatch = np.stack([inputArr[c] for c in range(b * batchSize, (b + 1) * batchSize)])

                    for layer in self._layers:
                        if self._layers.index(layer) == 0:
                            continue
                        inputBatch = layer._forwardPass(inputBatch)

                    if b == 0:
                        predicted = inputBatch.copy()
                    else:
                        predicted = np.concatenate((predicted, inputBatch))

                    lossGradient = losses.lossGradients[self._loss](actual, inputBatch)
                    for c in range(1, len(self._layers)):
                        lossGradient = self._layers[-c]._backwardPass(lossGradient)

                for m in range(len(self._metrics)):
                    if self._metrics[m].upper() == "LOSS":
                        print("Loss: ", losses.lossFunctions[self._loss](outputArr, predicted))   
                    else:
                        print(metrics.metricDisplays[self._metrics[m]], metrics.metrics[self._metrics[m]](outputArr, predicted))

    def test(self, inputs, outputs) -> None:
        try:
            inputArr = np.array(inputs, dtype = float)
            outputArr = np.array(outputs, dtype = float)
        except Exception:
            errors.printError([inputArr, outputArr], "Please input an iterable that is rectangular and consists of only numbers")
            raise
        else:
            if not (inputArr.ndim == 2 and outputArr.ndim == 2):
                errors.printError([inputArr, outputArr], "Please input either an array of vectors, nothing else")
                raise Exception("The input to the train function must be a stack of vectors, no more than 2 dimensions")
            if not (inputArr.shape[1] == self._layers[0]._number):
                errors.printError([inputArr, outputArr], "Please provide input data that matches with the shape of the first layer in the network")
                raise Exception("The input training data must consist of the same number of columns as the first layer in the network")
            if not (outputArr.shape[1] == self._layers[-1]._number):
                errors.printError([inputArr, outputArr], "Please provide label data that matches with the shape of the last layer in the network")
                raise Exception("The label training data must consist of the same number of columns as the last layer in the network")  
            if self._loss is None:
                errors.printError([inputArr, outputArr],"Please run the compile function before training the network")
                raise Exception("The loss function and optimizer are not defined, call the compile function on the model object to define them")
        
            rounds = len(inputArr) // self._batchSize
            predicted = None

            print(f"\nTest:\n")

            for b in range(rounds):
                inputBatch = np.stack([inputArr[c] for c in range(b * self._batchSize, (b + 1) * self._batchSize)])

                for layer in self._layers:
                    if self._layers.index(layer) == 0:
                        continue
                    print(inputBatch.shape)
                    inputBatch = layer._forwardPass(inputBatch)
                    print(inputBatch.shape, "\n")

                if b == 0:
                    predicted = inputBatch.copy()
                else:
                    predicted = np.concatenate((predicted, inputBatch))

            for m in range(len(self._metrics)):
                if self._metrics[m].upper() == "LOSS":
                    print("Loss: ", losses.lossFunctions[self._loss](outputArr, predicted))   
                else:
                    print(metrics.metricDisplays[self._metrics[m]], metrics.metrics[self._metrics[m]](outputArr, predicted))
