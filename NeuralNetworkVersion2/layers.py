import numpy as np
import activations
import initializations
import optimizers
import errors

class Input:
    def __init__(self, number: int) -> None:
        if not (isinstance(number, int) and number > 0):
            errors.printError(number, "Please provide an integer greater than zero as the number")
            raise Exception("The number of neurons must be an integer greater than zero")
        self._number = number
        self._optimizable = False
        self._initializable = False
        self._activatable = False

class Dense:
    def __init__(self, number: int, activation: str | None = None, initialization: str = "ZERO") -> None:
        if not (isinstance(number, int) and number > 0):
            errors.printError(number, "Please provide an integer greater than zero as the number")
            raise Exception("The number of neurons must be an integer greater than zero")
        if not (activation is None or (isinstance(activation, str) and activation.upper() in activations.activationFunctions.keys())):
            errors.printError(activation, "Please provide an valid activation function to this layer")
            raise Exception(f'This neural network framework does not support the activation function "{activation}." \n\nHere is a list of activation functions that this framework does support (case-insensitive): \n\n{activations.activationFunctions.keys()}')
        if not (isinstance(initialization, str) and initialization.upper() in initializations.initializationFunctions.keys()):
            errors.printError(initialization, "Please provide valid initialization function to this layer")
            raise Exception(f'This neural network framework does not support the weight initialization "{initialization}." \n\nHere is a list of weight initializations that this framework does support (case-insensitive): \n\n{initializations.initializationFunctions.keys()}')

        self._number = number
        self._activation = activation.upper()
        self._initialization = initialization.upper()

        self._currentInput = None
        self._weights = None
        self._biases = None
        self._ran = False
        self._inNum = None

        self._optimizer = None
        self._optimizerParamsWeights = None
        self._optimizerParamsBiases = None
        
        self._optimizable = True
        self._initializable = True
        self._activatable = True

    def _initialize(self, initialization: str) -> None:
        if not (isinstance(initialization, str) and initialization.upper() in initializations.initializationFunctions.keys()):
            errors.printError(initialization, "Please provide valid initialization function to this network")
            raise Exception(f'This neural network framework does not support the weight initialization "{initialization}." \n\nHere is a list of weight initializations that this framework does support (case-insensitive): \n\n{initializations.initializationFunctions.keys()}')
        if self._initialization == "ZERO" or self._initialization is None:
            self._initialization = initialization.upper()

    def _activate(self, activation: str) -> None:
        if not (activation is None or (isinstance(activation, str) and activation.upper() in activations.activationFunctions.keys())):
            errors.printError(activation, "Please provide a valid activation function to this network")
            raise Exception(f'This neural network framework does not support the activation function "{activation}." \n\nHere is a list of activation functions that this framework does support (case-insensitive): \n\n{activations.activationFunctions.keys()}')
        if self._activation is None:
            self._activation = activation.upper()

    def _optimize(self, optimizer: str, params: list[float]) -> None:
        if not (isinstance(optimizer, str) and optimizer.upper() in optimizers.optimizers.keys()):
            errors.printError(optimizer, "Please provide a valid optimizer to this network")
            raise Exception(f'The optimizer "{optimizer}" provided here is not recognized')
        if not (isinstance(params, list)):
            errors.printError(params, "Please provide list of parameters to the optimizer")
            raise Exception("The parameters provided to this optimizer must be a list")
        if not (len(params) == len(optimizers.optimizerParams[optimizer.upper()][1])):
            errors.printError(params, "Please provide the correct number of parameters")
            raise Exception(f'The number of parameters provided does not match the number of parameters needed for the optimizer "{optimizer}"')
        for param in params:
            if not (isinstance(param, float)):
                errors.printError(params, "Please make sure all parameters are floats")
                raise Exception("The parameters provided must be all float")
        self._optimizer = optimizer.upper()
        self._optimizerParamsWeights = optimizers.optimizerParams[self._optimizer].copy()
        self._optimizerParamsWeights[1] = params
        self._optimizerParamsBiases = self._optimizerParamsWeights.copy()

    def _forwardPass(self, inputArray, test: bool | None = False) -> np.ndarray:
        try:
            inputArr = np.array(inputArray, dtype = float)
        except Exception:
            errors.printError(inputArray, "Please input an iterable that is rectangular and consists of only numbers")
            raise
        else:
            if not isinstance(test, bool):
                errors.printError(test, "Please provide a True or False value")
                raise Exception("The test parameter in the Dense forward pass must be either True or False")
            if not inputArr.ndim == 2:
                errors.printError(inputArr, "Please input an array of vectors, nothing else")
                raise Exception("The input to the Dense forward pass must be a 2 dimensional array of vectors, nothing else")
            
            if not self._ran:
                self._inNum = inputArr.shape[1]
                self._weights = initializations.initializationFunctions[self._initialization](self._inNum, self._number)
                self._biases = np.zeros(self._number)
                self._ran = True

            if not (inputArr.shape[1] == self._inNum):
                errors.printError(inputArr, "Please provide an input that matches with the input shape of the corresponding layer")
                raise Exception("The input to this layer contains an incorrect shape of elements")
            
            if not test:
                self._currentInput = inputArr.copy()

            final = (inputArr @ self._weights) + self._biases

            if not self._activation is None:
                final = activations.activationFunctions[self._activation](final)

            return final
    
    def _backwardPass(self, inputArray) -> np.ndarray:
        try:
            inputArr = np.array(inputArray, dtype = float)
        except Exception:
            errors.printError(inputArray, "Please input an iterable that is rectangular and consists of only numbers")
            raise
        else:
            if not inputArr.ndim == 2:
                errors.printError(inputArr, "Please input an array of vectors, nothing else")
                raise Exception("The input to the Dense backward pass must be a 2 dimensional array of vectors, nothing else")
            if not (inputArr.shape[1] == self._number):
                errors.printError(inputArr, "Please provide an input that matches with the input shape of the corresponding layer")
                raise Exception("The input to this layer contains an incorrect shape of elements")
            if self._optimizer is None:
                errors.printError(inputArr, "Please provide an optimizer before running backpropagation")
                raise Exception("This model has not been optimized yet, please do so with the compile() function")
            
            backwards = inputArr.copy()

            if not self._activation is None:
                backwards = activations.activationGradients[self._activation](backwards)

            if self._currentInput is None:
                errors.printError(inputArr, "Please run the forward pass before running the backward pass")
                raise Exception("The Dense forward pass must be called before the backward pass")

            weightT = self._weights.T

            self._optimizerParamsBiases[0] = [self._biases, backwards]
            self._biases, self._optimizerParamsBiases[2] = optimizers.optimizers[self._optimizer](self._optimizerParamsBiases)
            self._optimizerParamsWeights[0] = [self._weights, self._currentInput.T @ backwards]
            self._weights, self._optimizerParamsWeights[2] = optimizers.optimizers[self._optimizer](self._optimizerParamsWeights)

            backwards = backwards @ weightT

            self._currentInput = None

            return backwards

class Activation:
    def __init__(self, activation: str) -> None:
        if not (isinstance(activation, str) and activation.upper() in activations.activationFunctions.keys()):
            errors.printError(activation, "Please provide a string representing a valid activation function")
            raise Exception(f'This neural network framework does not support the activation function "{activation}." \n\nHere is a list of activation functions that this framework does support (case-insensitive): \n\n{activations.activationFunctions.keys()}')
        self._activation = activation.upper()

        self._optimizable = False
        self._initializable = False
        self._activatable = False

    def _forwardPass(self, inputArray, test: bool | None = False) -> np.ndarray:
        try:
            inputArr = np.array(inputArray, dtype = float)
        except Exception:
            errors.printError(inputArray, "Please input an iterable that is rectangular and consists of only numbers")
            raise
        else:
            if not inputArr.ndim == 2:
                errors.printError(inputArr, "Please input an array of vectors, nothing else")
                raise Exception("The input to the Activation forward pass must be a 2 dimensional array of vectors, nothing else")
        
            return activations.activationFunctions[self._activation](inputArr)
    
    def _backwardPass(self, inputArray) -> np.ndarray:
        try:
            inputArr = np.array(inputArray, dtype = float)
        except Exception:
            errors.printError(inputArray, "Please input an iterable that is rectangular and consists of only numbers")
            raise
        else:
            if not inputArr.ndim == 2:
                errors.printError(inputArr, "Please input an array of vectors, nothing else")
                raise Exception("The input to the Activation backward pass must be a 2 dimensional array of vectors, nothing else")
        
            return activations.activationGradients[self._activation](inputArr)

class Dropout:
    def __init__(self, percentage: float) -> None:
        if not (isinstance(percentage, float) and percentage > 0 and percentage < 1):
            errors.printError(percentage, "Please input a percentage that is a float between zero and one")
            raise Exception("The Dropout percentage must be a float between zero and one, exclusive")
        
        self._percentage = percentage
        self._currentMask = None
        self._optimizable = False
        self._initializable = False
        self._activatable = False

    def _forwardPass(self, inputArray, test: bool | None = False) -> np.ndarray:
        try:
            inputArr = np.array(inputArray, dtype = float)
        except Exception:
            errors.printError(inputArray, "Please input an iterable that is rectangular and consists of only numbers")
            raise
        else:
            if not inputArr.ndim == 2:
                errors.printError(inputArr, "Please input an array of vectors, nothing else")
                raise Exception("The input to the Dropout forward pass must be a 2 dimensional array of vectors, nothing else")
            
            if test:
                return inputArr
            
            self._currentMask = np.random.binomial(1, 1 - self._percentage, size = inputArr.shape) 
            
            return inputArr * self._currentMask / (1 - self._percentage)
    
    def _backwardPass(self, inputArray) -> np.ndarray:
        try:
            inputArr = np.array(inputArray, dtype = float)
        except Exception:
            errors.printError(inputArray, "Please input an iterable that is rectangular and consists of only numbers")
            raise
        else:
            if not inputArr.ndim == 2:
                errors.printError(inputArr, "Please input an array of vectors, nothing else")
                raise Exception("The input to the Dropout backward pass must be a 2 dimensional array of vectors, nothing else")
            if self._currentMask is None:
                errors.printError(inputArr, "Please run the forward pass before running the backward pass")
                raise Exception("The forward pass must be called before the backward pass on Dropout")
            
            returnVar = inputArr * self._currentMask / (1 - self._percentage)
            self._currentMask = None

            return returnVar

class BatchNorm:
    def __init__(self, momentum: float) -> None:
        if not (isinstance(momentum, float) and momentum > 0 and momentum < 1):
            errors.printError(momentum, "Please input a momentum that is a float between zero and one")
            raise Exception("The Batch Norm momentum must be a float between zero and one, exclusive")
        
        self._momentum = momentum
        self._runningMean = None
        self._runningVar = None
        self._gamma = None
        self._beta = None

        self._optimizer = None
        self._optimizerParamsGamma = None
        self._optimizerParamsBeta = None

        self._optimizable = True
        self._initializable = False
        self._activatable = False

    def _optimize(self, optimizer: str, params: list[float]) -> None:
        if not (isinstance(optimizer, str) and optimizer.upper() in optimizers.optimizers.keys()):
            errors.printError(optimizer, "Please provide a valid optimizer to this network")
            raise Exception(f'The optimizer "{optimizer}" provided here is not recognized')
        if not (isinstance(params, list)):
            errors.printError(params, "Please provide list of parameters to the optimizer")
            raise Exception("The parameters provided to this optimizer must be a list")
        if not (len(params) == len(optimizers.optimizerParams[optimizer.upper()][1])):
            errors.printError(params, "Please provide the correct number of parameters")
            raise Exception(f'The number of parameters provided does not match the number of parameters needed for the optimizer "{optimizer}"')
        for param in params:
            if not (isinstance(param, float)):
                errors.printError(params, "Please make sure all parameters are floats")
                raise Exception("The parameters provided must be all float")
        self._optimizer = optimizer.upper()
        self._optimizerParamsGamma = optimizers.optimizerParams[self._optimizer]
        self._optimizerParamsGamma[1] = params
        self._optimizerParamsBeta = self._optimizerParamsGamma.copy()

    def _forwardPass(self, inputArray, test: bool | None = False) -> np.ndarray:
        try:
            inputArr = np.array(inputArray, dtype = float)
        except Exception:
            errors.printError(inputArray, "Please input an iterable that is rectangular and consists of only numbers")
            raise
        else:
            if not inputArr.ndim == 2:
                errors.printError(inputArr, "Please input an array of vectors, nothing else")
                raise Exception("The input to the Batch Norm forward pass must be a 2 dimensional array of vectors, nothing else")
            if self._gamma is None:
                self._gamma = np.ones(inputArr.shape)
                self._beta = np.zeros(inputArr.shape)
                self._runningMean = 0
                self._runningVar = 0

                self._currentInput = None
                self._currentNormal = None
                self._currentBatchMean = None
                self._currentBatchVar = None
            else:
                if inputArr.shape != self._gamma.shape:
                    errors.printError(inputArr, "Please provide an input that matches with the input shape of the corresponding layer")
                    raise Exception("The input to the Batch Normalization forward pass contains the wrong number of elements")
                
            if test:
                return self._gamma * ((inputArr - self._runningMean) / (np.sqrt(self._runningVar + 1e-8))) + self._beta
            else:
                batchMean = np.mean(inputArr)
                batchVar = np.var(inputArr)

                inputNormal = (inputArr - batchMean) / (np.sqrt(batchVar + 1e-8))
                outputArr = self._gamma * (inputNormal) + self._beta

                self._runningMean = self._momentum * self._runningMean + (1 - self._momentum) * batchMean
                self._runningVar = self._momentum * self._runningVar + (1 - self._momentum) * batchVar

                self._currentInput, self._currentNormal, self._currentBatchMean, self._currentBatchVar = inputArr, inputNormal, batchMean, batchVar

                return outputArr
        
    def _backwardPass(self, inputArray) -> np.ndarray:
        try:
            inputArr = np.array(inputArray, dtype = float)
        except Exception:
            errors.printError(inputArray, "Please input an iterable that is rectangular and consists of only numbers")
            raise
        else:
            if not inputArr.ndim == 2:
                errors.printError(inputArr, "Please input an array of vectors, nothing else")
                raise Exception("The input to the Batch Norm backward pass must be a 2 dimensional array of vectors, nothing else")
            if self._optimizer is None:
                errors.printError(inputArr, "Please provide an optimizer before running backpropagation")
                raise Exception("This model has not been optimized yet, please do so with the compile() function")

            normalGradient = inputArr * self._gamma

            self._optimizerParamsBeta[0] = [self._beta, np.sum(inputArr, axis = 0)]
            self._beta, self._optimizerParamsBeta[2] = optimizers.optimizers[self._optimizer](self._optimizerParamsBeta)

            self._optimizerParamsGamma[0] = [self._gamma, np.sum(inputArr * self._currentNormal, axis = 0)]
            self._gamma, self._optimizerParamsGamma[2] = optimizers.optimizers[self._optimizer](self._optimizerParamsGamma)

            varGradient = np.sum(normalGradient * (self._currentInput - self._currentBatchMean), axis = 0) * -0.5 * (self._currentBatchVar + 1e-8) ** -1.5
            meanGradient = np.sum(normalGradient * -1 / np.sqrt(self._currentBatchVar + 1e-8), axis = 0) + varGradient * np.mean(-2 * (self._currentInput - self._currentBatchMean), axis = 0)

            returnVar = (normalGradient / np.sqrt(self._currentBatchVar + 1e-8)) + (varGradient * 2 * (self._currentInput - self._currentBatchMean) / inputArr.shape[1]) + (meanGradient / inputArr.shape[1])

            self._currentInput, self._currentNormal, self._currentBatchMean, self._currentBatchVar = None, None, None, None

            return returnVar
        
layers = {
    "DENSE": Dense,
    "ACTIVATION": Activation,
    "DROPOUT": Dropout,
    "BATCHNORM": BatchNorm
}