import numpy as np
import errors

"""
These activation functions will take in a numpy array (can be multidimensional) and apply element-wise activation.

More activation functions can be added, please make sure to take in a single np.ndarray as input and output a single
np.ndarray, please add a gradient as well with the same input and output, and please update both dictionaries with
ALL CAPS for the function name.
"""

def Sigmoid(inp) -> np.ndarray:
    try:
        x = np.array(inp, dtype = float)
    except Exception:
        errors.printError(inp, "Please input an iterable that is rectangular and consists of only numbers")
        raise
    else:
        x = np.where(x > 100, 100, x)
        return np.round(1 / (1 + np.exp(-x)), decimals = 10)

def Tanh(inp) -> np.ndarray:
    try:
        x = np.array(inp, dtype = float)
    except Exception:
        errors.printError(inp, "Please input an iterable that is rectangular and consists of only numbers")
        raise
    else:
        return np.round(np.tanh(x), decimals = 10)

def ReLU(inp) -> np.ndarray:
    try:
        x = np.array(inp, dtype = float)
    except Exception:
        errors.printError(inp, "Please input an iterable that is rectangular and consists of only numbers")
        raise
    else:
        x = np.where(x > np.e ** 100, np.e ** 100, x)
        return np.round(np.maximum(np.zeros(x.shape), x), decimals = 10)

def Softmax(inp) -> np.ndarray:
    try:
        x = np.array(inp, dtype = float)
    except Exception:
        errors.printError(inp, "Please input an iterable that is rectangular and consists of only numbers")
        raise
    else:
        x2 = np.where(x > 100, np.e ** 100, np.where(x < -100, np.e ** -100, np.e ** x))
        return np.round(x2 / np.sum(x2), decimals = 10)

activationFunctions = { 
    "SIGMOID": Sigmoid, 
    "TANH": Tanh, 
    "RELU": ReLU, 
    "SOFTMAX": Softmax 
}

def SigmoidGradient(inp) -> np.ndarray:
    try:
        x = np.array(inp, dtype = float)
    except Exception:
        errors.printError(inp, "Please input an iterable that is rectangular and consists of only numbers")
        raise
    else:
        x2 = Sigmoid(x)
        return np.round(x2 * (1 - x2), decimals = 10)

def TanhGradient(inp) -> np.ndarray:
    try:
        x = np.array(inp, dtype = float)
    except Exception:
        errors.printError(inp, "Please input an iterable that is rectangular and consists of only numbers")
        raise
    else:
        return np.round((1 - Tanh(x) ** 2), decimals = 10)

def ReLUGradient(inp) -> np.ndarray:
    try:
        x = np.array(inp, dtype = float)
    except Exception:
        errors.printError(inp, "Please input an iterable that is rectangular and consists of only numbers")
        raise
    else:
        return np.round(np.where(x > 0, 1, 0), decimals = 10)

def SoftmaxGradient(inp) -> np.ndarray:
    try:
        x = np.array(inp, dtype = float)
    except Exception:
        errors.printError(inp, "Please input an iterable that is rectangular and consists of only numbers")
        raise
    else:
        x2 = Softmax(x)
        return np.round(x2 * (1 - x2), decimals = 10)

activationGradients = { 
    "SIGMOID": SigmoidGradient, 
    "TANH": TanhGradient, 
    "RELU": ReLUGradient, 
    "SOFTMAX": SoftmaxGradient 
}