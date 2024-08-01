import numpy as np
import errors

"""
These initialization functions will take in a number that corresponds to the number of neurons entering the current layer, and will
output an initialized weight matrix which will transform that vector into a new vector with the current layer's number of neurons.
Biases are initialized to zero by default.

More initialization functions can be added, please make sure to input two integers and return an np.ndarray that is a i x o matrix,
where i is the input number and o is the output number.  Also, please update the dictionary with an ALL CAPS name and a reference to
the function.
"""

def Zero(inNum: int, outNum: int) -> np.ndarray:
    try:
        if not (isinstance(inNum, int) and isinstance(outNum, int) and inNum > 0 and outNum > 0):
            raise Exception("The input and output numbers for weight initialization must be integers greater than zero")
    except Exception:
        errors.printError([inNum, outNum], "Please enter input and output numbers that are positive integers")
        raise
    return np.round(np.zeros((inNum, outNum)), decimals = 10)

def XavierNormal(inNum: int, outNum: int) -> np.ndarray:
    try:
        if not (isinstance(inNum, int) and isinstance(outNum, int) and inNum > 0 and outNum > 0):
            raise Exception("The input and output numbers for weight initialization must be integers greater than zero")
    except Exception:
        errors.printError([inNum, outNum], "Please enter input and output numbers that are positive integers")
        raise
    return np.round(np.random.normal(0, np.sqrt(2 / (inNum + outNum)), size = (inNum, outNum)), decimals = 10)

def XavierUniform(inNum: int, outNum: int) -> np.ndarray:
    try:
        if not (isinstance(inNum, int) and isinstance(outNum, int) and inNum > 0 and outNum > 0):
            raise Exception("The input and output numbers for weight initialization must be integers greater than zero")
    except Exception:
        errors.printError([inNum, outNum], "Please enter input and output numbers that are positive integers")
        raise
    bounds = np.sqrt(6 / (inNum + outNum))
    return np.round(np.random.uniform(-bounds, bounds, size = (inNum, outNum)), decimals = 10)

def HeNormal(inNum: int, outNum: int) -> np.ndarray:
    try:
        if not (isinstance(inNum, int) and isinstance(outNum, int) and inNum > 0 and outNum > 0):
            raise Exception("The input and output numbers for weight initialization must be integers greater than zero")
    except Exception:
        errors.printError([inNum, outNum], "Please enter input and output numbers that are positive integers")
        raise
    return np.round(np.random.normal(0, np.sqrt(2 / inNum), size = (inNum, outNum)), decimals = 10)

def HeUniform(inNum: int, outNum: int) -> np.ndarray:
    try:
        if not (isinstance(inNum, int) and isinstance(outNum, int) and inNum > 0 and outNum > 0):
            raise Exception("The input and output numbers for weight initialization must be integers greater than zero")
    except Exception:
        errors.printError([inNum, outNum], "Please enter input and output numbers that are positive integers")
        raise
    bounds = np.sqrt(6 / inNum)
    return np.round(np.random.uniform(-bounds, bounds, size = (inNum, outNum)), decimals = 10)

def LecunNormal(inNum: int, outNum: int) -> np.ndarray:
    try:
        if not (isinstance(inNum, int) and isinstance(outNum, int) and inNum > 0 and outNum > 0):
            raise Exception("The input and output numbers for weight initialization must be integers greater than zero")
    except Exception:
        errors.printError([inNum, outNum], "Please enter input and output numbers that are positive integers")
        raise
    return np.round(np.random.normal(0, np.sqrt(1 / inNum), size = (inNum, outNum)), decimals = 10)

def LecunUniform(inNum: int, outNum: int) -> np.ndarray:
    try:
        if not (isinstance(inNum, int) and isinstance(outNum, int) and inNum > 0 and outNum > 0):
            raise Exception("The input and output numbers for weight initialization must be integers greater than zero")
    except Exception:
        errors.printError([inNum, outNum], "Please enter input and output numbers that are positive integers")
        raise
    bounds = np.sqrt(3 / inNum)
    return np.round(np.random.uniform(-bounds, bounds, size = (inNum, outNum)), decimals = 10)

def RandomNormal(inNum: int, outNum: int) -> np.ndarray:
    try:
        if not (isinstance(inNum, int) and isinstance(outNum, int) and inNum > 0 and outNum > 0):
            raise Exception("The input and output numbers for weight initialization must be integers greater than zero")
    except Exception:
        errors.printError([inNum, outNum], "Please enter input and output numbers that are positive integers")
        raise
    return np.round(np.random.normal(0, 1, size = (inNum, outNum)), decimals = 10)

def RandomUniform(inNum: int, outNum: int) -> np.ndarray:
    try:
        if not (isinstance(inNum, int) and isinstance(outNum, int) and inNum > 0 and outNum > 0):
            raise Exception("The input and output numbers for weight initialization must be integers greater than zero")
    except Exception:
        errors.printError([inNum, outNum], "Please enter input and output numbers that are positive integers")
        raise
    return np.round(np.random.uniform(-1, 1, size = (inNum, outNum)), decimals = 10)

initializationFunctions = { 
    "ZERO": Zero, 
    "XAVIERNORMAL": XavierNormal, 
    "XAVIERUNIFORM": XavierUniform, 
    "HENORMAL": HeNormal, 
    "HEUNIFORM": HeUniform, 
    "LECUNNORMAL": LecunNormal, 
    "LECUNUNIFORM": LecunUniform, 
    "RANDOMNORMAL": RandomNormal, 
    "RANDOMUNIFORM": RandomUniform 
}