import numpy as np

"""
These optimizer functions will take in the parameter list given below as input (anything labeled None will be populated in the code) and
return the updated parameter, along with an array of data needed for the next iteration of the optimizer.

More optimizer functions can be added, please update the two dictionaries below with an ALL CAPS name referring to the function as well as
an input parameter list with the following format:

[[Theta, Loss Gradient], [Any static parameters provided by user], [Any dynamic parameters needed to be passed between calls]]

Above is the array format that the function will take in as input when called, and the output will be of the following format:

New Theta, [Any dynamic parameters needed to be passed between calls]

The input and output dynamic parameter list should be of the same length.  Also, please provide default values for the static parameters 
and initialize the dynamic parameters.
"""

def NOne(args):
    return args[0][0], []

def SGD(args):
    return args[0][0] - args[1][0] * args[0][1], []

def SGDMomentum(args):
    if args[1][0] < 0 or args[1][0] > 1:
        raise Exception("The parameter beta provided to the SGD Momentum optimizer must be between 0 and 1, inclusive")
    v = args[1][0] * args[2][0] + (1 - args[1][0]) * args[0][1]
    return args[0][0] - args[1][1] * v, [v]

def NAG(args):
    if args[1][0] < 0 or args[1][0] > 1:
        raise Exception("The parameter beta provided to the NAG optimizer must be between 0 and 1, inclusive")
    v = args[1][0] * args[2][0] - args[1][1] * args[0][1]
    return args[0][0] - args[1][0] * args[2][0], [v]

def RMSProp(args):
    if args[1][0] < 0 or args[1][0] > 1:
        raise Exception("The parameter beta provided to the RMSProp optimizer must be between 0 and 1, inclusive")
    g = args[1][0] * args[2][0] + (1 - args[1][0]) * (args[0][1] ** 2)
    return args[0][0] - args[1][1] * args[0][1] / np.sqrt(g + 1e-8), [g]

def Adam(args):
    if args[1][0] < 0 or args[1][0] > 1:
        raise Exception("The parameter beta1 provided to the Adam optimizer must be between 0 and 1, inclusive")
    if args[1][0] < 0 or args[1][0] > 1:
        raise Exception("The parameter beta2 provided to the Adam optimizer must be between 0 and 1, inclusive")
    v = args[1][0] * args[2][0] + (1 - args[1][0]) * args[0][1]
    g = args[1][1] * args[2][1] + (1 - args[1][1]) * (args[0][1] ** 2)
    return args[0][0] - args[1][2] * (v / (1 - args[1][0] ** args[2][2])) / (np.sqrt(g / (1 - args[1][1] ** args[2][2])) + 1e-8), [v, g, args[2][2] + 1]

def BFGS(args):
    y = np.array(args[0][1] - args[2][1])
    s = np.array(args[0][0] - args[2][0])
    B = args[2][2] + (y * y) / (y.T @ s) - (((args[2][2] @ s) @ s.T) @ args[2][2]) / ((s.T @ args[2][2]) @ s)
    return args[0][0] - args[1][0] * (B @ args[0][1]), [args[0][0], args[0][1], B]

optimizers = {
    "NONE": NOne,
    "SGD": SGD, 
    "SGDMOMENTUM": SGDMomentum, 
    "NAG": NAG, 
    "RMSPROP": RMSProp, 
    "ADAM": Adam, 
    "BFGS": BFGS 
}

optimizerParams = { 
    "NONE": [[None, None], [], []], #[[Theta, Loss Gradient], [], []]
    "SGD": [[None, None], [1e-4], []], #[[Theta, Loss Gradient], [Learning Rate], []]
    "SGDMOMENTUM": [[None, None], [0.9, 1e-4], [0]], #[[Theta, Loss Gradient], [Velocity Constant, Learning Rate], [Previous Velocity]]
    "NAG": [[None, None], [0.9, 1e-4], [0]], #[[Theta, Loss Gradient], [Velocity Constant, Learning Rate], [Previous Velocity]]
    "RMSPROP": [[None, None], [0.999, 1e-4], [0]], #[[Theta, Loss Gradient], [RMS Constant, Learning Rate], [Previous RMS]]
    "ADAM": [[None, None], [0.9, 0.999, 1e-4], [0, 0, 1]], #[[Theta, Loss Gradient], [First Moment Constant, Second Moment Constant, Learning Rate], [Previous First Moment, Previous Second Moment, Time]]
    "BFGS": [[None, None], [1e-4], [0, 0, 0]] #[[Theta, Loss Gradient], [Learning Rate], [Previous Theta, Previous Loss Gradient, Approximate Hessian Matrix]]
}

optimizerTemplates = {
    "NONE": '["None"]',
    "SGD": '["SGD", Learning Rate: float]',
    "SGDMOMENTUM": '["SGDMomentum", Velocity Constant: 0 < float < 1, Learning Rate: float]',
    "NAG": '["NAG", Velocity Constant: 0 < float < 1, Learning Rate: float]',
    "RMSPROP": '["RMSProp", RMS Constant: 0 < float < 1, Learning Rate: float]',
    "ADAM": '["Adam", First Moment Constant: 0 < float < 1, Second Moment Constant: 0 < float < 1: Learning Rate: float]',
    "BFGS": '["BFGS", Learning Rate: float]'
}