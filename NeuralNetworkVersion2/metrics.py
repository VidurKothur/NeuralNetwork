import numpy as np
import errors

"""
These metric functions take in a complete actual and predicted arrays as input and return a single float value that captures the metric desired.

More metrics functions can be added, please make sure to update the dictionaries at the bottom and provide a display name because the program will
treat all metric names as all caps.  Also make sure to input two numpy arrays of the same length, and return a float.
"""

def Accuracy(actual, predicted) -> float:
    try:
        actual = np.array(actual, dtype = float)
        predicted = np.array(predicted, dtype = float)
    except Exception:
        errors.printError([actual, predicted], "Please input an iterable that is rectangular and consists of only numbers")
        raise
    else:
        if not (actual.shape == predicted.shape):
            errors.printError([actual.shape, predicted.shape], "The actual and predicted arrays must be of the same shape")
            raise Exception("The actual and predicted arrays must be of the same shape")
        if actual.size == 0:
            errors.printError([actual, predicted], "The actual and predicted arrays must have at least one element")
            raise Exception("These metrics cannot be called on datasets that have zero entries")
        if not actual.ndim == 2:
            errors.printError([actual, predicted], "Please provide an array of vectors, nothing else")
            raise Exception("The actual and predicted datasets must be stacks of vectors, 2 dimensional, nothing else")
        correct = 0
        total = 0
        for a in range(len(actual)):
            if np.argmax(actual[a]) == np.argmax(predicted[a]):
                correct += 1
            total += 1
        if total == 0:
            return 0.0
        return np.round(correct / total, decimals = 10)
    
def Precision(actual, predicted) -> float:
    try:
        actual = np.array(actual, dtype = float)
        predicted = np.array(predicted, dtype = float)
    except Exception:
        errors.printError([actual, predicted], "Please input an iterable that is rectangular and consists of only numbers")
        raise
    else:
        if not (actual.shape == predicted.shape):
            errors.printError([actual, predicted], "The actual and predicted arrays must be of the same shape")
            raise Exception("The actual and predicted arrays must be of the same shape")
        if actual.size == 0:
            errors.printError([actual, predicted], "The actual and predicted arrays must have at least one element")
            raise Exception("These metrics cannot be called on datasets that have zero entries")
        if not actual.ndim == 2:
            errors.printError([actual, predicted], "Please provide an array of vectors, nothing else")
            raise Exception("The actual and predicted datasets must be stacks of vectors, 2 dimensional, nothing else")
        tp, fp, tn, fn = 0, 0, 0, 0
        for a in range(len(actual)):
            for b in range(len(actual[0])):
                act = np.argmax(actual[a])
                pred = np.argmax(predicted[a])
                if pred == b and act == b:
                    tp += 1
                elif pred == b and act != b:
                    fp += 1
                elif pred != b and act != b:
                    tn += 1
                elif pred != b and act == b:
                    fn += 1
        if tp + fp == 0:
            return 0.0
        return np.round((tp) / (tp + fp), decimals = 10)

def Recall(actual, predicted) -> float:
    try:
        actual = np.array(actual, dtype = float)
        predicted = np.array(predicted, dtype = float)
    except Exception:
        errors.printError([actual, predicted], "Please input an iterable that is rectangular and consists of only numbers")
        raise
    else:
        if not (actual.shape == predicted.shape):
            errors.printError([actual, predicted], "The actual and predicted arrays must be of the same shape")
            raise Exception("The actual and predicted arrays must be of the same shape")
        if actual.size == 0:
            errors.printError([actual, predicted], "The actual and predicted arrays must have at least one element")
            raise Exception("These metrics cannot be called on datasets that have zero entries")
        if not actual.ndim == 2:
            errors.printError([actual, predicted], "Please provide an array of vectors, nothing else")
            raise Exception("The actual and predicted datasets must be stacks of vectors, 2 dimensional, nothing else")
        tp, fp, tn, fn = 0, 0, 0, 0
        for a in range(len(actual)):
            for b in range(len(actual[0])):
                act = np.argmax(actual[a])
                pred = np.argmax(predicted[a])
                if pred == b and act == b:
                    tp += 1
                elif pred == b and act != b:
                    fp += 1
                elif pred != b and act != b:
                    tn += 1
                elif pred != b and act == b:
                    fn += 1
        if tp + fn == 0:
            return 0.0
        return np.round((tp) / (tp + fn), decimals = 10)

def F1Score(actual, predicted) -> float:
    try:
        actual = np.array(actual, dtype = float)
        predicted = np.array(predicted, dtype = float)
    except Exception:
        errors.printError([actual, predicted], "Please input an iterable that is rectangular and consists of only numbers")
        raise
    else:
        if not (actual.shape == predicted.shape):
            errors.printError([actual, predicted], "The actual and predicted arrays must be of the same shape")
            raise Exception("The actual and predicted arrays must be of the same shape")
        if actual.size == 0:
            errors.printError([actual, predicted], "The actual and predicted arrays must have at least one element")
            raise Exception("These metrics cannot be called on datasets that have zero entries")
        if not actual.ndim == 2:
            errors.printError([actual, predicted], "Please provide an array of vectors, nothing else")
            raise Exception("The actual and predicted datasets must be stacks of vectors, 2 dimensional, nothing else")
        tp, fp, tn, fn = 0, 0, 0, 0
        for a in range(len(actual)):
            for b in range(len(actual[0])):
                act = np.argmax(actual[a])
                pred = np.argmax(predicted[a])
                if pred == b and act == b:
                    tp += 1
                elif pred == b and act != b:
                    fp += 1
                elif pred != b and act != b:
                    tn += 1
                elif pred != b and act == b:
                    fn += 1
        precision = (tp) / (tp + fp)
        recall = (tp) / (tp + fn)
        if precision + recall == 0:
            return 0.0
        return np.round((2 * precision * recall) / (precision + recall), decimals = 10)

def Specificity(actual, predicted) -> float:
    try:
        actual = np.array(actual, dtype = float)
        predicted = np.array(predicted, dtype = float)
    except Exception:
        errors.printError([actual, predicted], "Please input an iterable that is rectangular and consists of only numbers")
        raise
    else:
        if not (actual.shape == predicted.shape):
            errors.printError([actual, predicted], "The actual and predicted arrays must be of the same shape")
            raise Exception("The actual and predicted arrays must be of the same shape")
        if actual.size == 0:
            errors.printError([actual, predicted], "The actual and predicted arrays must have at least one element")
            raise Exception("These metrics cannot be called on datasets that have zero entries")
        if not actual.ndim == 2:
            errors.printError([actual, predicted], "Please provide an array of vectors, nothing else")
            raise Exception("The actual and predicted datasets must be stacks of vectors, 2 dimensional, nothing else")
        tp, fp, tn, fn = 0, 0, 0, 0
        for a in range(len(actual)):
            for b in range(len(actual[0])):
                act = np.argmax(actual[a])
                pred = np.argmax(predicted[a])
                if pred == b and act == b:
                    tp += 1
                elif pred == b and act != b:
                    fp += 1
                elif pred != b and act != b:
                    tn += 1
                elif pred != b and act == b:
                    fn += 1
        if fp + tn == 0:
            return 0.0
        return np.round((tn) / (fp + tn), decimals = 10)

def MSE(actual, predicted) -> float:
    try:
        actual = np.array(actual, dtype = float)
        predicted = np.array(predicted, dtype = float)
    except Exception:
        errors.printError([actual, predicted], "Please input an iterable that is rectangular and consists of only numbers")
        raise
    else:
        if not (actual.shape == predicted.shape):
            errors.printError([actual, predicted], "The actual and predicted arrays must be of the same shape")
            raise Exception("The actual and predicted arrays must be of the same shape")
        if actual.size == 0:
            errors.printError([actual, predicted], "The actual and predicted arrays must have at least one element")
            raise Exception("These metrics cannot be called on datasets that have zero entries")
        if not actual.ndim == 2:
            errors.printError([actual, predicted], "Please provide an array of vectors, nothing else")
            raise Exception("The actual and predicted datasets must be stacks of vectors, 2 dimensional, nothing else")
        totalLoss = 0
        totalNum = 0
        actual = actual.flatten()
        predicted = predicted.flatten()
        for a in range(actual.size):
            totalNum += 1
            totalLoss += (actual[a] - predicted[a]) ** 2
        return np.round(totalLoss / totalNum, decimals = 10)

def MAE(actual, predicted) -> float:
    try:
        actual = np.array(actual, dtype = float)
        predicted = np.array(predicted, dtype = float)
    except Exception:
        errors.printError([actual, predicted], "Please input an iterable that is rectangular and consists of only numbers")
        raise
    else:
        if not (actual.shape == predicted.shape):
            errors.printError([actual, predicted], "The actual and predicted arrays must be of the same shape")
            raise Exception("The actual and predicted arrays must be of the same shape")
        if actual.size == 0:
            errors.printError([actual, predicted], "The actual and predicted arrays must have at least one element")
            raise Exception("These metrics cannot be called on datasets that have zero entries")
        if not actual.ndim == 2:
            errors.printError([actual, predicted], "Please provide an array of vectors, nothing else")
            raise Exception("The actual and predicted datasets must be stacks of vectors, 2 dimensional, nothing else")
        totalLoss = 0
        totalNum = 0
        actual = actual.flatten()
        predicted = predicted.flatten()
        for a in range(actual.size):
            totalNum += 1
            totalLoss += abs(actual[a] - predicted[a])
        return np.round(totalLoss / totalNum, decimals = 10)

def RMSE(actual, predicted) -> float:
    try:
        actual = np.array(actual, dtype = float)
        predicted = np.array(predicted, dtype = float)
    except Exception:
        errors.printError([actual, predicted], "Please input an iterable that is rectangular and consists of only numbers")
        raise
    else:
        if not (actual.shape == predicted.shape):
            errors.printError([actual, predicted], "The actual and predicted arrays must be of the same shape")
            raise Exception("The actual and predicted arrays must be of the same shape")
        if actual.size == 0:
            errors.printError([actual, predicted], "The actual and predicted arrays must have at least one element")
            raise Exception("These metrics cannot be called on datasets that have zero entries")
        if not actual.ndim == 2:
            errors.printError([actual, predicted], "Please provide an array of vectors, nothing else")
            raise Exception("The actual and predicted datasets must be stacks of vectors, 2 dimensional, nothing else")
        totalLoss = 0
        totalNum = 0
        actual = actual.flatten()
        predicted = predicted.flatten()
        for a in range(actual.size):
            totalNum += 1
            totalLoss += (actual[a] - predicted[a]) ** 2
        return np.round(np.sqrt(totalLoss / totalNum), decimals = 10)

def R2(actual, predicted) -> float:
    try:
        actual = np.array(actual, dtype = float)
        predicted = np.array(predicted, dtype = float)
    except Exception:
        errors.printError([actual, predicted], "Please input an iterable that is rectangular and consists of only numbers")
        raise
    else:
        if not (actual.shape == predicted.shape):
            errors.printError([actual, predicted], "The actual and predicted arrays must be of the same shape")
            raise Exception("The actual and predicted arrays must be of the same shape")
        if actual.size == 0:
            errors.printError([actual, predicted], "The actual and predicted arrays must have at least one element")
            raise Exception("These metrics cannot be called on datasets that have zero entries")
        if not actual.ndim == 2:
            errors.printError([actual, predicted], "Please provide an array of vectors, nothing else")
            raise Exception("The actual and predicted datasets must be stacks of vectors, 2 dimensional, nothing else")
        totalNumer = 0
        totalDenom = 0
        actual = actual.flatten()
        predicted = predicted.flatten()
        mean = np.mean(actual)
        for a in range(len(actual)):
            totalNumer += (actual[a] - predicted[a]) ** 2
            totalDenom += (actual[a] - mean) ** 2
        return np.round(1 - totalNumer / totalDenom, decimals = 10)

metrics = {
    "ACCURACY": Accuracy,
    "PRECISION": Precision,
    "RECALL": Recall,
    "F1SCORE": F1Score,
    "SPECIFICITY": Specificity,
    "MSE": MSE,
    "MAE": MAE,
    "RMSE": RMSE,
    "R2": R2
}

metricDisplays = {
    "ACCURACY": "Accuracy:",
    "PRECISION": "Precision:",
    "RECALL": "Recall:",
    "F1SCORE": "F1Score:",
    "SPECIFICITY": "Specificity:",
    "MSE": "Mean Squared Error:",
    "MAE": "Mean Absolute Error:",
    "RMSE": "Root Mean Squared Error:",
    "R2": "R Squared:"
}