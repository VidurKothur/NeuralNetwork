import numpy as np
import errors

"""
These loss functions taken in the actual and predicted vectors as input and return a single loss value as output, while the loss gradients
will take in an actual and predicted vectors as input and return a vector of gradients with respect to each predicted value for backpropagation.

More loss functions can be added, please take in two np.ndarrays of the same length as input and return a single float value for the output
"""

def TotalSimpleError(actual, predicted) -> float:
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
            raise Exception("These loss functions cannot be called on datasets that have zero entries")
        return np.round(np.sum(actual - predicted), decimals = 10)

def MeanSimpleError(actual, predicted) -> float:
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
            raise Exception("These loss functions cannot be used on datasets that have zero entries")
        return np.round(np.mean(actual - predicted), decimals = 10)

def TotalSquaredError(actual, predicted) -> float:
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
            raise Exception("These loss functions cannot be used on datasets that have zero entries")
        x2 = actual - predicted
        return np.round(np.sum(x2 * x2), decimals = 10)

def MeanSquaredError(actual, predicted) -> float:
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
            raise Exception("These loss functions cannot be used on datasets that have zero entries")
        x2 = actual - predicted
        return np.round(np.mean(x2 * x2), decimals = 10)

def TotalAbsoluteError(actual, predicted) -> float:
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
            raise Exception("These loss functions cannot be used on datasets that have zero entries")
        return np.round(np.sum(np.abs(actual - predicted)), decimals = 10)

def MeanAbsoluteError(actual, predicted) -> float:
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
            raise Exception("These loss functions cannot be used on datasets that have zero entries")
        return np.round(np.mean(np.abs(actual - predicted)), decimals = 10)

def BinaryCrossEntropy(actual, predicted) -> float:
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
            raise Exception("These loss functions cannot be used on datasets that have zero entries")
        for val in np.nditer(predicted):
            if val <= 0 or val >= 1:
                errors.printError([actual, predicted], "Please make sure that all values in the dataset are between zero and one, exclusive")
                raise Exception("When using the Binary Cross Entropy Error, all values in the predicted dataset must be strictly between zero and one")
        return np.round(-np.mean(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted)), decimals = 10)

def CategoricalCrossEntropy(actual, predicted) -> float:
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
            raise Exception("These loss functions cannot be used on datasets that have zero entries")
        predicted = np.where(predicted == 0, np.e ** -100, predicted)
        for val in np.nditer(predicted):
            if val <= 0:
                errors.printError([actual, predicted], "Please make sure that all values in the predicted dataset are strictly greater than zero")
                raise Exception("When using the Categorical Cross Entropy Error, all values in the predicted dataset must be strictly greater than zero")
        return np.round(-np.mean(actual * np.log(predicted)), decimals = 10)

def KLDivergence(actual, predicted) -> float:
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
            raise Exception("These loss functions cannot be used on datasets that have zero entries")
    for val in np.nditer(predicted):
        if val <= 0 or val >= 1:
            errors.printError([actual, predicted], "Please make sure that every value in the predicted dataset is strictly between zero and one")
            raise Exception("When using the KL Divergence Error, all values in the predicted dataset must be strictly between zero and one like a true probability distribution")
    for val2 in np.nditer(actual):
        if val2 <= 0 or val2 >= 1:
            errors.printError([actual, predicted], "Please make sure that every value in the actual dataset is strictly between zero and one")
            raise Exception("When using the KL Divergence Error, all values in the actual dataset must be strictly between zero and one like a true probability distribution")
    return np.round(np.sum(actual * np.log(actual / predicted)), decimals = 10)

def HuberLoss(actual, predicted) -> float:
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
            raise Exception("These loss functions cannot be used on datasets that have zero entries")
    absErr = np.abs(actual - predicted)
    return np.round(np.mean(np.where(absErr <= 1.0, 0.5 * absErr ** 2, absErr - 0.5)), decimals = 10)

def CosineSimilarity(actual, predicted) -> float:
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
        if actual.size == 0 or predicted.size == 0:
            errors.printError([actual, predicted], "The actual and predicted arrays must have at least one element")
            raise Exception("These loss functions cannot be used on datasets that have zero entries")
    return np.round(np.sum(actual * predicted) / (np.linalg.norm(actual) * np.linalg.norm(predicted)), decimals = 10)

lossFunctions = { 
    "TOTALSIMPLEERROR": TotalSimpleError, 
    "MEANSIMPLEERROR": MeanSimpleError, 
    "TOTALSQUAREDERROR": TotalSquaredError, 
    "MEANSQUAREDERROR": MeanSquaredError, 
    "TOTALABSOLUTEERROR": TotalAbsoluteError, 
    "MEANABSOLUTEERROR": MeanAbsoluteError, 
    "BINARYCROSSENTROPY": BinaryCrossEntropy, 
    "CATEGORICALCROSSENTROPY": CategoricalCrossEntropy, 
    "KLDIVERGENCE": KLDivergence, 
    "HUBERLOSS": HuberLoss, 
    "COSINESIMILARITY": CosineSimilarity 
}

def TotalSimpleErrorGradient(actual, predicted) -> np.ndarray:
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
            raise Exception("These loss functions cannot be called on datasets that have zero entries")
        return np.round(-np.ones(actual.shape), decimals = 10)

def MeanSimpleErrorGradient(actual, predicted) -> np.ndarray:
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
            raise Exception("These loss functions cannot be called on datasets that have zero entries")
        return np.round(-np.ones(actual.shape) / actual.size, decimals = 10)

def TotalSquaredErrorGradient(actual, predicted) -> np.ndarray:
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
            raise Exception("These loss functions cannot be called on datasets that have zero entries")
        return np.round(-2 * (actual - predicted), decimals = 10)

def MeanSquaredErrorGradient(actual, predicted) -> np.ndarray:
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
            raise Exception("These loss functions cannot be called on datasets that have zero entries")
        return np.round(-2 * (actual - predicted) / actual.size, decimals = 10)

def TotalAbsoluteErrorGradient(actual, predicted) -> np.ndarray:
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
            raise Exception("These loss functions cannot be called on datasets that have zero entries")
        return np.round(-np.sign(actual - predicted), decimals = 10)

def MeanAbsoluteErrorGradient(actual, predicted) -> np.ndarray:
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
            raise Exception("These loss functions cannot be called on datasets that have zero entries")
        return np.round(-np.sign(actual, predicted) / actual.size, decimals = 10)

def BinaryCrossEntropyGradient(actual, predicted) -> np.ndarray:
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
            raise Exception("These loss functions cannot be used on datasets that have zero entries")
        for val in np.nditer(predicted):
            if val <= 0 or val >= 1:
                errors.printError([actual, predicted], "Please make sure that all values in the dataset are between zero and one, exclusive")
                raise Exception("When using the Binary Cross Entropy Error, all values in the predicted dataset must be strictly between zero and one")
        return np.round(-(actual / predicted - (1 - actual) / (1 - predicted)) / actual.size, decimals = 10)

def CategoricalCrossEntropyGradient(actual, predicted) -> np.ndarray:
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
            raise Exception("These loss functions cannot be used on datasets that have zero entries")
        predicted = np.where(predicted == 0, np.e ** -100, predicted)
        for val in np.nditer(predicted):
            if val <= 0:
                errors.printError([actual, predicted], "Please make sure that all values in the predicted dataset are strictly greater than zero")
                raise Exception("When using the Categorical Cross Entropy Error, all values in the predicted dataset must be strictly greater than zero")
        return np.round(-(actual / predicted) / actual.size, decimals = 10)

def KLDivergenceGradient(actual, predicted) -> np.ndarray:
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
            raise Exception("These loss functions cannot be used on datasets that have zero entries")
    for val in np.nditer(predicted):
        if val <= 0 or val >= 1:
            errors.printError([actual, predicted], "Please make sure that every value in the predicted dataset is strictly between zero and one")
            raise Exception("When using the KL Divergence Error, all values in the predicted dataset must be strictly between zero and one like a true probability distribution")
    for val2 in np.nditer(actual):
        if val2 <= 0 or val2 >= 1:
            errors.printError([actual, predicted], "Please make sure that every value in the actual dataset is strictly between zero and one")
            raise Exception("When using the KL Divergence Error, all values in the actual dataset must be strictly between zero and one like a true probability distribution")
    return np.round(-predicted / actual, decimals = 10)

def HuberLossGradient(actual, predicted) -> np.ndarray:
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
            raise Exception("These loss functions cannot be used on datasets that have zero entries")
    absErr = np.abs(actual - predicted)
    return np.round(np.where(absErr <= 1.0, predicted - actual, np.sign(predicted - actual)), decimals = 10)

def CosineSimilarityGradient(actual, predicted) -> np.ndarray:
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
        if actual.size == 0 or predicted.size == 0:
            errors.printError([actual, predicted], "The actual and predicted arrays must have at least one element")
            raise Exception("These loss functions cannot be used on datasets that have zero entries")
        return np.round((actual * np.linalg.norm(predicted) - CosineSimilarity(actual, predicted) * predicted) / (np.linalg.norm(predicted) ** 2), decimals = 10)

lossGradients = { 
    "TOTALSIMPLEERROR": TotalSimpleErrorGradient, 
    "MEANSIMPLEERROR": MeanSimpleErrorGradient, 
    "TOTALSQUAREDERROR": TotalSquaredErrorGradient, 
    "MEANSQUAREDERROR": MeanSquaredErrorGradient, 
    "TOTALABSOLUTEERROR": TotalAbsoluteErrorGradient, 
    "MEANABSOLUTEERROR": MeanAbsoluteErrorGradient, 
    "BINARYCROSSENTROPY": BinaryCrossEntropyGradient, 
    "CATEGORICALCROSSENTROPY": CategoricalCrossEntropyGradient, 
    "KLDIVERGENCE": KLDivergenceGradient, 
    "HUBERLOSS": HuberLossGradient, 
    "COSINESIMILARITY": CosineSimilarityGradient 
}