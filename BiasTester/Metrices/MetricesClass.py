import numpy as np
import math


def cossim(x, y):
    return np.dot(x, y) / math.sqrt(np.dot(x, x) * np.dot(y, y))


def softmax(arr, axis=1):
    e = np.exp(arr)
    return e / e.sum(axis=axis, keepdims=True)