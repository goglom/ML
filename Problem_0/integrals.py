import numpy as np


def _trapeze_square(func_values, points):
    diff = points[1] - points[0]
    if diff < 0:
        raise RuntimeWarning("left bound greater than right bond")
    return (func_values[0] + func_values[1]) * diff / 2


def integral_trapeze(func: np.array, steps: np.array) -> float:
    result = 0.0
    for i in range(len(func) - 1):
        result += _trapeze_square(func[i: i + 2], steps[i: i + 2])
    return result

