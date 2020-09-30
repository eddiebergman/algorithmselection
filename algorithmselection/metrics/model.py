from typing import cast, Callable, Any

import numpy as np
import sklearn.metrics

Metric = Callable[[np.ndarray, np.ndarray], Any]


def accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    acc = sklearn.metrics.accuracy_score(y_true, y_pred)
    return cast(float, acc)  # Pretty sure this is safe by their docs


def mean_abs_error(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)
    return cast(float, mae)  # Pretty sure this is safe by their docs
