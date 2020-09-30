import numpy as np


def binary_classifications(predictions: np.ndarray) -> np.ndarray:
    assert predictions.shape[1] == 2, \
        f'Predictions are not binary, {predictions.shape=}'

    return (predictions[:, 1] > 0.5).astype(int)

def classifications(predictions: np.ndarray) -> np.ndarray:
    return np.argmax(predictions, axis=1)

def multilabel_classifications(predictions: np.ndarray) -> np.ndarray:
    return (predictions > 0.5).astype(int)


def instance_wise(predictions: np.ndarray) -> np.ndarray:
    """
    Converts predictions to be instance wise ordered as opposed
    to model ordered.
    Done by inverting the 1st axis (models) and 2nd axis (instances)
    """
    axes = (1, 0, *range(2, predictions.ndim))
    return np.transpose(predictions, axes=axes)
