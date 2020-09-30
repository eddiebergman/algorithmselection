from functools import wraps
from typing import Callable, Union, Dict

import numpy as np

from .ensemble import Ensemble
from ..util import instance_wise

EnsembleEvaluator = Callable[
    [Ensemble, np.ndarray, Union[np.ndarray, None]], np.ndarray
]
ensemble_evaluators: Dict[str, EnsembleEvaluator] = {}

def register_ensemble_evaluator(
    func: EnsembleEvaluator
) -> EnsembleEvaluator:
    """
    Type checks for type annotation

    Ensures that X and the returned results have the same length for training

    Registers the function to be used in the experiments
    """
    @wraps(func)
    def wrapped_func(ensemble: Ensemble, X: np.ndarray, y: np.ndarray):
        results = func(ensemble, X, y)
        if len(y) != len(results):
            raise RuntimeError(f'{len(y)} must match {len(results)}')
        return results

    ensemble_evaluators[func.__name__] = func
    return wrapped_func

@register_ensemble_evaluator
def correct_classifications(
    ensemble: Ensemble,
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """
    Returns a list binary vectors indicating whether the model
    correctly predicted the class or not for each instance.

    model_predictions = [[0.3, 0.7], [0.2, 0.8], [0.9, 0.1]]
    y = [1, 0, 0]
    return [1, 0, 1]

    Params
    ======
    ensemble | len (k)

    X | shape (n, m)

    y | binary, shape (n)

    Returns
    =======
    np.ndarray | binary, shape (n, k)
    """
    assert ensemble.kind() == 'classifier', \
        '`correct_classifcations` only works for classifier ensembles'

    all_model_probabilities = ensemble.model_predictions(X)
    all_model_class_predictions = np.asarray([
        np.argmax(model_probabilities, axis=1)
        for model_probabilities in all_model_probabilities
    ])
    all_model_correct_vectors = np.asarray([
        (model_class_predictions == y).astype(int)
        for model_class_predictions in all_model_class_predictions
    ])
    return instance_wise(all_model_correct_vectors)

@register_ensemble_evaluator
def prediciton_error(
    ensemble: Ensemble,
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """
    Params
    ======
    ensemble | len (k)

    X | shape (n, m)

    y | shape (n)

    Returns
    =======
    np.ndarray | shape (n, kelect
    """
    assert ensemble.kind() == 'regressor', \
        '`prediction_differences` only works for regressor ensembles'

    model_predictions = ensemble.model_predictions(X)
    ensemble_prediction_differences = np.asarray([
        np.abs(model_prediction_vec - y)
        for model_prediction_vec in model_predictions
    ])
    return instance_wise(ensemble_prediction_differences)

@register_ensemble_evaluator
def correct_class_probability_error(
    ensemble: Ensemble,
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """
    Assumed that the predicitons given back by the ensemble models
    are for a binary classification task
    i.e. [0.3, 0.7] to indicate 0.3 probability of class 0 and 0.7 of class 1

    Params
    ======
    ensemble | len (k)

    X | shape (n, m)

    y | binary, shape (n)

    Returns
    =======
    np.ndarray | in [0,1], shape (n)

    """
    assert ensemble.kind() == 'classifier', \
        'correct_class_probability_error only works for classifier ensembles'
    model_probabilities = ensemble.model_predictions(X)
    ensemble_probability_distances = np.asarray([
        np.asarray([
            1 - class_probabilities[cls]
            for class_probabilities, cls in zip(model_probability_vec, y)
        ])
        for model_probability_vec in model_probabilities
    ])
    return instance_wise(ensemble_probability_distances)

# TODO
# Fix this, at least by name
@register_ensemble_evaluator
def multiclass_probability_errors(
    ensemble: Ensemble,
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """
    Params
    ======
    ensemble | len (k)

    X | shape (n, m)

    y | shape (n, c)

    Returns
    =======
    np.ndarray | shape (n, k, c)
    """
    assert ensemble.kind() == 'classifier', \
        'multiclass_probability_difference only works for classifier ensembles'

    model_probabilities = ensemble.model_predictions(X)
    probability_differences = np.asarray([
        np.abs(model_probability_vec - class_labels)
        for model_probability_vec, class_labels in zip(model_probabilities, y)
    ])
    return instance_wise(probability_differences)
