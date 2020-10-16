from typing import Dict, Type, cast, Callable, Any

import numpy as np
import sklearn.metrics

from .tasks import Task
from .models import Oracle
from .util import classifications

# TODO This is very inneficient as it gets models to do predictions several
# times, should really make the results cached

# TODO Add oracle names to configurations and do the whole register
# things as well to make configurations easier, this will also
# entail some checking to make sure that the oracle.kind() matches up with
# the actual kind of task being performed

# TODO This should really just let you pass an ensemble or selector with
# some data rather than trying to make it super high level and must include
# an oracle
Metric = Callable[[np.ndarray, np.ndarray], Any]

def score(
    task: Task,
    metric: Metric,
    oracle_cls: Type[Oracle]
) -> Dict[str, Any]:

    metric_results: Dict[str, Any] = {}
    for key in task.key_iterator():
        data = task.dataset(key)
        X, y = data['X_test'], data['y_test']

        results: Dict[str, Any] = {}

        # Ensemble results
        ensemble = task.ensembles.get(key, None)
        if ensemble is None:
            raise ValueError('No ensemble found for task {task.task.id=}.'
                             + 'Please run the task first')

        ensemble_predictions = ensemble.predict(X)
        results['ensemble'] = metric(y, ensemble_predictions)

        # Inidividual model predictions
        model_predictions = ensemble.model_predictions(X)
        results['models'] = [metric(y, preds) for preds in model_predictions]

        # Selector results
        if task.has_selector():
            selector = task.selectors.get(key, None)
            if selector is None:
                raise ValueError('No selector found for task {task.task.id=}.'
                                 + 'Please run the task first')

            selector_predictions = selector.predict(X)
            results['selector'] = metric(y, selector_predictions)

        oracle = oracle_cls(ensemble, y)

        oracle_predictions = oracle.predict(X)
        results['oracle'] = metric(y, oracle_predictions)

        keyname = task.name(key)
        metric_results[keyname] = results

    # If there was only 1 key, flatten it out to take out the key
    if len(metric_results) == 1:
        metric_results = dict(next(iter(metric_results.values())))
    return metric_results


def selection_accuracies(
    task: Task,
    oracle_cls: Type[Oracle],
) -> Dict[str, Any]:
    assert task.has_selector(), 'Task must have selector enabled'

    selection_results: Dict[str, Any] = {}
    for key in task.key_iterator():
        results: Dict[str, Any] = {}

        data = task.dataset(key)
        X, y = data['X_test'], data['y_test']

        ensemble = task.ensembles.get(key, None)
        if ensemble is None:
            raise ValueError('No ensemble found for task {task.task.id=}.'
                             + 'Please run the task first')
        selector = task.selectors.get(key, None)
        if selector is None:
            raise ValueError('No selector found for task {task.task.id=}.'
                             + 'Please run the task first')

        oracle = oracle_cls(ensemble, y)

        # Individual Model Accuracies
        # this is (n, k) as there could be multiple correct
        # choices
        n_models = len(ensemble.models())
        selections = selector.selections(X)
        correct_choices = oracle.correct_selections(X)

        correct_selection_tally = [0] * n_models
        for choices, selection in zip(correct_choices, selections):
            # choices[selection] is 1 if selected correctly, 0 otherwise
            correct_selection_tally[selection] += choices[selection]

        # How many times each model was chosen
        selection_tally, _ = np.histogram(selections, bins=range(n_models + 1))
        accuracies = [
            n_correct / n_selected if n_selected != 0 else 0
            for n_correct, n_selected
            in zip(correct_selection_tally, selection_tally)
        ]

        results['selection_counts'] = selection_tally
        results['selection_accuracies'] = accuracies

        keyname = task.name(key)
        selection_results[keyname] = results

    # If there was only 1 key, flatten it out to take out the key
    if len(selection_results) == 1:
        selection_results = dict(next(iter(selection_results.values())))

    return selection_results


def accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    predicted_classes = classifications(y_pred)
    acc = sklearn.metrics.accuracy_score(y_true, predicted_classes)
    return cast(float, acc)  # Pretty sure this is safe by their docs


def mean_abs_error(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)
    return cast(float, mae)  # Pretty sure this is safe by their docs
