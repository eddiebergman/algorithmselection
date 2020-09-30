from typing import Dict, Any

# TODO make this import nicer
from ..tasks.task import Task
from .model import Metric
from ..model import ClassificationOracle


def measure_metric(task: Task, metric: Metric) -> Dict[str, Any]:

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

        # Oracle results
        # TODO make these a Literal Type
        oracle = None
        if task.task_type == 'classification':
            oracle = ClassificationOracle(ensemble, y)
        else:
            raise NotImplementedError(f'No oracle found for {task.task_type=}')

        oracle_predictions = oracle.predict(X)
        results['oracle'] = metric(y, oracle_predictions)

        keyname = task.name(key)
        metric_results[keyname] = results

    return metric_results
