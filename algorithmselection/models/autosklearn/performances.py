import time
from typing import Dict, Any, Tuple, List

import pandas as pd
from smac.tae import StatusType

from .base import AutoSklearnModel

def run_history_data(model: AutoSklearnModel):
    return model.autosklearn_model().automl_.runhistory_.data

def models_performances(model: AutoSklearnModel) -> List[Dict[str, Any]]:
    model_performances = []
    data = run_history_data(model)

    succesful_runs = {
        key: run for key, run in data.items()
        if run.status == StatusType.SUCCESS
    }

    metric = model.autosklearn_model().automl_._metric
    optimum = metric._optimum
    sign = metric._sign

    for key, run in succesful_runs.items():
        cost = run.cost
        train_loss = run.additional_info['train_loss']

        endtime = pd.Timestamp(time.strftime(
            '%Y-%m-%d %H:%M:%S', time.localtime(run.endtime)
        ))
        performance = {
            'time': endtime,
            'optimization_score': optimum - (sign * cost),
            'train_score': optimum - (sign * train_loss)
        }

        if 'test_loss' in run.additional_info:
            test_loss = run.additional_info['test_loss']
            performance['test_score'] = optimum - (sign * test_loss)

        model_performances.append(performance)

    return model_performances

def _overall_performance(model: AutoSklearnModel) -> List[Dict[str, Any]]:
    """
    [{
        'time' :
        'optimzation_score':
        'test_score':
    }]
    """
    performances : List[Dict[str, Any]] = \
        model.autosklearn_model().automl_.ensemble_performance_history

    # rename them to be consistent
    for perf in performances:
        perf['time'] = perf.pop('Timestamp')
        perf['optimization_score'] = perf.pop('ensemble_optimization_score')
        perf['test_score'] = perf.pop('ensemble_test_score')
    return performances

ensemble_performance = _overall_performance
selector_performance = _overall_performance


def performances(
    model: AutoSklearnModel
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    overall_performances = _overall_performance(model)
    model_performances = models_performances(model)
    return overall_performances, model_performances
