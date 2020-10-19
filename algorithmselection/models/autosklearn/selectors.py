import numpy as np
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor
from autosklearn.metrics import make_scorer
from dask.distributed import Client

from ..selector import Selector, register_selector
from ..ensemble import Ensemble
from ..model import ModelType
from .base import AutoSklearnModel

#TODO clean this up and make a configurable option
#   Currently it is being passed in through the config
def log_correct(y, pred):
    single = np.any(y == pred).astype(int)
    log_extra = np.log(np.sum(y == pred))
    normalizer = 1 + np.log(pred.size)
    return (single + log_extra) / normalizer

learning_metrics = {
    'log_correct' : make_scorer(
        'log_correct',
        log_correct,
        optimum=1.0,
        worst_possible_result=0.0,
        greater_is_better=True
    )
}

@register_selector
class AutoSklearnClassifierSelector(AutoSklearnModel, Selector):
    """
    Wrapper around autosklearn classifier for use as a Selector.

    It is to be trained on multi classification labels that indicate which
    algorithms correctly predicted the instance. For selection, this selector
    will then choose the model with the highest predicted probability of being
    correct.
    """
    _kind: ModelType = 'classifier'

    def __init__(self, ensemble: Ensemble, **kwargs) -> None:
        Selector.__init__(self, ensemble)
        client = Client(processes=False,
                        n_workers=kwargs['n_jobs'],
                        threads_per_worker=1,
                        dashboard_address=None)

        if (metric_name := kwargs.get('metric', None)):
            if (metric := learning_metrics.get(metric_name, None)):
                kwargs.update({'metric': metric})
            else:
                raise NotImplementedError(f'{metric_name=} unknown')
        self.model = AutoSklearnClassifier(**kwargs, dask_client=client)

    def selections(self, X: np.ndarray) -> np.ndarray:
        """
        Selects the model with the highest predicited probability of correctly
        classifying each sample.
        """
        model_correct_probabilities = self.model.predict_proba(X)
        return np.argmax(model_correct_probabilities, axis=1)

    @classmethod
    def kind(cls) -> ModelType:
        return cls._kind

    def autosklearn_model(self) -> AutoSklearnClassifier:
        return self.model


@register_selector
class AutoSklearnRegressorSelector(AutoSklearnModel, Selector):
    """
    Wrapper around autosklearn regressor for use as a a Selctor.

    It is to be trained on vector labels that indicate the error of each
    model in an ensemble. For selection, this selector will then choose
    the model in the ensemble predicted to have the lowest error.
    """
    _kind: ModelType = 'regressor'

    def __init__(self, ensemble: Ensemble, **kwargs) -> None:
        Selector.__init__(self, ensemble)
        client = Client(processes=False,
                        n_workers=kwargs['n_jobs'],
                        threads_per_worker=1,
                        dashboard_address=None)
        self.model = AutoSklearnRegressor(**kwargs, dask_client=client)

    def selections(self, X: np.ndarray) -> np.ndarray:
        """
        Selects the model predicted to have the lowest error.
        """
        model_error_predictions = self.model.predict(X)
        return np.argmin(model_error_predictions, axis=1)

    @classmethod
    def kind(cls) -> ModelType:
        return cls._kind

    def autosklearn_model(self) -> AutoSklearnRegressor:
        return self.model
