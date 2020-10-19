"""
Wrappers around autosklearn models to make them Ensembles
"""
import numpy as np
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor
from dask.distributed import Client

from ..ensemble import Ensemble, register_ensemble
from ..model import ModelType
from .base import AutoSklearnModel

@register_ensemble
class AutoSklearnClassifierEnsemble(AutoSklearnModel, Ensemble):
    """
    Wrapper around an autosklearn model.
    """
    _kind: ModelType = 'classifier'

    def __init__(self, **kwargs) -> None:
        Ensemble.__init__(self)
        client = Client(processes=False,
                        n_workers=kwargs['n_jobs'],
                        threads_per_worker=1,
                        dashboard_address=None,
                        )
        self.model = AutoSklearnClassifier(**kwargs, dask_client=client)

    def autosklearn_model(self) -> AutoSklearnClassifier:
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Get the models prediction """
        return self.model.predict_proba(X)

    def model_predictions(self, X: np.ndarray) -> np.ndarray:
        """ Get the models probability predicitons """
        return np.asarray([m.predict_proba(X) for m in self.models()])

    @classmethod
    def kind(cls) -> ModelType:
        return cls._kind

@register_ensemble
class AutoSklearnRegressorEnsemble(AutoSklearnModel, Ensemble):
    """
    Wrapper around an autosklearn model.
    """
    _kind: ModelType = 'regressor'

    def __init__(self, **kwargs) -> None:
        Ensemble.__init__(self)
        client = Client(processes=False,
                        n_workers=kwargs['n_jobs'],
                        thread_per_worker=1,
                        dashboard_address=None)
        self.model = AutoSklearnRegressor(**kwargs, dask_client=client)

    def autosklearn_model(self) -> AutoSklearnRegressor:
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Get the models prediction """
        return self.model.predict(X)

    def model_predictions(self, X: np.ndarray) -> np.ndarray:
        """ Get the models probability predicitons """
        return np.asarray([m.predict(X) for m in self.models()])

    @classmethod
    def kind(cls) -> ModelType:
        return cls._kind
