"""
Wrappers around autosklearn models to make them Ensembles
"""
from typing import Any, List

import numpy as np
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor

from ..ensemble import Ensemble, register_ensemble
from ..base import ModelType, is_predictor, is_classifier


@register_ensemble
class AutoSklearnClassifierEnsemble(Ensemble):
    """
    Wrapper around an autosklearn model.
    """
    _kind: ModelType = 'classifier'

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.model = AutoSklearnClassifier(**kwargs)

    # Ensemble Methods
    def trained(self) -> bool:
        """ Whether the model is trained or not """
        # pylint: disable=protected-access
        return self.model.automl_ is not None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """ Fit the model """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Get the models prediction """
        return self.model.predict_proba(X)

    def models(self) -> List[Any]: # type:ignore
        models = [member for _, member in self.model.get_models_with_weights()]

        # NOTE: autosklearn is not type safe and it loads objects from pickle
        #   Must do a runtime check to assert that they can all predict
        for m in models:
            assert is_classifier(m), f'`predict_proba(X)` missing, {m=}'

        return models

    def model_predictions(self, X: np.ndarray) -> np.ndarray:
        """ Get the models probability predicitons """
        return np.asarray([m.predict_proba(X) for m in self.models()])

    @classmethod
    def kind(cls) -> ModelType:
        return cls._kind

@register_ensemble
class AutoSklearnRegressorEnsemble(Ensemble):
    """
    Wrapper around an autosklearn model.
    """
    _kind: ModelType = 'regressor'

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.model = AutoSklearnRegressor(**kwargs)

    # Ensemble Methods
    def trained(self) -> bool:
        """ Whether the model is trained or not """
        # pylint: disable=protected-access
        return self.model.automl_ is not None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """ Fit the model """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Get the models prediction """
        return self.model.predict(X)

    def models(self) -> List[Any] : # type:ignore
        models = [member for _, member in self.model.get_models_with_weights()]

        # NOTE: autosklearn is not type safe and it loads objects from pickle
        #   Must do a runtime check to assert that they can all predict
        for m in models:
            assert is_predictor(m), f'`predict(X)` missing, {m=}'

        return models

    def model_predictions(self, X: np.ndarray) -> np.ndarray:
        """ Get the models probability predicitons """
        return np.asarray([m.predict(X) for m in self.models()])

    @classmethod
    def kind(cls) -> ModelType:
        return cls._kind
