import numpy as np
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor

from ..selector import Selector, register_selector
from ..ensemble import Ensemble
from ..base import ModelType


@register_selector
class AutoSklearnClassifierSelector(Selector):
    """
    Wrapper around autosklearn classifier for use as a Selector.

    It is to be trained on multi classification labels that indicate which
    algorithms correctly predicted the instance. For selection, this selector
    will then choose the model with the highest predicted probability of being
    correct.
    """
    _kind: ModelType = 'classifier'

    def __init__(self, ensemble: Ensemble, **kwargs) -> None:
        super().__init__(ensemble)
        # processes=False is key for ensuring automl stays within
        # its processor limitations
        dask_client = Client(n_works=kwargs['n_jobs'], processes=False)
        self.model = AutoSklearnClassifier(**kwargs, dask_client=dask_client)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

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

@register_selector
class AutoSklearnRegressorSelector(Selector):
    """
    Wrapper around autosklearn regressor for use as a a Selctor.

    It is to be trained on vector labels that indicate the error of each
    model in an ensemble. For selection, this selector will then choose
    the model in the ensemble predicted to have the lowest error.
    """
    _kind: ModelType = 'regressor'

    def __init__(self, ensemble: Ensemble, **kwargs) -> None:
        super().__init__(ensemble)
        # processes=False is key for ensuring automl stays within
        # its processor limitations
        dask_client = Client(n_works=kwargs['n_jobs'], processes=False)
        self.model = AutoSklearnRegressor(**kwargs, dask_client=dask_client)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def selections(self, X: np.ndarray) -> np.ndarray:
        """
        Selects the model predicted to have the lowest error.
        """
        model_error_predictions = self.model.predict(X)
        return np.argmin(model_error_predictions, axis=1)

    @classmethod
    def kind(cls) -> ModelType:
        return cls._kind
