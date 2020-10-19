import pickle
from abc import ABC, abstractmethod
from typing import Any, Union, List

from numpy import ndarray
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor
from ..model import Model

AutoSklearnModelTypes = Union[AutoSklearnClassifier, AutoSklearnRegressor]


class AutoSklearnModel(ABC, Model):

    def trained(self) -> bool:
        return self.autosklearn_model().automl_ is not None

    def fit(self, X: ndarray, y: ndarray) -> None:
        """ Fit the model """
        self.autosklearn_model().fit(X, y)

    def fit_and_test(self, X: ndarray, y: ndarray,
                     X_test: ndarray, y_test: ndarray) -> None:
        self.autosklearn_model().fit(X, y, X_test, y_test)

    def supports_fit_and_test(self) -> bool:
        return True

    def save(self, path) -> None:
        # Remove dask_client to enable pickling
        self.autosklearn_model().dask_client = None
        self.autosklearn_model().automl_._dask_client = None
        pickle.dump(self, open(path, 'wb'))

    @classmethod
    def load(cls, path) -> Any:  # type: ignore
        return pickle.load(open(path, 'rb'))

    @abstractmethod
    def autosklearn_model(self) -> AutoSklearnModelTypes: ...

    def models(self) -> List[Any]:  # type:ignore
        automodel = self.autosklearn_model()
        return [member for _, member in automodel.get_models_with_weights()]
