from typing import Literal, Any, Protocol

from numpy import ndarray

ModelType = Literal['classifier', 'regressor']

class Model(Protocol):

    def trained(self) -> bool: ...

    def fit(self, X: ndarray, y: ndarray) -> None: ...

    def fit_and_test(self, X: ndarray, y: ndarray,
                     X_test: ndarray, y_test: ndarray) -> None:
        raise NotImplementedError

    def supports_fit_and_test(self) -> bool: ...

    def predict(self, X: ndarray) -> ndarray: ...

    @classmethod
    def kind(cls) -> ModelType: ...

    def save(self, path) -> None: ...

    @classmethod
    def load(cls, path) -> Any: ... #type: ignore
