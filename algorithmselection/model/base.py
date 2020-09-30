"""
Defines interfaces for allowing use of third party libraries.
Primarily aimed at sklearn models use of predict() and predict_proba()

Protocol as interfaces for type checking
https://mypy.readthedocs.io/en/latest/protocols.html#simple-user-defined-protocols

For @runtime_checkable
https://www.python.org/dev/peps/pep-0544/#runtime-checkable-decorator-and-narrowing-types-by-isinstance
"""
from typing import Protocol, runtime_checkable, Literal

from numpy import ndarray

ModelType = Literal['classifier', 'regressor']


@runtime_checkable
class Predictor(Protocol):
    """ Can predict things """

    def predict(self, X: ndarray) -> ndarray: ...


@runtime_checkable
class Regressor(Predictor, Protocol):
    """ Can do regression """


@runtime_checkable
class Classifier(Predictor, Protocol):
    """
    Used as a runtime interface for a Model type, allowing other library
    types to use this interface as long as they have the predict
    and predict_proba methods
    """

    def predict_proba(self, X: ndarray) -> ndarray: ...


def is_predictor(model):
    """ Check if model fufills Predictor Protocol """
    return isinstance(model, Predictor)


def is_regressor(model):
    """ Checks if model fufills Regressor Protocol """
    return isinstance(model, Regressor)


def is_classifier(model):
    """ Checks if model fufills Classifier Protocol """
    return isinstance(model, Classifier)
