"""
Defines an Ensemble Protocol
"""
from typing import Dict, Type, List, Any
from abc import ABC, abstractmethod

import numpy as np

from .base import ModelType, Predictor

class Ensemble(ABC, Predictor):
    """
    Encapsulates an ensemble, implementors must
    focus on two main points,
        - A way to accumulate predictions from its ensemble models
            for the methods `predict()` and `predict_proba()`

    In addition, you may want to overwrite `fit()` if fitting logic is more
    complicated than just iterating through and fitting each individual model
    """

    @abstractmethod
    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def trained(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def model_predictions(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def kind(cls) -> ModelType:
        raise NotImplementedError

    @abstractmethod
    def models(self) -> List[Any]: #type: ignore
        raise NotImplementedError

ensembles : Dict[str, Type[Ensemble]] = {}

def register_ensemble(cls: Type[Ensemble]):
    ensembles[cls.__name__] = cls
    return cls
