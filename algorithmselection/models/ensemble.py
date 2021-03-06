"""
Defines an Ensemble Protocol
"""
from typing import Dict, Type, List, Any
from abc import ABC, abstractmethod

import numpy as np

from .model import Model

class Ensemble(ABC, Model):
    """
    Encapsulates an ensemble, implementors must
    focus on two main points,
        - A way to accumulate predictions from its ensemble models
            for the methods `predict()` and `predict_proba()`

    In addition, you may want to overwrite `fit()` if fitting logic is more
    complicated than just iterating through and fitting each individual model
    """
    @abstractmethod
    def __init__(self, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def model_predictions(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def models(self) -> List[Any]: #type: ignore
        raise NotImplementedError

ensembles : Dict[str, Type[Ensemble]] = {}

def register_ensemble(cls: Type[Ensemble]):
    ensembles[cls.__name__] = cls
    return cls
