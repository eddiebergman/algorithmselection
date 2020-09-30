"""
Module for defining what a selector does as well as providing
common traning paradigms and selection strategies
"""
from typing import Dict, Type
from abc import ABC, abstractmethod

import numpy as np

from .ensemble import Ensemble
from .base import ModelType
from ..util import instance_wise

class Selector(ABC):
    """
    A base class for a selector that encapsulates common training types
    and computations required.
    """

    @abstractmethod
    def __init__(self, ensemble: Ensemble, **kwargs) -> None:
        self.ensemble = ensemble

    @abstractmethod
    def selections(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def kind(cls) -> ModelType:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        selections = self.selections(X)

        model_predictions = self.ensemble.model_predictions(X)
        instance_wise_predictions = instance_wise(model_predictions)

        # TODO Sadly np.fromiter doesn't work for creating 2d arrays
        # For now list comprehension will do but it could be vastly
        # sped up by pre-allocating space and not having to copy over
        # results to the new np.ndarray.
        # However the output array might be 2d/3d which makes allocation a bit
        # harder.
        #
        # Consider:
        #   len(X) = n, len(ensemble) = k, len(single_prediction) = c
        #
        # In regression tasks to a single valued label c=1 but even in
        # simple probability predictions, the prediction is a probability for
        # each class, c>1
        selected_predictions = np.asarray([
            predictions[i] for predictions, i
            in zip(instance_wise_predictions, selections)
        ])
        return selected_predictions


selectors : Dict[str, Type[Selector]] = {}

def register_selector(cls: Type[Selector]):
    selectors[cls.__name__] = cls
    return cls
