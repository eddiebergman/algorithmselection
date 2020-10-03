from abc import abstractmethod

import numpy as np


from .base import ModelType
from .ensemble_evaluators import correct_classifications
from .selector import Selector
from .ensemble import Ensemble


class Oracle(Selector):

    def __init__(self, ensemble: Ensemble, y: np.ndarray) -> None:
        super().__init__(ensemble)
        self.y = y

    @abstractmethod
    def correct_selections(self, X: np.ndarray) -> np.ndarray:
        """
        As there may be more than one correct choice per instance
        Should return somethingl like
        [
            [0, 1, 1],
            [0, 0, 0],
            ...
            [1, 0, 0]
        ]
        """
        raise NotImplementedError


class ClassificationOracle(Oracle):
    # TODO if ensemble model_predictions are costly, this
    # needlessly recomputes them
    # Can either pass them in or have Ensemble cache results
    _kind: ModelType = 'classifier'

    def selections(self, X: np.ndarray) -> np.ndarray:
        # Just selects the first occurence of a 1 it sees
        # for each instance, indicating that a model correctly
        # classified the instance, else it defaults to the first member
        return np.argmax(self.correct_selections(X), axis=1)

    def correct_selections(self, X: np.ndarray) -> np.ndarray:
        return correct_classifications(self.ensemble, X, self.y)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        print('Oracle does not need to be fit')

    @classmethod
    def kind(cls) -> ModelType:
        return cls._kind


class MultiLabelClassificationOracle:

    def __init__(self):
        raise NotImplementedError


class RegressionOracle:

    def __init__(self):
        raise NotImplementedError
