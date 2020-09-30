import numpy as np

from .base import ModelType
from .ensemble_evaluators import correct_classifications
from .selector import Selector
from .ensemble import Ensemble


class ClassificationOracle(Selector):
    # TODO if ensemble model_predictions are costly, this
    # needlessly recomputes them
    # Can either pass them in or have Ensemble cache results
    _kind: ModelType = 'classifier'

    def __init__(self, ensemble: Ensemble, y: np.ndarray) -> None:
        super().__init__(ensemble)
        self.y = y

    def selections(self, X: np.ndarray) -> np.ndarray:
        # Just selects the first occurence of a 1 it sees
        # for each instance, indicating that a model correctly
        # classified the instance, else it defaults to the first member
        instance_wise_correct_classifications \
            = correct_classifications(self.ensemble, X, self.y)

        return np.argmax(instance_wise_correct_classifications, axis=1)

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
