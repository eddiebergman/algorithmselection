"""
Defines the base class for a Task as well trainers for
ensembles and selectors
"""
import os
import pickle
from abc import ABC, abstractmethod
from typing import (
    Any, TypeVar, Dict, Union, TypedDict, Iterator, Generic, cast, Optional
)

from numpy import ndarray

from ..config_defaults import defaults
from ..models import Ensemble, ensembles
from ..models import Selector, selectors
from ..models import ensemble_evaluators

# TODO fix the registering and pickling issue, something to do
#   with how it's imported
# PickleError: object <class X> is not the same as X
# pylint: disable=unused-import
from ..models.autosklearn.ensembles import (
    AutoSklearnClassifierEnsemble, AutoSklearnRegressorEnsemble
)
# pylint: disable=unused-import
from ..models.autosklearn.selectors import (
    AutoSklearnClassifierSelector, AutoSklearnRegressorSelector
)

Config = Dict[str, Any]
Dataset = TypedDict('Dataset', {
    'X_ensemble': ndarray,
    'y_ensemble': ndarray,
    'X_selector': Union[ndarray, None],
    'y_selector': Union[ndarray, None],
    'X_test': ndarray,
    'y_test': ndarray
}
)


def train_ensemble(ensemble_config: Config, data: Dataset) -> Ensemble:
    """ Trains an ensemble """
    ensemble_kind = ensemble_config['kind']
    ensemble_cls = ensembles[ensemble_kind]

    ensemble_params = ensemble_config['params']
    ensemble = ensemble_cls(**ensemble_params)

    X, y = data['X_ensemble'], data['y_ensemble']

    ensemble.fit(X, y)

    # TODO This was put in as a stop measure against single model ensembles
    # Ultimately this should be allowed and a default behaviour should be added
    # to the selector to not have to fit (as there is only a single model it can
    # choose)
    if len(ensemble.models()) <= 1:
        raise RuntimeError(
            f'Ensemble requires more than 1 model, {len(ensemble.models())=}'
            + 'perhaps allocate more resource to finding a suitable ensemble'
        )
    return ensemble


def train_selector(
    selector_config: Config,
    ensemble: Ensemble,
    data: Dataset
) -> Selector:
    """ Trains a selector  """
    selector_kind = selector_config['kind']
    selector_cls = selectors[selector_kind]

    evaluator_kind = selector_config['ensemble_evaluator']
    ensemble_evaluator = ensemble_evaluators[evaluator_kind]

    selector_params = selector_config['params']
    selector = selector_cls(ensemble, **selector_params)

    # Selector get's trained to predict some metric of the ensembles
    # ability
    X = data['X_selector']
    y = ensemble_evaluator(ensemble, X, data['y_selector'])
    selector.fit(X, y)

    return selector


Key = TypeVar('Key')


# TODO still need some way to indicate the tasks type
class Task(ABC, Generic[Key]):
    """
    Subclasses of Task define an iterator through keys which are linked to
    a specific ensemble, selector and dataset, obtainable through the other
    methods the subclass implements.
    """

    @abstractmethod
    def __init__(
        self,
        save_dir: str,
        task_config: Config,
        store_models: Optional[bool] = True
    ) -> None:
        self.task = {**defaults['openml_task'], **task_config}
        self.save_dir = save_dir
        self.store_models = store_models

        self.ensembles: Dict[Key, Ensemble] = {}
        self.selectors: Dict[Key, Selector] = {}

        self.selector_split: float \
            = self.task['selector'].get('train_split', 0.0)

        self.task_dir = os.path.join(self.save_dir, self.task['id'])
        if not os.path.exists(self.task_dir):
            os.mkdir(self.task_dir)

        self.model_dir = os.path.join(self.task_dir, 'models')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

    def run(self) -> None:
        """ Runs the task """
        print(f'Running task {self.task["id"]}')
        ensemble_config = self.task['ensemble']
        selector_config = self.task['selector']

        for key in self.key_iterator():
            print(f'key = {self.name(key)}')

            data = self.dataset(key)

            ensemble = None
            if self.ensemble_exists(key):
                print(f'Ensemble already exists, {ensemble_config}')

                if self.store_models:
                    ensemble = self.load_ensemble(key)
                    self.ensembles[key] = ensemble

            else:
                try:
                    print(f'\tFitting new ensemble, {ensemble_config}')

                    ensemble = train_ensemble(ensemble_config, data)
                    self.save_ensemble(ensemble, key)

                    if self.store_models:
                        self.ensembles[key] = ensemble

                except ValueError:
                    print(f'Failed to train {ensemble_config=} for {self.task=}')


            if selector_config['kind'] is not None and ensemble is not None:

                selector = None
                if self.selector_exists(key):
                    print(f'\tSelector already exists, {selector_config}')

                    if self.store_models:
                        selector = self.load_selector(key)
                        self.selectors[key] = selector
                else:
                    try:
                        print(f'\tFitting new selector, {selector_config}')

                        selector = train_selector(selector_config,
                                              ensemble,
                                              data)
                        self.save_selector(selector, key)

                        if self.store_models:
                            self.selectors[key] = selector

                    except ValueError:
                        print(f'Failed to train {selector_config=} for {self.task=}')

            print(f'Finised task {self.task["id"]}')

    @abstractmethod
    def dataset(self, key: Key) -> Dataset:
        """ Return the dataset associated with a model key """

    @abstractmethod
    def key_iterator(self) -> Iterator[Key]:
        """ Returns a list of subtasks in the task by key """

    @abstractmethod
    def name(self, key: Key) -> str:
        """ Returns a string representation of a model key """

    def has_selector(self) -> bool:
        return self.task['selector']['kind'] is not None

    def ensemble_path(self, key: Key) -> str:
        """ Gives the path the ensemble would be stored at """
        name = self.name(key)
        return os.path.join(self.model_dir, 'ensemble_' + name + '.pkl')

    def selector_path(self, key: Key) -> str:
        """ Gives the path where the selector model would be stored at """
        name = self.name(key)
        return os.path.join(self.model_dir, 'selector_' + name + '.pkl')

    def ensemble_exists(self, key: Key) -> bool:
        """ Check if the ensemble model exists """
        fpath = self.ensemble_path(key)
        return os.path.exists(fpath)

    def selector_exists(self, key: Key) -> bool:
        """ Checks if the selector model exists """
        fpath = self.selector_path(key)
        return os.path.exists(fpath)

    def save_ensemble(self, ensemble: Ensemble, key: Key) -> None:
        """ Saves the ensemble at path basd on key """
        fpath = self.ensemble_path(key)
        pickle.dump(ensemble, open(fpath, 'wb'))

    def save_selector(self, selector: Selector, key: Key) -> None:
        """ Saves a selector model """
        fpath = self.selector_path(key)
        print(selector)
        pickle.dump(selector, open(fpath, 'wb'))

    def load_ensemble(self, key: Key) -> Ensemble:
        """ Loads an ensemble model """
        fpath = self.ensemble_path(key)
        obj = pickle.load(open(fpath, 'rb'))
        return cast(Ensemble, obj)

    def load_selector(self, key: Key) -> Selector:
        """ Loads a selector model """
        fpath = self.selector_path(key)
        obj = pickle.load(open(fpath, 'rb'))
        return cast(Selector, obj)
