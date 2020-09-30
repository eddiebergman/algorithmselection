"""
Module for managing the state and running of tasks
as described by OpenML.
https://docs.openml.org/APIs/
"""
from itertools import product
from typing import Tuple, Iterator, Optional

import pandas
import openml
from openml.tasks.task import OpenMLTask
from numpy import ndarray
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from .task import Task, Config


def process_openml_task(task: OpenMLTask) -> Tuple[ndarray, ndarray]:
    """
    Process an openml task in a generic way,
        LabelEncoder for the categorical labels
        One Hot Encoding for categorical features """
    X, y, categorical_mask, _ = task.get_dataset().get_data(task.target_name)

    # Process labels
    if y is not None:
        if y.dtype == 'category' or y.dtype == object:
            encoded_labels = LabelEncoder().fit_transform(y.values)
            y = pandas.Series(encoded_labels)

        elif y.dtype == bool:
            y = y.astype('int')

    # Process Categorical features
    encoding_frames = []
    for col_name in list(X.columns[categorical_mask]):
        encodings = pandas.get_dummies(
            X[col_name], prefix=col_name, prefix_sep='_'
        )
        encoding_frames.append(encodings)
        X.drop(col_name, axis=1, inplace=True)

    X = pandas.concat([X, *encoding_frames], axis=1)

    return X, y


Key = Tuple[int, int, int]


class OpenMLTaskWrapper(Task[Key]):
    """
    Wrapper around running tasks on an openml tasks
    """

    def __init__(
            self,
            save_dir: str,
            task: Config,
            store_models: Optional[bool] = True
    ):
        super().__init__(save_dir, task, store_models)

        self.openml_task = openml.tasks.get_task(self.task['openml_task_id'])

        _, _, n_samples = self.openml_task.get_split_dimensions()
        if not self.task['allow_samples'] and n_samples != 1:
            raise ValueError(f'Task {self.openml_task.id} has {n_samples} samples'
                             + 'and allow_samples has been set to False')

        self.X, self.y = process_openml_task(self.openml_task)

    def key_iterator(self) -> Iterator[Key]:
        """ Returns an iterator over  """
        n_repeats, n_folds, n_samples = self.openml_task.get_split_dimensions()

        max_folds = self.task.get('max_folds', None)
        if max_folds is not None and max_folds < n_folds:
            n_folds = max_folds

        allow_samples = self.task['allow_samples']
        sample_range = range(n_samples) if allow_samples else [0]

        generator = product(range(n_repeats), range(n_folds), sample_range)
        return generator

    def dataset(self, key: Key):
        """ Returns the dataset for the associated key """
        assert 0.0 <= self.selector_split <= 1

        # Get our main batch of training data
        train_i, test_i = self.openml_task.get_train_test_split_indices(*key)
        X_train = self.X.loc[train_i].copy()
        y_train = self.y[train_i].copy()
        X_test = self.X.loc[test_i].copy()
        y_test = self.y[test_i].copy()

        if self.selector_split == 0.0:
            X_ensemble, y_ensemble, X_selector, y_selector \
                = X_train, y_train, None, None
        else:
            X_ensemble, X_selector, y_ensemble, y_selector \
                = train_test_split(X_train, y_train, test_size=self.selector_split)

        return {
            'X_ensemble': X_ensemble,
            'y_ensemble': y_ensemble,
            'X_selector': X_selector,
            'y_selector': y_selector,
            'X_test': X_test,
            'y_test': y_test
        }

    @staticmethod
    def name(key):
        """ Returns string representation of the iter_key """
        r, f, s = key
        return f'r{r}_f{f}_s{s}'
