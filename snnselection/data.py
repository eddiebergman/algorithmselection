import pandas
import openml
from itertools import product, repeat

from sklearn.preprocessing import LabelEncoder

class Dataset:
    """
    Wrapper around a dataset that is to be trained on,
    WILL corrupt data

    Consists of training samples, and test samples with their corresponding
    labels as well as a unique identifier
    """

    def __init__(self, X_train, y_train, X_test, y_test, id_string):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.id_string = id_string

    def get(self):
        return self.X_train, self.y_train, self.X_test, self.y_test

    def id_string(self):
        return self.id_string


class OpenMLTask:

    def __init__(self, task_id, allow_samples=False):
        task = openml.tasks.get_task(task_id)
        X, y = OpenMLTask.process_openml_task(task)

        self.task = task
        self.X = X
        self.y = y
        self.allow_samples = False

        n_repeats, n_folds, n_samples = self.task.get_split_dimensions()

        if not allow_samples and n_samples != 1:
                raise ValueError(f'Task {task.id} has {n_samples} samples'
                                + 'and allow_samples has been set to False')

        if not allow_samples:
            self.itr = product(range(n_repeats), range(n_folds), [0])
        else:
            self.itr = product(
                range(n_repeats), range(n_folds), range(n_samples)
            )

    def __iter__(self):
        return self

    def __next__(self):
        i_repeat, i_fold, i_sample = next(self.itr)
        return self.get_dataset(i_repeat, i_fold, i_sample)

    def get_dataset(self, repeat=0, fold=0, sample=0):
        train_idxs, test_idxs = self.task.get_train_test_split_indices(
            repeat=repeat, fold=fold, sample=sample
        )

        X, y = self.X, self.y
        task_id = self.task.id
        id_string = f't{task_id}_r{repeat}_f{fold}_s{sample}'

        return Dataset(
            X_train=X.loc[train_idxs].copy(),
            y_train=y[train_idxs].copy(),
            X_test=X.loc[test_idxs].copy(),
            y_test=y[test_idxs].copy(),
            id_string=id_string
            )

    @staticmethod
    def process_openml_task(task):
        X, y, categorical_mask, _ = task.get_dataset().get_data(task.target_name)

        # Process labels
        if y.dtype == 'category' or y.dtype == object:
            binary_labels = LabelEncoder().fit_transform(y.values)
            y = pandas.Series(binary_labels)

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

class LocalDataset:

    def __init__(self):
        raise NotImplementedError()

class OpenMLSuiteIterator:

    def __init__(self):
        raise NotImplementedError()
