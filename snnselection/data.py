from itertools import product, repeat

from sklearn.preprocessing import LabelEncoder

def process_task(task):
    X, y, categorical_mask, _ = task.get_dataset().get_data(task.target_name)

    # Process labels
    if y.dtype == 'category' or y.dtype == object:
        binary_labels = LabelEncoder().fit_transform(y.values)
        y = pd.Series(binary_labels)

    elif y.dtype == bool:
        y = y.astype('int')

    # Process Categorical features
    encoding_frames = []
    for col_name list(X.columns[categorical_mask]):
        encodings = pd.get_dummies(X[col_name], prefix=col_name, prefix_sep='_')
        encoding_frames.append(encodings)
        X.drop(col_name, axis=1, inplace=True)

    X = pd.concat([X, *encoding_frames], axis=1)

    return X, y

class OpenMLTaskIterator:

    def __init__(self, X, y, task, allow_samples=False):
        self.X = X
        self.y = y
        self.task = task
        self.allow_samples = False
        n_repeats, n_folds, n_samples = task.get_split_dimensions()

        if not allow_samples and n_samples != 1:
                raise ValueError('Task {task.id} has {n_samples} samples \
                                and allow_samples has been set to False')

        if not allow_samples:
            self.iterator = product(
                range(n_repeats), range(n_folds), repeat(1)
            )
        else:
            self.iterator = product(
                range(n_repeats), range(n_folds), range(n_samples)
            )

    def __iter__(self):
        return self

    def __next__(self):
        i_repeat, i_fold, i_sample = next(self.iterator)
        train_idxs, test_idxs = self.task_get_train_test_split_indices(
            repeat=i_repeat, fold=i_fold, sample=i_sample
        )
        return {
            'X_train' : X.loc[train_idxs].copy(),
            'X_test' : X.loc[test_idxs].copy(),
            'y_train' : y[train_idxs].copy(),
            'y_test' : y[test_idxs].copy(),
            'i_repeat' : i_repeat,
            'i_fold' : i_fold,
            'i_sample' : i_sample,
            'id_string' : f'r{i_repeat}_f{i_fold}_s{i_sample}'
        }
