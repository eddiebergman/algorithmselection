import os
import json
import pickle
from hashlib import sha1

import numpy as np
from sklearn import metrics
from autosklearn.classification import AutoSklearnClassifier

from .state import State
from .data import LocalDataset, OpenMLTask, OpenMLSuiteIterator

config_defaults = {
    'seed' : 1337,
    'time_per_task_iteration' : 60,
    'verbose' : True,
    'save_models' : 'all'
}

# TODO: change to logging
dflag = True
dprint = lambda s, dflag=dflag: (_ := print(s)) if dflag is True else None

class Experiment:
    """
    Class that is useful for ipython sessions
    Note to self: try to return something useful for each function
    """

    def __init__(self, config_path, debug=True):
        self.state = State(config_path)

        self.config = { **config_defaults, **self.state.config }
        self.results = self.state.results
        self.progress = self.state.progress

        self.automodels = {}

        dflag = debug

        # Create the directories if they do not exist
        save_dir = self.config['save_dir']
        save_dir = os.path.abspath(save_dir)
        models_dir = os.path.join(save_dir, 'models')

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        if not os.path.isdir(models_dir):
            os.mkdir(models_dir)

        # Creating this object will automatically run it
        data_kind = self.config['kind']
        if data_kind == 'local_dataset':
            raise NotImplementedError

        elif data_kind == 'openml_task':
            self.run_openml_task()

        elif data_kind == 'openml_suite':
            raise NotImplementedError

        else:
            raise ValueError(f'{data_kind} not handled')


    def run_openml_task(self, retrain=False):
        """
        Runs experiment on an openml_task 

        Params
        ======
        retrain | bool : False
            Whether to train even if models already exists

        Returns
        =======
        List[AutoSklearnClassifier]
            A list of trained autosklearn classifiers
        """
        task_id = self.config['openml_task']['id']
        max_folds = self.config['openml_task'].get('max_folds', None)
        self._task = OpenMLTask(task_id, max_folds=max_folds)

        save_mode = self.config['save_models']
        should_save = (save_mode == 'all')

        for dataset in self._task:

            automodel = None
            if self.model_exists(dataset) and not retrain:
                automodel = self.load_automodel(dataset)

            else:
                automodel = self.train_automodel(dataset, save=should_save)

            self.automodels[dataset.id_string] = automodel


    def get_dataset(self, task_id=None, repeat=0, fold=0, sample=0):
        if self.config['kind'] == 'openml_task' and task_id is None:
            return self._task.get_dataset(repeat, fold, sample)
        else:
            raise NotImplemented

    def get_automodel(self, dataset):
        return self.automodels[dataset.id_string]


    def get_ensemble(self, dataset):
        return [m for _, m in self.get_ensemble_with_weights(dataset)]


    def get_ensemble_with_weights(self, dataset):
        base_automl = self.automodels[dataset.id_string]._automl[0]

        models = None
        if base_automl._resampling_strategy in ['cv', 'cv-iterative-fit']:
            models = base_automl.cv_models_
        else:
            models = base_automl.models_

        return base_automl.ensemble_.get_models_with_weights(models)

    def get_ensemble_evaluations(self, dataset):
        ensemble = self.get_ensemble(dataset)
        #TODO: only works for classifiers with predict_proba

        X = dataset.X_test.to_numpy()
        y = dataset.y_test.to_numpy()

        prob_vecs = [m.predict_proba(X) for m in ensemble]
        pred_vecs = [(prob_vec[:,1] > 0.5).astype(int) for prob_vec in prob_vecs]
        corr_vecs = [(pred_vec == y).astype(int) for pred_vec in pred_vecs]
        accuracies = [metrics.accuracy_score(y, preds) for preds in pred_vecs]
        return {
            'probabilities' : prob_vecs,
            'predictions' : pred_vecs,
            'correct' : corr_vecs,
            'accuracies' : accuracies
        }


    def train_automodel(self, dataset, save=True, ensemble_size=50):
        """ 
        Trains a model on a dataset
        """
        # TODO
        # It would probably be better to not rely on our default being the same
        # as autosklearn's defaults (e.g. ensemble_size=50), findout a pythonic
        # way to only specify the parameter if passed to this function
        dprint(f'Training {Experiment._automodel_path(self.config, dataset)}')
        cfg = self.config

        automodel = AutoSklearnClassifier(
            seed=cfg['seed'],
            time_left_for_this_task=cfg['time_per_task_iteration'],
            initial_configurations_via_metalearning=0, #TODO, discuss with Joeran
            resampling_strategy='cv',
            resampling_strategy_arguments={'folds': 5}
            ensemble_size=ensemble_size
        )

        X_train, y_train, _, _ = dataset.get()
        automodel.fit(X_train, y_train)

        # TODO: Contact developers about this?
        # Fixes error:
        #   DataPreprocess has no attribute ColumnTransformer
        # Experiment._fit_transformers_hack_(automodel, dataset)

        if save:
            self.save_automodel(automodel, dataset)

        return automodel

    def task(self):
        kind = self.config['kind']
        if kind != 'openml_task':
            raise RunTimeError(f'Experiment is for {kind} and not openml_task')

        return self._task


    @staticmethod # _fit_transformers_hack_
    def _fit_transformers_hack_(automodel, dataset):
        """
        Due to the weirdness that models can't predict before their
        DataPreprocessor is fit, which it should be to enable the ensemble to
        predict...anyways
        """
        for model in Experiment.get_ensemble(automodel):
            model.fit_transformer(dataset.X_train, dataset.y_train)


    def save_automodel(self, model, dataset):
        Experiment._save_automodel(self.config, model, dataset)

    @staticmethod
    def _save_automodel(config, model, dataset):
        mpath = Experiment._automodel_path(config, dataset)
        dprint(f'Saving {mpath}')
        pickle.dump(model, open(mpath, 'wb'))


    def load_automodel(self, dataset):
        return Experiment._load_automodel(self.config, dataset)

    @staticmethod
    def _load_automodel(config, dataset):
        mpath = Experiment._automodel_path(config, dataset)
        if not os.path.exists(mpath):
            raise FileNotFoundError(f'No model found at {mpath}')

        dprint(f'Loading {mpath}')
        return pickle.load(open(mpath, 'rb'))


    def model_exists(self, dataset):
        return Experiment._model_exists(self.config, dataset)

    @staticmethod
    def _model_exists(config, dataset):
        mpath = Experiment._automodel_path(config, dataset)
        return os.path.exists(mpath)


    def automodel_path(self, dataset):
        """ Gets the path for a model on a task given a confgi """
        return Experiment._automodel_path(self.config, dataset)

    @staticmethod
    def _automodel_path(config, dataset):
        """ Gets the path for a model on a task iteration """
        config_id = config['id']
        save_dir = config['save_dir']
        id_string = dataset.id_string

        json_string = json.dumps(config, sort_keys=True)
        config_hash = sha1(json_string.encode('utf-8')).hexdigest()[:12]

        model_name = f'{config_id}_{config_hash}_{id_string}.pkl'
        automodel_path = os.path.join(save_dir, 'models', model_name)
        return automodel_path
