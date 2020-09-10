import os
import json
import pickle
from hashlib import sha1
from autosklearn.classification import AutoSklearnClassifier

import logging

import data
from state import State
from data import LocalDataset, OpenMLTask, OpenMLSuiteIterator

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
        self._task = OpenMLTask(task_id)

        save_mode = self.config['save_models']
        should_save = (save_mode == 'all')

        for dataset in self._task:

            automodel = None
            if self.model_exists(dataset) and not retrain:
                automodel = self.load_model(dataset)
            else:
                automodel = self.train_model(dataset, save=should_save)

            self.automodels[dataset.id_string] = automodel

    def dataset(self, task_id=-1, repeat=0, fold=0, sample=0):
        if self.config['kind'] == 'openml_task':
            return self._task.get_dataset(repeat, fold, sample)
        else:
            raise NotImplemented

    def train_model(self, dataset, save=True):
        """ Trains a model on a dataset """
        dprint(f'Training {Experiment._model_path(self.config, dataset)}')
        cfg = self.config

        automodel = AutoSklearnClassifier(
            seed=cfg['seed'],
            time_left_for_this_task=cfg['time_per_task_iteration'],
            initial_configurations_via_metalearning=0, #TODO, discuss with Joeran
            resampling_strategy='cv',
            resampling_strategy_arguments={'folds': 5}
        )

        X_train, y_train, _, _ = dataset.get()
        automodel.fit(X_train, y_train)

        # TODO: Contact developers about this?
        # Fixes error:
        #   DataPreprocess has no attribute ColumnTransformer
        Experiment._fit_transformers_hack_(automodel, X_train, y_train)

        if save:
            self.save_model(automodel, dataset)

        return automodel


    def task(self):
        kind = self.config['kind']
        if kind != 'openml_task':
            raise RunTimeError(f'Experiment is for {kind} and not openml_task')

        return self._task

    @staticmethod
    def _fit_transformers_hack_(automodel, X_train, y_train):
        """
        Due to the weirdness that models can't predict before their
        DataPreprocessor is fit, which it should be to enable the ensemble to
        predict...anyways
        """
        for model in Experiment.get_ensemble(automodel):
            model.fit_transformer(X_train, y_train)

    @staticmethod
    def get_ensemble(automodel):
        return [model for weight, model in automodel.get_models_with_weights()]

    def save_model(self, model, dataset):
        return Experiment._save_model(self.config, model, dataset)

    @staticmethod
    def _save_model(config, model, dataset):
        mpath = Experiment._model_path(config, dataset)
        dprint(f'Saving {mpath}')
        pickle.dump(model, open(mpath, 'wb'))

    def load_model(self, dataset):
        return Experiment._load_model(self.config, dataset)

    @staticmethod
    def _load_model(config, dataset):
        mpath = Experiment._model_path(config, dataset)
        if not os.path.exists(mpath):
            raise FileNotFoundError(f'No model found at {mpath}')

        dprint(f'Loading {mpath}')
        return pickle.load(open(mpath, 'rb'))


    def model_exists(self, dataset):
        return Experiment._model_exists(self.config, dataset)

    @staticmethod
    def _model_exists(config, dataset):
        mpath = Experiment._model_path(config, dataset)
        return os.path.exists(mpath)


    def model_path(self, dataset):
        """ Gets the path for a model on a task given a confgi """
        return Experiment._model_path(self.config, dataset)

    @staticmethod
    def _model_path(config, dataset):
        """ Gets the path for a model on a task iteration """
        config_id = config['id']
        save_dir = config['save_dir']
        id_string = dataset.id_string

        json_string = json.dumps(config, sort_keys=True)
        config_hash = sha1(json_string.encode('utf-8')).hexdigest()[:12]

        model_name = f'{config_id}_{config_hash}_{id_string}.pkl'
        model_path = os.path.join(save_dir, 'models', model_name)
        return model_path
