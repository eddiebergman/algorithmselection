"""
Defaults for configuration objects.
"""
from typing import Dict, Any

ConfigType = Dict[str, Any]

config : ConfigType = {
    'seed' : 1337
}

automodel_opts : ConfigType = {
    'initial_configurations_via_metalearning':0,
    'resampling_strategy':'cv',
    'resampling_strategy_arguments':{'folds': 5},
    'time_left_for_this_task': 300,
    'ensemble_size': 10
}

algorithm_selector : ConfigType = {
    'kind': 'autosklearn',
    'train_split': 0.3,
    'train_on': 'correct_classifications',
    'model_opts': automodel_opts
}

openml_task : ConfigType = {
    'save_automodels': True,
    'max_folds': 1,
    'allow_samples': False,
    'automodel_opts': automodel_opts,
    'algorithm_selector': algorithm_selector,
}

defaults : ConfigType = {
    'config': config,
    'openml_task': openml_task,
    'automodel_opts': automodel_opts,
    'algorithm_selector': algorithm_selector
}
