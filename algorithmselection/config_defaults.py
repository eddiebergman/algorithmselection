"""
Defaults for configuration objects.
"""
from typing import Dict, Any

ConfigType = Dict[str, Any]

config : ConfigType = {
    'seed' : 1337
}

openml_task : ConfigType = {
    'save_automodels': True,
    'max_folds': 1,
    'allow_samples': False,
}

defaults : ConfigType = {
    'config': config,
    'openml_task': openml_task,
}
