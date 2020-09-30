"""
Main runner for experiments.
Runs configs described in json, read README.md for full description
"""
import os
import gc
import json
from typing import Dict, Any, Optional

import numpy as np

from .tasks.open_ml_task import Task, OpenMLTaskWrapper
from .config_defaults import defaults


def taskrunner(
    config_path: str,
    store_tasks: Optional[bool] = True,
    autorun: Optional[bool] = False
):
    """ Runs a task described by config path """

    config: Dict[str, Any] = {}
    tasks: Dict[str, Task] = {}

    with open(config_path) as file:
        config = json.load(file)

    config = {**defaults['config'], **config}

    np.random.seed(config['seed'])

    # Create the directories if they do not exist
    save_dir = config['save_dir']
    save_dir = os.path.abspath(save_dir)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    task_descriptions = config['tasks']
    for i, task_description in enumerate(task_descriptions):
        # Populate with default id if not present
        idd = {'id': str(i)}
        task_description = {**idd, **task_description}

        task_kind = task_description['kind']

        task = None
        if task_kind == 'openml_task':
            task = OpenMLTaskWrapper(save_dir,
                                     task_description,
                                     store_models=store_tasks)

        elif task_kind == 'local_dataset':
            raise NotImplementedError

        elif task_kind == 'openml_suite':
            raise NotImplementedError

        else:
            raise ValueError(f'{task_kind} not handled')

        if autorun:
            task.run()

        if store_tasks:
            task_id = task_description['id']
            tasks[task_id] = task

        # Explicit garbage collecting to combat memory issues
        gc.collect()

    return tasks
