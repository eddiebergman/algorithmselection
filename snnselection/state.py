import os
import json

def from_json(fpath):
    obj = {}
    with open(fpath, 'r') as fp:
         obj = json.load(fp)
    return obj

def to_json(obj, fpath):
    with open(fpath, 'w') as fp:
        json.dump(obj, fp, sort_keys=True, indent=4)

class State:

    def __init__(self, config_path):
        if not os.path.exists(config_path):
            raise ValueError('No file found at {config_path}')

        self.config = from_json(config_path)
        self.config_path = os.path.abspath(config_path)

        save_dir = os.path.abspath(self.config['save_dir'])

        self.progress_path = os.path.join(save_dir, 'progress.json')
        self.results_path = os.path.join(save_dir, 'results.json')

        if os.path.exists(self.results_path):
            self.results = from_json(self.results_path)
        else:
            self.results = {}

        if os.path.exists(self.progress_path):
            self.progress = from_json(self.progress_path)
        else:
            self.progress = {}

    def save(self, which='all'):
        if which == 'all':
            to_json(self.config, self.config_path)
            to_json(self.results, self.results_path)
            to_json(self.progress, self.progress_path)
        elif which == 'config':
            to_json(self.config, self.config_path)
        elif which == 'results':
            to_json(self.results, self.results_path)
        elif which == 'progress':
            to_json(self.progress, self.progress_path)
        else:
            raise ValueError('Must specify which={all | config | results |\
                             progress}')

    def __str__(self):
        return '\n'.join([
            str(self.config), str(self.results), str(self.progress)
        ])
