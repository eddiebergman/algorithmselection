import os
import json

def load(config_path):
    return State(config_path)

class State:

    def __init__(self, config_path):
        if not os.path.exists(config_path):
            raise ValueError('No file found at {config_path}')

        dirname = os.path.dirname(os.path.realpath(config_path))

        progress_path = os.path.join(dirname, 'progress.json')
        results_path = os.path.join(dirname, 'results.json')

        config = JSONStateFile.load(config_path)

        # Checks for existing results/progress or creates a new state file
        if os.path.exists(results_path):
            results = JSONStateFile.load(results_path)
        else:
            results = JSONStateFile({}, results_path)

        if os.path.exists(progress_path):
            progress = JSONStateFile.load(progress_path)
        else:
            progress = JSONStateFile({}, progress_path)

        self.config = config
        self.results = results
        self.progress = progress

    def save(self):
        self.config.save()
        self.results.save()
        self.progress.save()

    def __str__(self):
        return '\n'.join([
            str(self.config), str(self.results), str(self.progress)
        ])

class JSONStateFile:

    def __init__(self, obj, save_to):
        self.obj = obj
        self.save_to = save_to

    @staticmethod
    def load(filepath):
        with open(filepath, 'r') as fp:
            obj = json.load(fp)
            return JSONStateFile(obj, filepath)

    def save(self, to=None):
        to = self.save_to if to is None else to
        with open(to, 'w') as fp:
            json.dump(self.obj, fp, sort_keys=True, indent=4)

    def __delitem__(self, key):
        self.obj.__delattr__(key)

    def __getitem__(self, key):
        return self.obj.__getattribute__(key)

    def __setitem__(self, key, value):
        self.obj.__setattr__(key, value)

    def __repr__(self):
        return f"<JSONStateFile fp:{self.save_to}>"

    def __str__(self):
        return self.obj.__str__()
