## Module Structure
A seperate folder for each experiment to keep everything together
* `store -> <configuration_name>`

Each `<configuration_name>` folder should have a `config.json` that specifies
what should be run. These do not have to be in the module folder as a result.

## Commands
* `--info <config>`
    * Should return info on the current state of the experiment in case it was
        interupted early.

* `--verify <config>`
    * Should verify the integrity of the entire state and give a diagnostics of what went wrong if not. 
    Verify if all directory's are there, if all configs are in place and all configs can be parsed.

* `--run <config> [--next]`
    * Runs the listed config and resumes progress if it was halted. The option
    `--next` lets you specify to only run the next step.

## Configs
```JSON
# Config
{
    'seed': 1337,
    'kind' : 'dataset' | 'openml_task' | 'openml_suite',
    'save_dir': './mytestdirectory'

    'dataset': {
        'path' : '/path/to/dataset',
        'kind' : 'regression' | 'classification' | 'clustering'
        'label_column': 'label_name' | 13,
    } |
    'openml_task' : {
        'id' : 99,
        'save_models' : 'all' | 'best' | 'none'
    } |
    'openml_suite' : {
        'id' : 99,
        'alias' : 'OpenML-CC18'
    }

    'split' : [0.3, 0.5, 0.2]

    'layers' : [10, 5, 3] || 'auto' || 'method',

    'snn_training' : {
        'performance_normalization': 'none' | 'something'
        'method' : 'default',
        'loss_function' : {
            'kind': 'contrastive_loss',
            'param1': 'y',
            'param2': 'x'
        },
    },

    'algorithms' : {
        'performance_measure': 'something'
        'pool' : {
            'id1' : {
                'kind': 'from_selected_list',
                'params': {
                    'algo_specific' : 'value'
                }
            },
            'id2' : {
                ....
            }
        }
    },

    # Whether to save the algorithm models
    'save_algorithms' : True,

    # Whether to save the SNN model
    'save_snn': True

    # Result tracker, the set of results to keep track of and write to file
    'results': {
        'timings' : True
    }
}
```

## Progress
The project takes place in several steps at which it could fail at any point.
It would be good to put in some progress tracking.

Some major milestones are:
* algorithm training
* algorithm predictions
* algorithm performances
* snn training
* snn evaluation

```JSON
{
    'algorithms_training' : {
        'id1': { 'trained': True, 'model': 'path/to/model'},
        'id2': {...}
    },
    'algorithm_predictions' : {
        'id1': { 'predictions' : 'path/to/predictions'},
        'id2': {...}
    },
    'algorithm_performances' : {
        'id1' : { 'performances': 'path/to/performances'},
## Module Structure
A seperate folder for each experiment to keep everything together
* `store -> <configuration_name>`

Each `<configuration_name>` folder should have a `config.json` that specifies
what should be run. These do not have to be in the module folder as a result.

## Commands
* `--info <config>`
    * Should return info on the current state of the experiment in case it was
        interupted early.

* `--verify <config>`
    * Should verify the integrity of the entire state and give a diagnostics of
      what went wrong if not.  Verify if all directory's are there, if all
      configs are in place and all configs can be parsed.

* `--run <config> [--next]`
    * Runs the listed config and resumes progress if it was halted. The option
