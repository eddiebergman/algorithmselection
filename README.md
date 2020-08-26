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

    'dataset': {
        # If a benchmark is specified, most of the rest is ignored
        'path' : '/path/to/dataset' || 'openmlcc18' ,
        'kind' : 'regression',
        'label_column': 'label',

        # Options betwen 'kfold' and 'train_test' for snn evaluation
        # Must specify a segment for algorithm training though
        'split': {
            'seed': 1337,
            'kind': 'kfold' || 'train_test'
            'k' : 5                 # Specify if kind=kfold

            'algorithm_training' : 0.3
            'snn_training' : 0.7
            'snn_testing' : 0.0
        }

        # Whether to keep a copy of the original dataset, default to true
        'copy' : true,
    }

    # Mutually exclusive [train_test_split or kfold]
    'train_test_split': {
        'algorithm_training' : .3,
        'snn_training': .5,
        'snn_evaluation': .2
    },
    'kfold': {
        'k': 5,
        'algo_training': .3,
        'snn': .7
    }

    # Since we have a variable amount of inputs in the benchmark
    'layers' : [] || 'auto' || 'method',

    # How the snn should be trained, options to be added here
    'snn_training_opts' : {
        'performance_normalizing': 'something'
        'training_method' : 'default',
        'loss_function' : {
            'kind': 'contrastive_loss',
            'param1': 'y',
            'param2': 'x'
        },
    },

    # How an algorithms performance should be measured
    'algorithm_performance_function' : 'something'

    # The seleciton of algorithms to train
    'algorithms' : {
        'id1' : {
            'kind': 'from_selected_list',
            'params': {
                'algo_specific' : 'value'
            }
        },
        'id2' : {
            ....
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
        'id2': {...}
    }
    'snn_training' : {
        'trained': false,
        'model': 'path/to/model',
    }
    'snn_evaluation' : {
        'trained': false,
        'selections': 'path/to/choices'
    }
}
```

## make
Read the make file
```
make <option>
```

