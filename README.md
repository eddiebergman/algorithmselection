## Module Structure
A seperate folder for each experiment to keep everything together
* `store -> <configuration_name>`

Each `<configuration_name>` folder should have a `config.json` that specifies
what should be run. These do not have to be in the module folder as a result.

## Commands
* `--state <config>`
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
    'dataset': '/path/to/dataset' | 'openmlcc18' ,
    'split': '[.3, .5, .2]',    # algo training, snn training, snn testing

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
    'training_opts' : {
        'training_method' : 'default',
        'loss_function' : {
            'kind': 'contrastive_loss',
            'param1': 'y',
            'param2': 'x'
        },
    },

    # The seleciton of algorithms to train
    'algorithms' : {
        'id1' : {
            'kind': 'from_selected_list',
            'params': {
                'algo_specific' : 'value'
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
* algorithm evaluation on dataset
* snn training
* snn evaluation

## make
Read the make file
```
make <option>
```

