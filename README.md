# Local help me readme
Incase of returning after a while

## Directory structure
I think a seperate directory for the following would help
organize things.

* SNN
    * Configs
    * Saved Models

* Algorithm
    * Configs
    * Saved Models

* Results
    * algorithm_performances

## Commands
* `--list`, `--list-all`
    * Should list the filename of everything available under headers

* `--list-snn`
    * Should list the available configs along with whether that particular config has been trained

* `--list-algorithms`
    * Same as above but for algorithms, should also list the available algorithms,
    probably Sklearn models for now.

* `--list-results`
    * List all available result files


* `--verify`
    * Should verify the integrity of the entire state and give a diagnostics of what went wrong if not. Verify if all directory's are there, if all configs are in place and all configs can be parsed.

* `--train-algorithm [config | configdir]`
    * Trains the listed algorithm configs on the specified dataset. If a
    directory is specified then it trains all of them as a 'suite'.


* `--train-snn [config | configdir]`
    * Trains the listed snn configs on the specified dataset.
    Likewise for directory

* `--run-openml-cc18-benchmark` [config | configdir]
    * Runs an snn config/configs on the openmlcc18 benchmark

## Configs
The filename is used as a unique identifier when saving a model.
If `kfold` and `save` is specified, the names are suffixed with `<name>_k001`.

```JSON
# Algorithm Config
{
    'algorithm': 'some_keyword_from_available list',
    'dataset': '/path/to/dataset',
    'params': {
        # Individual algorithm parameters that need to be tuned
    },
    'kfold' : '5',
    'save': True
}
```

```JSON
# SNN Config
{
    'layers' : [] || 'auto' || style, # Since we have a variable amount of
                                      # inputs in the benchmark
    'training_opts' : {
        'training_method' : 'default',
        'loss_function' : {
            'kind': 'contrastive_loss',
            'param1': 'y',
            'param2': 'x'
        },
    }
    'algorithms' : [] || 'path/to/suite_dir', # Algorithms to use in evaluation
    'save_algorithms' : True,
    'kfold' : '5',
    'save': True
}
```

## make
Read the make file
```
make <option>
```

