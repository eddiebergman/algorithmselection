# TODO
### Idea reworks needed
Currently AutoSklearnClassifierSelector does so based on the model
most likely to predict correctly. This probability returned by `predict_proba`
is not garunteed to be the best choice.

### Config Verification
Need to make some sort of config verification for the different
sets of parameters a user can provide.

**Tests**
* `AutoSklearnClassifier` must be given classification labels
    * `! (selector=AutoSklearnClassifier && ensemble_evaluator != binary_correct_classifications)` 
