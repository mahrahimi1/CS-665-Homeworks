# More information about Assignment 6

## Code structure

The loss functions you're expected to write are in `loss_functions.py`.

The regularizer you're expected to write is in `regularization.py`.

The generic gradient descent code you're expected to write is in
`linear_model.py`.  During the development of your gradient descent
code, note the following hint.  Because the size of your validation
set is different from the training set, it is more convenient to
divide the loss of each point by the size of the appropriate set,
interpreting it as an average loss. This lets you compare the training
loss and the validation loss more directly. Make sure to scale the
gradient as well: this makes it easy to compare regularization values
and learning rates across datasets.

## Driver scripts

The three driver scripts are `regression.py`, `logistic.py`, and `svm.py`.

Note that `evaluate_model` in `utils.py` only works for classification
models: `RegressionPredictor` in `regression.py` does not include a
`classify` method. You'll have to write your own version of
`evaluate_model` that works with regression models for your report.
