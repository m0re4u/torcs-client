# Neural TORCS driver with ES optimization

## Requirements
- [pytorch](http://pytorch.org/)
- [numpy](http://www.numpy.org/)
- [pandas](https://pandas.pydata.org/)
#### included in this repo:
- [pytocl(for TORCS)](https://github.com/moltob/pytocl)


## File structure
- `csv_data` contains data extracted from 2013 SCR driver AUTOPIA
- `models` contains pretrained FNN models
- `nn` contains definitions for the neural nets, their trainig scripts and the dataset loader
- `pytocl` contains the pytocl TORCS interface, which was customized to our needs
- `race-config` contains `.xml` race configurations for running TORCS in headless mode
- `results` contains observed results
- `test` is a test folder included in the _pytocl_ package

## Runnning the client
```
python3 run.py
```
All options for running the client can be seen when passing the `-h` flag. By default, we set all arguments such that they correspond with the model that is provided by the default training procedure. Changes in the trained model require changes in arguments passed to this script, and as such should be handled with care.

## Training the Neural network
```
python3 nn/train.py
```
All options for training can be seen when passing the `-h` flag. By default, we set all arguments to an adequate default value to be used in training.

## Evolving the driver using ES
```
python3 run_ES_OPP.py
```
Again, the defaults that are set work with the default values in the other two scripts. Changes in either of those require changes in the arguments given for this script.
