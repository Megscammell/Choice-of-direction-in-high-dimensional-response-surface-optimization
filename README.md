# The choice of direction in high-dimensional response surface optimization
[![CI](https://github.com/Megscammell/Estimate-of-direction-in-RSM/actions/workflows/config.yml/badge.svg)](https://github.com/Megscammell/Estimate-of-direction-in-RSM/actions/workflows/config.yml)
[![codecov](https://codecov.io/gh/Megscammell/Estimate-of-direction-in-RSM/branch/main/graph/badge.svg?token=HMOJXTZXV4)](https://codecov.io/gh/Megscammell/Estimate-of-direction-in-RSM)
[![Documentation Status](https://readthedocs.org/projects/estimate-of-direction-in-rsm/badge/?version=latest)](https://estimate-of-direction-in-rsm.readthedocs.io/en/latest/?badge=latest)

Response surface methodology (RSM) is used to approximate a minimizer of a response function from a series of observations that contain errors. The first phase of RSM involves constructing a linear model, whose coefficients are used to determine the search direction for steepest descent. However, if the number of variables is large, it can be computationally expensive to approximate the search direction. Hence, an alternative search direction is proposed in order to improve the efficiency and accuracy of the first phase of RSM when the number of variables is large.

Numerical experiments and outputs can be found at [here](https://github.com/Megscammell/Estimate-of-direction-in-RSM/tree/main/numerical_experiments). The aim of experiments is to compare the alternative search direction with the standard search direction, consisting of the coefficients of the local linear model.


## Documentation
Documentation for Estimate-of-direction-in-RSM can be found at https://estimate-of-direction-in-rsm.readthedocs.io/.

## Installation
To install and test the the Estimate-of-direction-in-RSM source code, type the following into the command line:

```console
$ git clone https://github.com/Megscammell/Estimate-of-direction-in-RSM.git
$ cd Estimate-direction-in-RSM
$ python setup.py develop
$ pytest
```

## Quickstart
A quickstart guide to run all numerical experiments can be found [here](https://estimate-of-direction-in-rsm.readthedocs.io/en/latest/Run%20numerical%20experiments/index.html).
