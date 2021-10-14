# The choice of direction in high-dimensional response surface optimization
[![CI](https://github.com/Megscammell/Estimate-of-direction-in-RSM/actions/workflows/config.yml/badge.svg)](https://github.com/Megscammell/Estimate-of-direction-in-RSM/actions/workflows/config.yml)
[![codecov](https://codecov.io/gh/Megscammell/Estimate-of-direction-in-RSM/branch/main/graph/badge.svg?token=HMOJXTZXV4)](https://codecov.io/gh/Megscammell/Estimate-of-direction-in-RSM)
[![Documentation Status](https://readthedocs.org/projects/estimate-of-direction-in-rsm/badge/?version=latest)](https://estimate-of-direction-in-rsm.readthedocs.io/en/latest/?badge=latest)

Response surface methodology (RSM) is used to approximate a minimizer of a response function from a series of observations that contain errors. The first phase of RSM involves constructing a linear model, whose coefficients are used to determine the search direction for steepest descent. However, if the number of variables is large, it can be computationally expensive to approximate the search direction. Hence, an alternative search direction is proposed to improve the efficiency and accuracy of the first phase of RSM when the number of variables is large.

## Documentation
Documentation for Estimate-of-direction-in-RSM can be found at https://estimate-of-direction-in-rsm.readthedocs.io/.

## Installation
To install and test the Estimate-of-direction-in-RSM source code, type the following into the command line:

```console
$ git clone https://github.com/Megscammell/Estimate-of-direction-in-RSM.git
$ cd Estimate-direction-in-RSM
$ python setup.py develop
$ pytest
```

## Quickstart

Apply the first phase of RSM with an alternative search direction for large dimensions.

```python
>>> import est_dir
>>> import numpy as np
>>>
>>> np.random.seed(10)
>>> d = 100
>>> no_vars = d
>>> n = 16
>>> region = 0.1
>>> max_func_evals = 1000
>>> lambda_1 = 1
>>> lambda_2 = 4
>>> cov = np.identity(d)
>>> diag_vals = np.zeros(d)
>>> diag_vals[:2] = np.array([lambda_1, lambda_2])
>>> diag_vals[2:] = np.random.uniform(lambda_1 + 0.1,
...                                   lambda_2 - 0.1, (d - 2))
>>> A = np.diag(diag_vals)
>>> starting_point = np.random.multivariate_normal(np.zeros((d)), cov)
>>> minimizer = np.zeros((d, ))
>>> 
>>> def f(x, minimizer, A, mu, sd):
...    return ((x - minimizer).T @ A @ (x - minimizer) + np.random.normal(mu, sd))
...    
>>> func_args = (minimizer, A, 0, 0.5)
>>> (final_point,
...  sp_func_val,
...  fp_func_val,
...  time_taken,
...  func_evals_step,
...  func_evals_dir,
...  number_its) = (est_dir.rsm_alternative_search_direction
                    (starting_point, f, func_args, n, d,
                     no_vars, region, max_func_evals))
>>> assert(np.linalg.norm(starting_point - minimizer) >
...        np.linalg.norm(final_point - minimizer))
>>> assert(sp_func_val > fp_func_val)

```


## Examples

Additional examples of the first phase of RSM with alternative search direction can be found at https://github.com/Megscammell/Estimate-of-direction-in-RSM/tree/main/Examples. All examples have an intuitive layout and structure, which can be easily followed. 
