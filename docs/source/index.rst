Welcome to the documentation for Estimate-of-direction-in-RSM
================================================================
.. image:: https://github.com/Megscammell/Estimate-of-direction-in-RSM/actions/workflows/config.yml/badge.svg
   :target: https://github.com/Megscammell/Estimate-of-direction-in-RSM/actions/workflows/config.yml

.. image:: https://codecov.io/gh/Megscammell/Estimate-of-direction-in-RSM/branch/main/graph/badge.svg?token=HMOJXTZXV4
   :target: https://codecov.io/gh/Megscammell/Estimate-of-direction-in-RSM


Response surface methodology (RSM) can be applied to approximate a minimizer of a response function from a series of observations that contain errors.
The first phase of RSM involves constructing a linear model, whose coefficients are used to determine the search direction for steepest descent.
However, if there are many variables, it can be computationally expensive to approximate the search direction.
Hence, an alternative search direction is proposed to improve the efficiency and accuracy of the first phase of RSM when the dimension is large.

Numerical experiments compare the alternative search direction with the standard search direction, consisting of the coefficients of the local linear model.
Numerical experiments and outputs can be found `here <https://github.com/Megscammell/Estimate-direction-in-RSM/tree/main/numerical_experiments>`_.


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   background
   Installation/index
   Run numerical experiments/index
   References/index
