.. role:: bash(code)
   :language: bash

.. _installation:

Installation
=============
To install the Estimate-of-direction-in-RSM repository:

.. code-block:: bash

   $ git clone https://github.com/Megscammell/https://github.com/Megscammell/Estimate-of-direction-in-RSM.git
   $ cd Estimate-direction-in-RSM
   $ python setup.py develop

To ensure all tests are working, create an environment and run the tests using :bash:`pytest`:

.. code-block:: bash

   $ conda env create -f environment.yml
   $ conda activate est_dir_env
   $ pytest



Quickstart
---------------------

An example of applying the first phase of RSM with a noisy quadratic response function is presented below:

.. code-block:: python
  :linenos:

   >>> import est_dir
   >>> import numpy as np
   >>>
   >>> np.random.seed(10)
   >>> d = 100
   >>> no_vars = d
   >>> n = 16
   >>> region = 0.1
   >>> max_func_evals = 1000
   >>> cov = np.identity(d)
   >>> lambda_1 = 1
   >>> lambda_2 = 4
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

The purpose of each line of code within the example is discussed in the following table.

.. list-table::
   :widths: 10 50
   :header-rows: 1

   * - Line number
     - Purpose of each line of code within the example
   * - 1 - 2
     - Import the required libraries. 
   * - 4
     - Initialize the pseudo-random number generator seed.
   * - 5
     - Set the dimension as :bash:`d = 100`.	
   * - 6
     - Set the number of variables to update to :bash:`no_vars = d`.	
   * - 7
     - The number of observations of the design matrix is set to :bash:`n = 16`.
   * - 8
     - Set the region of exploration to :bash:`region = 0.1`. 
   * - 9
     - Set the maximum number of function evaluations before terminating the first phase of RSM with alternative search direction to :bash:`max_func_evals = 1000`. 
   * - 10
     - Create the variable :bash:`cov`, which is assigned an identity matrix.
   * - 11 - 17
     - Create the variable :bash:`A`, which is assigned a diagonal matrix with smallest and largest eigenvalues :bash:`lambda_1 = 1` and :bash:`lambda_2 = 4` respectively.
   * - 18
     - Create the variable :bash:`starting_point`, which is a sample from the multivariate normal distribution with mean zero and identity covariance matrix.
   * - 19
     - Create the variable :bash:`minimizer`, which is a 1-D array containing zeros.
   * - 21 - 22
     - Define a function :bash:`f` to apply the first phase of RSM with alternative search direction.  
   * - 24
     - Set :bash:`minimizer`, :bash:`A`, :bash:`mu=0` and :bash:`sd=0.5` as response function arguments. The arguments are required to run :bash:`f`.
   * - 25 - 33
     - Run the first phase of RSM with alternative search direction with specified input parameters.
   * - 34 - 36
     - Check outputs of the first phase of RSM with alternative search direction.

Additional examples
---------------------

Additional examples of the first phase of RSM with alternative search direction can be found 
`here <https://github.com/Megscammell/Estimate-of-direction-in-RSM/tree/main/Examples>`_.
All examples have an intuitive layout and structure, which can be easily followed. 