.. role:: bash(code)
   :language: bash

Output parameters
===================

List of available results
--------------------------

The table below contains details of the outputs from :bash:`alternative_search_direction()`.

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Output
     - Type
     - Description
   * - :bash:`upd_point`
     - 1-D array
     - Final point produced at the final iteration of the first phase of RSM with alternative search direction.
   * - :bash:`init_func_val`
     - float
     - Response function value at the initial point :math:`x^{(1)}`.
   * - :bash:`f_val`
     - float
     - Response function value at the final point.
   * - :bash:`full_time`
     - float
     - Total amount of time taken.
   * - :bash:`total_func_evals_step`
     - integer
     - Total number of function evaluations used to compute the step length :math:`\gamma^{(k)}` in :eq:`sd` at each iteration.
   * - :bash:`total_func_evals_dir`
     - integer
     - Total number of function evaluations used to compute the search direction :math:`s^{(k)}` in :eq:`sd` at each iteration.
   * - :bash:`no_iterations`
     - integer
     - Total number of iterations.
