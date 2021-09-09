.. role:: bash(code)
   :language: bash

Input parameters
==================

This section provides details on the required and optional inputs of the first phase of RSM with alternative search direction.

Required Inputs
----------------------

The required inputs of :bash:`alternative_search_direction()` are listed below, along with the variable type. All required inputs need to be updated before running the first phase of RSM with alternative search direction. 

.. list-table::
   :widths: 10 10 50
   :header-rows: 1

   * - Input parameter
     - Type
     - Description
   * - :bash:`centre_point`
     - 1-D array
     - Starting point :math:`x^{(1)}` to initialize the first phase of RSM with alternative search direction.
   * - :bash:`f`
     - function
     - Response function.
   * - :bash:`func_args`
     - tuple
     - Extra arguments passed to :bash:`f`.
   * - :bash:`n`
     - integer
     - Number of observations of the design matrix :math:`M`. The design matrix is used to construct the alternative search direction :eq:`search_1`.
   * - :bash:`d`
     - integer
     - Size of dimension.
   * - :bash:`no_vars`
     - integer
     - Number of variables of :bash:`centre_point` to update. A key advantage of the search direction :eq:`search_1` is it can be computed for any dimension. Hence, all variables of :bash:`centre_point` can be updated.
   * - :bash:`region`
     - float
     - Region of exploration :math:`\pm r`. That is, the design matrix :math:`M` represents a change of :math:`\pm r` in the coordinates of the point :math:`x^{(k)}`.
   * - :bash:`max_func_evals`
     - integer
     - Maximum number of function evaluations before stopping.


Optional Inputs
----------------------

The optional inputs of :bash:`alternative_search_direction()` are listed below, along with the variable type.


.. list-table::
   :widths: 10 10 12 45
   :header-rows: 1

   * - Input parameter name
     - Default input
     - Type
     - Description
   * - :bash:`const_back`
     - :bash:`0.5`
     - integer
     - If backward tracking is required to compute the step length :math:`\gamma^{(k)}` in :eq:`sd`, the initial guess of the
       step length will be multiplied by :bash:`const_back` at each iteration
       of backward tracking.
   * - :bash:`back_tol`
     - :bash:`0.000001`
     - integer
     - It must be ensured that the step length :math:`\gamma^{(k)}` computed by backward
       tracking is not smaller than :bash:`back_tol`. If this is the case,
       iterations of backward tracking are terminated. Typically,
       :bash:`back_tol` is a very small number.
   * - :bash:`const_forward`
     - :bash:`2`
     - integer
     - If forward tracking is required to compute the step length :math:`\gamma^{(k)}` in :eq:`sd`, the initial guess of the step length will be multiplied by :bash:`const_forward` at each iteration of forward tracking.
   * - :bash:`forward_tol`
     - :bash:`100000000`
     - integer
     - It must be ensured that the step length :math:`\gamma^{(k)}` computed by forward tracking is not larger than :bash:`forward_tol`. If this is the case,
       iterations of forward tracking are terminated. Typically, :bash:`forward_tol` is a very large number.    