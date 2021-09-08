.. role:: bash(code)
   :language: bash

.. highlight:: rst

Numerical experiments
=======================

Response functions
---------------------
Numerical experiments can be applied with one of the following two response functions.

.. math::
    :label: quad

    y(x) = (x - x^*)^T A^T \Lambda A (x - x^*) + \epsilon,

and

 .. math::
    :label: sqrt_quad

    y(x) = \sqrt{(x - x^*)^T A^T \Lambda A (x - x^*)} + \epsilon,

where :math:`x \in R^{d}`, :math:`x^* = (0,0,...,0)^T`, :math:`A` is a random orthogonal matrix, :math:`\Lambda` is a diagonal positive definite matrix with smallest
and largest eigenvalues :math:`\lambda_{min}` and :math:`\lambda_{max}` respectively, and :math:`\epsilon \sim N(0, \sigma^2)`.

Purpose
--------
Numerical experiments compare the first phase of RSM with an initial point :math:`x^{(1)}`, where search directions :ref:`LS <search_LS>`, :ref:`MY <alt_search_1>` and :ref:`MP <alt_search_2>` are
computed for :eq:`sd`,
with 100 different response functions of the form :eq:`quad` or :eq:`sqrt_quad`, with various :math:`\lambda_{max}` and :math:`\sigma^2`.
In addition, different values of :math:`N` will be tested in order to construct the design matrix :math:`M` for directions :ref:`MY <alt_search_1>` and :ref:`MP <alt_search_2>`.


How to run numerical experiments in Python
------------------------------------------------
To run numerical experiments comparing directions :ref:`LS <search_LS>`, :ref:`MY <alt_search_1>` and :ref:`MP <alt_search_2>` for :eq:`sd`,
navigate to the `numerical_experiments folder <https://github.com/Megscammell/Estimate-of-direction-in-RSM/tree/main/numerical_experiments>`_.

The program quad_num_exp_SNR.py is used to compare search directions for a large number of variables.
Values for :bash:`N`, :bash:`d`, :bash:`lambda_max`, :bash:`region`, :bash:`function_type`, :bash:`type_inverse` and :bash:`func_evals` will need to be provided to run quad_num_exp_SNR.py. The following
table describes each input parameter.

.. list-table:: Outputs of metod.py
   :widths: 25 25 50
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - :bash:`N`
     - integer
     - Number of observations of :math:`M`. The design matrix :math:`M` is centred at each point :math:`x^{(k)}` and each observation of :math:`M` is evaluated with the response function.
   * - :bash:`d`
     - integer
     - Number of variables.
   * - :bash:`lambda_max`
     - integer
     - Largest eigenvalue of :math:`\Lambda` in :eq:`quad` and :eq:`sqrt_quad`.
   * - :bash:`region`
     - float
     - Region of exploration :math:`\pm r`. That is, the design matrix represents a change of :math:`\pm r` in the coordinates of the centre point :math:`x^{(k)}`.
   * - :bash:`function_type`
     - string
     - Choose from :bash:`function_type = ‘quad’` or :bash:`function_type = ‘sqr_quad’` to apply numerical experiments with response functions of the form :eq:`quad` or :eq:`sqrt_quad`.
   * - :bash:`type_inverse`
     - string
     - To apply the search direction :ref:`MP <alt_search_2>`, the type of inverse needs to be provided. That is, if a left inverse is required
       (i.e. :eq:`search_MP_left`) then :bash:`type_inverse = ‘left’`.
       Otherwise, if a right inverse is required (i.e. :eq:`search_MP_right`), then :bash:`type_inverse = ‘right’`.
   * - :bash:`func_evals`
     - integer
     - Number of response function evaluations permitted before terminating the first phase of RSM with search directions :ref:`MY <alt_search_1>` and :ref:`MP <alt_search_2>`.
       If  :bash:`func_evals = 0`, the number of response function evaluations permitted will be determined by applying the first phase of RSM with search direction :ref:`LS <search_LS>`.

To run all numerical experiments with search directions :ref:`LS <search_LS>`, :ref:`MY <alt_search_1>` and :ref:`MP <alt_search_2>`,
we must set :bash:`func_evals = 0`.
Furthermore, a :math:`2^{10-6}` fractional factorial design is used to compute search directions :ref:`LS <search_LS>`. Hence, only :math:`N=16` observations are used to evaluate
the response function at each iteration, and only :math:`10` coordinates of :math:`x^{(k)}` are updated if the linear regression model is significant. Therefore, 
updating values of :bash:`N` and :bash:`d` will only have an effect for search directions :ref:`MY <alt_search_1>` and :ref:`MP <alt_search_2>`.

The following is an example of running all numerical experiments with search directions :ref:`LS <search_LS>`, :ref:`MY <alt_search_1>` and :ref:`MP <alt_search_2>`, where :bash:`N=16`, :bash:`d=100`, :bash:`lambda_max=4`,
:bash:`region=0.1`, :bash:`function_type=‘quad’`, :bash:`type_inverse=‘left’` and :bash:`func_evals=0`::

   $ python quad_num_exp_SNR.py 16 100 4 0.1 'quad' 'left' 0

The following is an example of running numerical experiments with search directions :ref:`MY <alt_search_1>` and :ref:`MP <alt_search_2>`, where the number of response function evaluations
permitted is 1000 (i.e. :bash:`func_evals=1000`). ::

   $ python quad_num_exp_SNR.py 16 100 4 0.1 'quad' 'left' 1000

All outputs are saved within csv files and various plots are generated comparing all search directions.
Suppose :math:`x_{LS}^{(K)}`, :math:`x_{MY}^{(K)}` and :math:`x_{MP}^{(K)}` are the final points found by applying the first phase of RSM
with search directions :ref:`LS <search_LS>`, :ref:`MY <alt_search_1>` and :ref:`MP <alt_search_2>` for :eq:`sd`.
The response function values :math:`\eta(x_{LS}^{(K)})`, :math:`\eta(x_{MY}^{(K)})` and :math:`\eta(x_{MP}^{(K)})` are compared.
Furthermore, the distances :math:`||x_{LS}^{(K)} - x^*||`, :math:`||x_{MY}^{(K)} - x^*||` and :math:`||x_{MP}^{(K)} - x^*||` are also compared.
