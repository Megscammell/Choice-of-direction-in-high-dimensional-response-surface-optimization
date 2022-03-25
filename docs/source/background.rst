.. highlight:: rst

.. _styled-numbered-lists:

Background
==========


Notation
---------

Consider the following notation used throughout.

* :math:`d`: number of variables;

* :math:`x^{(1)} \sim N(0, I_d)`: Initial point, where :math:`I_d` is the identity matrix of size :math:`d \times d`;

* steepest descent iteration:

.. math::
    :label: sd

    x^{(k+1)} = x^{(k)} - \gamma^{(k)} s^{(k)},

where :math:`\gamma^{(k)}` is the step-length, :math:`s^{(k)}` is the search direction and :math:`k=1,2,...` is the iteration number;

* :math:`M = (m_{j, i})_{j, i=1}^{N,d}`: Design matrix with :math:`N` observations and :math:`d` variables centred at :math:`x^{(k)}`;

* :math:`Y = (y_1,...,y_N)^T`: response function values at each observation of the design matrix :math:`M`.


Outline
--------

Suppose it is required to find an approximate minimizer of the following response function,

 .. math::
    :label: response

    y_j = \eta(z_j) + \epsilon_j,

where :math:`j = 1,2,...` and :math:`z_1, z_2,...` are values of a :math:`d`-dimensional predictor :math:`x \in R^{d}`.
Typically response surface methodology (RSM) is employed to approximate a minimizer of :math:`\eta(x)`. 
The most often cited RSM strategy is the Box-Wilson algorithm (see :cite:`box1951experimental` and :cite:`hill1966review`), which consists of two phases.
The first phase checks whether :math:`\eta(x)` is approximately linear in the neighbourhood of :math:`x^{(k)}`. If so, a local linear model is constructed
using a fractional factorial design centred at :math:`x^{(k)}` and the response function values at each observation of the design matrix. The coefficients
:math:`\hat{\theta} = (\hat{\theta}_0,\hat{\theta}_1,....,\hat{\theta}_d)^T` of the local linear model are computed using least-squares. If the local linear model is significant then the search direction is
:math:`s^{(k)} = (\hat{\theta}_1,....,\hat{\theta}_d)^T`. Then :math:`x^{(k+1)}` is obtained by applying :eq:`sd` and we set :math:`k \gets k + 1`. The first phase of RSM is repeated with new point :math:`x^{(k)}`.
Otherwise, if the linear model is insignificant, the second phase of RSM is applied.
That is, a second-order model is required, and the choice of design is changed accordingly.
The focus throughout will be on the first phase of RSM and, in particular, the choice of search direction :math:`s^{(k)}`.

.. _search_LS:

Extension of RSM for large dimensions (LS)
--------------------------------------------

For RSM, the number of variables :math:`d` is typically small since a fractional factorial design is used to construct the linear model within the first phase of RSM. 
However, to extend the use of RSM for large :math:`d`, several coordinates of :math:`x^{(k)}` may be updated at each iteration of :eq:`sd`. The number of
coordinates updated will depend on the number of variables within the fractional factorial design. If the local linear model is insignificant, a different subset of
coordinates may be chosen, and the updating of :eq:`sd` may be attempted again. If all coordinates of :math:`x^{(k)}` have been explored, and the linear model is insignificant at
all subsets of coordinates, then the first phase of RSM will terminate.

Throughout, :ref:`LS <search_LS>` will be used to denote the discussed search direction. 

.. _alt_search_1:

Alternative search direction (MY)
------------------------------------------

Suppose entries of :math:`M` are chosen randomly as :math:`+1's` and :math:`-1's`, with the condition that the number of :math:`+1's` and :math:`-1's` in each column are equal.
Furthermore, suppose :math:`Y` contains the response function values at each observation of :math:`M`. Consider the following alternative search direction, inspired by the work in :cite:`gillard2018optimal`, :cite:`gillard2018optimalest` and :cite:`zhigljavsky1991theory`,

.. math::
    :label: search_1

    s^{(k)} = M^TY.

The advantage of using search directions of form :eq:`search_1` is that all variables of :math:`x^{(k)}` can be updated simultaneously. Furthermore, numerical comparisons show that search directions of form
:eq:`search_1` are more accurate than using search directions discussed in :ref:`LS <search_LS>` for large dimensions. The code and outputs for numerical comparisons can be found 
`here <https://github.com/Megscammell/Estimate-of-direction-in-RSM/tree/main/numerical_experiments>`_.
