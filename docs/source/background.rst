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
:math:`\hat{\theta} = (\hat{\theta}_0,\hat{\theta}_1,....,\hat{\theta}_d)` of the local linear model are computed using least-squares. If the local linear model is significant then the search direction is
:math:`s^{(k)} = (\hat{\theta}_1,....,\hat{\theta}_d)`, :math:`x^{(k+1)}` is obtained by applying :eq:`sd` and :math:`k \gets k + 1`. The first phase of RSM is repeated with new point :math:`x^{(k)}`.
Otherwise, if the linear model is insignificant, then the second phase of RSM is applied.
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
Furthermore, suppose :math:`Y` contains the response function values at each observation of :math:`M`. Consider the following alternative search direction

.. math::
    :label: search_1

    s^{(k)} = M^TY

Throughout, :ref:`MY <alt_search_1>` will be used to denote the search direction :eq:`search_1`.

.. _alt_search_2:

Alternative search direction with Moore-Penrose inverse (MP)
------------------------------------------------------------------

Consider the design matrix :math:`M` and response vector :math:`Y` defined in :ref:`MY <alt_search_1>`. Consider the following alternative search directions using the Moore-Penrose inverse.

.. math::
    :label: search_MP_left

    s^{(k)} = (M^TM)^-M^TY,

and

.. math::
    :label: search_MP_right

    s^{(k)} = M^T(MM^T)^-Y,
 
where :math:`(.)^-` is the Moore-Penrose pseudo inverse.
Throughout, :ref:`MP <alt_search_2>` will be used to denote the search directions :eq:`search_MP_left` and :eq:`search_MP_right`.

Comparison of search directions
---------------------------------------------------------

Iterations of :eq:`sd` with search directions :ref:`MY <alt_search_1>` and :ref:`MP <alt_search_2>` will terminate when
a predefined total number of response function evaluations are met. The number of response function evaluations can be fixed or can be determined
by applying the first phase of RSM with search direction :ref:`LS <search_LS>` first, and then applying the same number of response function evaluations
with search directions :ref:`MY <alt_search_1>` and :ref:`MP <alt_search_2>`.
