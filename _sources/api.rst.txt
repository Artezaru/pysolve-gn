.. currentmodule:: pysolvegn

API Reference
==============

.. contents:: Table of Contents
   :local:
   :depth: 2
   :backlinks: top


This section contains a detailed description of the functions
included in ``pysolvegn``. The reference describes how the methods work and which
parameters can be used. It assumes that you have an understanding of the key concepts.

The API is organized in three parts:

- **Single problem**: solve one least squares problem with :func:`solve`.
- **Batch of independent problems**: solve ``k`` independent least squares problems
  at once with :func:`solve_batch`. Each object of the single API has a batched
  counterpart, prefixed by ``Batch`` (classes) or ``build_batch_`` (builders).
- **Shared tools**: the robust loss functions, used by both modes.

For more detailed explanations of the mathematical background, please refer to the
Mathematical Background section of the documentation.

.. seealso::

   - :doc:`Mathematical Background <math>` for the mathematical foundation and theoretical
     concepts underlying the package.


Single problem
--------------

Solver
~~~~~~

The main function of the package is the :func:`solve` function, which implements the
robust Gauss-Newton (or Levenberg-Marquardt) algorithm for solving a non-linear least
squares problem. This function takes as input :class:`Term` objects that define the
residuals and Jacobians of the problem, an optional :class:`Parametrization`, and
returns a :class:`SolveResult` containing the optimized parameters and information
about the optimization.

.. autosummary::
   :toctree: _autosummary

   solve
   SolveResult
   Term
   Parametrization

Implemented Parametrizations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The package provides several implemented parametrizations that can be used to define
how the input parameters are transformed into the output parameters.

.. autosummary::
   :toctree: _autosummary

   build_affine_parametrization
   build_fixed_parametrization
   build_sigmoid_parametrization
   build_positive_parametrization

Implemented Regularizations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The package provides several implemented regularizations that can be used to define
additional constraints or penalties on the parameters in the least squares problem.

.. autosummary::
   :toctree: _autosummary

   build_squared_regularization
   build_soft_squared_regularization
   build_absolute_regularization

Additional Utilities
~~~~~~~~~~~~~~~~~~~~

The package also includes additional utility functions, such as computing the Jacobian
of the residuals by finite differences, selecting a regularization weight or studying
the conditioning of the problem.

.. autosummary::
   :toctree: _autosummary

   build_numerical_jacobian
   perform_Lcurve_analysis
   study_optimization


Batch of independent problems
-----------------------------

Solver
~~~~~~

The :func:`solve_batch` function solves ``k`` independent least squares problems
sharing the same terms, each with its own parameters (array with shape
``(k, n_parameters)``). The stopping criteria are applied to each problem independently:
a converged problem is removed from the processing while the others keep iterating.
Each problem gives the same result as if it was solved alone with :func:`solve`.

The callables of a :class:`BatchTerm` take the parameters of the ``m`` problems in
processing with shape ``(m, n_parameters)`` and their indices in the batch with shape
``(m,)``, where ``m`` changes during the optimization. A :class:`BatchParametrization`
applies the same transformation to all the problems (its callables do not take the
indices).

.. autosummary::
   :toctree: _autosummary

   solve_batch
   BatchSolveResult
   BatchTerm
   BatchParametrization

Implemented Parametrizations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Batched counterparts of the implemented parametrizations: the same transformation
is applied to all the problems of the batch.

.. autosummary::
   :toctree: _autosummary

   build_batch_affine_parametrization
   build_batch_fixed_parametrization
   build_batch_sigmoid_parametrization
   build_batch_positive_parametrization

Implemented Regularizations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Batched counterparts of the implemented regularizations. The prior values can be
shared by all the problems (shape ``(n,)``) or defined for each problem
(shape ``(k, n)``), and the weight can be a scalar or one value per problem
(shape ``(k,)``).

.. autosummary::
   :toctree: _autosummary

   build_batch_squared_regularization
   build_batch_soft_squared_regularization
   build_batch_absolute_regularization

Additional Utilities
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   build_batch_numerical_jacobian


Robust loss functions
---------------------

The package includes several robust cost functions that can be used to reduce the
influence of outliers in the optimization process. They are selected with the ``loss``
argument of :class:`Term` and :class:`BatchTerm` (by name, or as a custom callable).
:math:`x = r^2` denotes the squared residual.

+------------------------------+------------------------------------------------------------------------------+
| Robust Function :math:`\rho` | Equation                                                                     |
+==============================+==============================================================================+
| ``linear``                   | :math:`\rho(x) = x`                                                          |
+------------------------------+------------------------------------------------------------------------------+
| ``soft_l1``                  | :math:`\rho(x) = 2 ((1 + x)^{1/2} - 1)`                                      |
+------------------------------+------------------------------------------------------------------------------+
| ``huber``                    | :math:`\rho(x) = x` if :math:`x \leq 1`, :math:`2 x^{1/2} - 1` otherwise     |
+------------------------------+------------------------------------------------------------------------------+
| ``cauchy``                   | :math:`\rho(x) = \log(1 + x)`                                                |
+------------------------------+------------------------------------------------------------------------------+
| ``arctan``                   | :math:`\rho(x) = \arctan(x)`                                                 |
+------------------------------+------------------------------------------------------------------------------+
| ``tukey``                    | :math:`\rho(x) = (1 - (1 - x)^3) / 3` if :math:`x \leq 1`, :math:`1/3`       |
|                              | otherwise                                                                    |
+------------------------------+------------------------------------------------------------------------------+

The transition between inliers and outliers is located at :math:`|r| = 1`. The
``loss_scale`` argument :math:`C` of :class:`Term` and :class:`BatchTerm` moves it to
:math:`|r| = C` (in the unit of the residuals) by replacing any loss function, predefined
or custom, by :math:`\rho_C(x) = C^2 \rho(x / C^2)` (see :func:`scale_rho_function`).

.. autosummary::
   :toctree: _autosummary

   linear_rho
   soft_l1_rho
   huber_rho
   cauchy_rho
   arctan_rho
   tukey_rho
   scale_rho_function
   get_rho_function_by_name