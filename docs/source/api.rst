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
The API is organized into several sections, each corresponding to a specific aspect of 
the package.

For more detailed explanations of the mathematical background, please refer to the
Mathematical Background section of the documentation.

.. seealso::

   - :doc:`Mathematical Background <math>` for the mathematical foundation and theoretical concepts underlying the package.

Solver Function
----------------

The main function of the package is the ``solve_gauss_newton`` function, which implements the
Gauss-Newton optimization algorithm for solving non-linear least squares problems. 
This function takes as input ``Callable`` objects allowing to compute 
the values of a function and its jacobian with respect to its parameters.

.. autosummary::
   :toctree: _autosummary

   solve_gauss_newton


The package includes several robust cost functions that can be used to reduce the 
influence of outliers in the optimization process.

+------------------------------+---------------------------------------------------------+
| Robust Function :math:`\rho` | Equation                                                |
+==============================+=========================================================+
| ``linear``                   | :math:`\rho(x) = x`                                     |
+------------------------------+---------------------------------------------------------+
| ``soft_l1``                  | :math:`\rho(x) = 2 * ((1 + x) ** 0.5 - 1)`              |
+------------------------------+---------------------------------------------------------+
| ``cauchy``                   | :math:`\rho(x) = \log(1 + x)`                           |
+------------------------------+---------------------------------------------------------+
| ``arctan``                   | :math:`\rho(x) = \arctan(x)`                            |
+------------------------------+---------------------------------------------------------+


Additional Utilities
------------------------
The package also includes additional utility functions that can be used for various
purposes, such as studying the jacobian, computing the jacobian of the residuals
by finite differences or creating residuals and jacobians for regularization terms.

.. autosummary::
   :toctree: _autosummary

   build_numerical_jacobian
   build_squared_regularization
   build_soft_squared_regularization
   study_jacobian
