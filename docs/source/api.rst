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


Rhobust Cost Functions
------------------------
The package includes several robust cost functions that can be used to reduce the 
influence of outliers in the optimization process.

.. autosummary::
   :toctree: _autosummary

   linear_rho_at_R2
   soft_l1_rho_at_R2
   cauchy_rho_at_R2
   arctan_rho_at_R2


Regularization Utilities
------------------------
The package includes utilities for regularization of the parameters, which can be used to
impose constraints on the parameters or to incorporate prior knowledge about the parameters.

.. autosummary::
   :toctree: _autosummary

   build_squared_regularization
   build_soft_squared_regularization


Additional Utilities
------------------------
The package also includes additional utility functions that can be used for various
purposes, such as studying the jacobian or computing the jacobian of the residuals
by finite differences.

.. autosummary::
   :toctree: _autosummary

   build_jacobian
   study_jacobian
