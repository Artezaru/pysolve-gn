"""
pysolve-gn - Robust Gauss-Newton Least Squares Solver.
Copyright (C) 2026 Artezaru, artezaru.github@proton.me

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from typing import Callable, Any, Sequence, Union, Optional, Dict, Tuple, List
from numbers import Real, Integral
from numpy.typing import ArrayLike

import numpy
import scipy

from .term import Term
from .solver import solve


class GaussNewtonSolver(object):
    r"""
    A class that implements the Gauss-Newton optimization method for solving
    non-linear least squares problems.

    The Gauss-Newton method is an iterative optimization algorithm that minimizes the sum of squared residuals
    between a set of observed data points and a model defined by a set of parameters. It is particularly
    effective for solving non-linear least squares problems where the residuals are non-linear functions of
    the parameters.

    The optimization is performed by linearizing the problem at each iteration and solving a linear least
    squares problem to update the parameters. The process is repeated until convergence, which is typically
    determined by a stopping criterion based on the change in parameters or the norm of the residuals.

    The solved problem is defined by several terms (class :class:`Term`), such that :

    .. math::

        \min_{\mathbf{p}} \frac{1}{2} \sum_{i} w_i \sum_j \rho_i\left(\| \mathbf{r}_{i,j}(\mathbf{p}) \|^2\right)

    To perform the optimization, the user can create an instance of the GaussNewtonSolver
    class and call the :meth:`solve` method with the appropriate parameters and options.

    .. seealso::

        For more details on notations for the optimization problem and the algorithm,
        please refer to the mathematical section of the documentation.

    Parameters
    ----------
    terms : Union[Term, Sequence[Term]]
        An instance of the Term class or a sequence of Term instances that encapsulates terms
        of the optimization problem, including the residual functions, jacobian functions,
        and other properties of the terms.


    Version
    -------

    - 0.1.0: Initial version for the GaussNewtonSolver class.


    """

    __slots__ = [
        "_terms",
        "_callback_func",
        "_update_func",
    ]

    def __init__(
        self,
        terms: Union[Term, Sequence[Term]],
    ) -> None:
        if isinstance(terms, Term):
            terms = [terms]
        if not isinstance(terms, Sequence):
            raise ValueError(
                "terms must be an instance of Term or a sequence of Term instances."
            )
        for term in terms:
            if not isinstance(term, Term):
                raise ValueError(
                    "All elements of terms must be instances of the Term class."
                )
        if len(terms) == 0:
            raise ValueError("terms sequence cannot be empty.")

        self._terms = terms
        self._callback_func = None
        self._update_func = None

    @property
    def n_terms(self) -> int:
        """
        Get the number of Term instances in the GaussNewtonSolver.

        Returns
        -------
        int
            The number of Term instances in the GaussNewtonSolver.
        """
        return len(self._terms)

    def get_term(self, index: Integral) -> Term:
        """
        Get the Term instance at the specified index.

        Parameters
        ----------
        index : Integral
            The index of the Term instance to retrieve.

        Returns
        -------
        Term
            The Term instance at the specified index.

        Raises
        ------
        IndexError
            If the index is out of range for the terms sequence.
        """
        if not isinstance(index, Integral):
            raise ValueError("index must be an integer.")
        index = int(index)
        if index < 0 or index >= len(self._terms):
            raise IndexError("index out of range for terms sequence.")
        return self._terms[index]

    # ============= Optimization Method =============
    def set_callback_function(self, callback: Callable[[Dict], bool]) -> None:
        """
        Set the callback function for the optimization process.

        The callback function is called at each iteration of the optimization process with a dictionary
        containing information about the current state of the optimization. The callback function can
        return True to stop the optimization process early based on custom criteria.

        Parameters
        ----------
        callback : Callable[[Dict], bool]
            A function that takes a dictionary as input and returns a boolean value. The dictionary contains
            information about the current state of the optimization, such as the current parameters, residuals,
            iteration number, etc.

        Returns
        -------
        None
            This method does not return anything, but it sets the callback function for the optimization process.
        """
        if not callable(callback):
            raise ValueError("callback must be a callable function.")

        self._callback = callback

    def remove_callback_function(self) -> None:
        """
        Remove the callback function for the optimization process.

        This method removes the callback function that was previously set for the optimization process, if any.

        Returns
        -------
        None
            This method does not return anything, but it removes the callback function for the optimization process.
        """
        self._callback = None

    def has_callback_function(self) -> bool:
        """
        Check if a callback function is set for the optimization process.

        Returns
        -------
        bool
            True if a callback function is set for the optimization process, False otherwise.
        """
        return hasattr(self, "_callback") and self._callback is not None

    # ============= Optimization Method =============
    def solve(
        self,
        x0: ArrayLike,
        *,
        max_iteration: Optional[Integral] = None,
        max_time: Optional[Real] = None,
        ftol: Optional[Real] = None,
        xtol: Optional[Real] = None,
        gtol: Optional[Real] = None,
        atol: Optional[Real] = None,
        ptol: Optional[Real] = None,
        callback_func: Optional[Callable[[Dict], bool]] = None,
        update_func: Optional[
            Callable[[numpy.ndarray, numpy.ndarray], ArrayLike]
        ] = None,
        verbosity: Integral = 0,
        history: bool = False,
        history_details: Optional[Union[str, Sequence[str]]] = None,
    ) -> Tuple[numpy.ndarray, Optional[List[Dict]]]:
        """
        Solve the optimization problem using the Gauss-Newton method.

        This method performs the optimization by iteratively updating the parameters
        based on the residuals and Jacobians of the terms defined in the GaussNewtonSolver.
        The optimization continues until one of the stopping criteria is met, which can be
        based on the maximum number of iterations, maximum time, or convergence thresholds
        for the change in parameters or residuals.

        .. seealso::

            :func:`pysolvegn.solve`  for the detailed
            description of the input parameters, stopping criteria, and the optimization process.


        Parameters
        ----------
        x0 : ArrayLike
            The initial guess for the parameters to be optimized. Shape (n_parameters,).

        **kwargs : Any
            Additional keyword arguments to pass to the optimization function.
            This can include stopping criteria,
            verbosity level, history options, etc.


        """
        return solve(
            self._terms,
            x0,
            max_iteration=max_iteration,
            max_time=max_time,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            atol=atol,
            ptol=ptol,
            callback_func=callback_func,
            update_func=update_func,
            verbosity=verbosity,
            history=history,
            history_details=history_details,
        )
