Mathematical Background
=======================

.. contents:: Table of Contents
   :local:
   :depth: 2


Gauss-Newton optimization
--------------------------

Consider a least squares problem on the form:

.. math::

    \min_{\mathbf{p}} = \frac{1}{2} \left(\| \mathbf{r}(\mathbf{p}) \|^2\right)

where :math:`\mathbf{r}` is a residual function (:math:`\mathbb{R}^{n_p} \rightarrow \mathbb{R}^{n_r}`)
depending on the parameters :math:`\mathbf{p} \in \mathbb{R}^{n_p}` and returning a vector in :math:`\mathbb{R}^{n_r}`.

The problem is solve using an iterative algorithm based on the Gauss-Newton method.
At each iteration, an update :math:`\Delta \mathbf{p}` to the current parameters :math:`\mathbf{p_k}`
is searched to minimize the cost function. 

The solution is given by:

.. math::

   \Delta \mathbf{p} = - \left(\mathbf{J}^T \mathbf{J}\right)^{-1} \mathbf{J}^T \mathbf{r}

where :math:`\mathbf{r}` is the full residual vector in :math:`\mathbb{R}^{n_r}` evaluated at :math:`\mathbf{p_k}`
and :math:`\mathbf{J}` is the full jacobian matrix in :math:`\mathbb{M}_{n_r, n_p}(\mathbb{R})` containing
each :math:`\frac{\partial r_j}{\partial p_l} \forall j \in ( 1, n_r ) \forall l \in ( 1, n_p )`.

The next iteration is performed for :math:`\mathbf{p_{k+1}} = \mathbf{p_k} + \Delta \mathbf{p}` until convergence.

.. seealso::

   The class :class:`pysolvegn.Term` to represent a least square problem. 
   
   Using ``rJ``-type term, this class store two ``Callable`` to compute 
   the residual vector :math:`\mathbf{r}` (as a 1D-array with shape ``(n_residuals,)``)
   and jacobian matrix :math:`\mathbf{J}` (as a 2D-array with shape ``(n_residuals, n_parameters)``)
   from a set of parameters :math:`\mathbf{p}` (as a 1D-array with shape ``(n_parameters,)``).

   Using ``gH``-type term, this class store two ``Callable`` to compute 
   the Hessian matrix :math:`\mathbf{H} = \mathbf{J}^T \mathbf{J}` (as a 2D-array with shape ``(n_parameters, n_parameters)``)
   and the gradient vector :math:`\mathbf{g} = \mathbf{J}^T \mathbf{r}` (as a 1D-array with shape ``(n_parameters,)``)
   from a set of parameters :math:`\mathbf{p}` (as a 1D-array with shape ``(n_parameters,)``).


Demonstration
~~~~~~~~~~~~~~

Consider a least squares problem of the form:

.. math::

    \min_{\mathbf{p}} = \frac{1}{2} \left(\| \mathbf{r}(\mathbf{p}) \|^2\right)

Developing the residuals :math:`\mathbf{r}(\mathbf{p})` around the current parameters :math:`\mathbf{p_k}`:

.. math::

   \mathbf{r}(\mathbf{p_k} + \Delta \mathbf{p})
   \approx
   \mathbf{r}
   + \mathbf{J} \Delta \mathbf{p}
   + \frac{1}{2}
   \begin{bmatrix}
   \Delta \mathbf{p}^T \mathbf{H}_1 \Delta \mathbf{p} \\
   \vdots \\
   \Delta \mathbf{p}^T \mathbf{H}_{n_r} \Delta \mathbf{p}
   \end{bmatrix}
   + \ldots

where :math:`\mathbf{J}(\mathbf{p_k}) = \nabla \mathbf{r}(\mathbf{p_k})` is the Jacobian
of the residuals with respect to the parameters, and :math:`\mathbf{H}_j(\mathbf{p_k})`
is the Hessian of the :math:`j`-th residual with respect to the parameters. By convention,
the quantities are considered evaluated at :math:`\mathbf{p_k}` such that
:math:`\mathbf{r} = \mathbf{r}(\mathbf{p_k})`.

Therefore, for the least squares cost

.. math::

   F(\mathbf{p}) = \frac{1}{2} \| \mathbf{r}(\mathbf{p}) \|^2

Using the Taylor expansion above and retaining terms up to second order in
:math:`\Delta \mathbf{p}`:

.. math::

   F(\mathbf{p_k} + \Delta \mathbf{p})
   \approx
   F(\mathbf{p_k})
   + \mathbf{J}^T \mathbf{r} \, \Delta \mathbf{p}
   + \frac{1}{2}
   \Delta \mathbf{p}^T
   \left(
      \mathbf{J}^T \mathbf{J}
      + \sum_{j=1}^{n_r} r_j \mathbf{H}_j
   \right)
   \Delta \mathbf{p}

The gradient of the least squares cost is therefore

.. math::

   \mathbf{g} = \nabla F(\mathbf{p_k}) = \mathbf{J}^T \mathbf{r}

and its exact Hessian is

.. math::

   \mathbf{H} = \mathbf{J}^T \mathbf{J} + \sum_{j=1}^{n_r} r_j \mathbf{H}_j

The Gauss-Newton approximation consists in neglecting the second term of the
Hessian, giving

.. math::

   \mathbf{H}_{GN} = \mathbf{J}^T \mathbf{J}

The Gauss-Newton step is then obtained by solving

.. math::

   \mathbf{H}_{GN} \Delta \mathbf{p} = -\mathbf{g}.

When :math:`\mathbf{J}^T \mathbf{J}` is invertible, this can formally be written as

.. math::

   \Delta \mathbf{p} = -(\mathbf{J}^T \mathbf{J})^{-1}\mathbf{J}^T\mathbf{r}.


Robust least squares optimization by the Gauss-Newton Method
-------------------------------------------------------------

Robust cost functions :math:`\rho : \mathbb{R} \rightarrow \mathbb{R}` can be used to reduce the influence of outliers.
In that case, the standard least squares cost is replaced by a robust cost function
that reduces the contribution of residuals with large errors.

Consider a robust least squares problem of the form:

.. math::

   \min_{\mathbf{p}} \frac{1}{2} \sum_{j \in ( 1, n_r )}
   \rho \left(\| \mathbf{r}_j(\mathbf{p}) \|^2\right)

where :math:`\mathbf{r}_j(\mathbf{p})` is the :math:`j`-th residual function (:math:`\mathbb{R}^{n_p} \rightarrow \mathbb{R}`)
depending on the parameters :math:`\mathbf{p} \in \mathbb{R}^{n_p}`.

The update :math:`\Delta \mathbf{p}` to the current parameters :math:`\mathbf{p_k}`
is searched to minimize the cost function and the solution is given by:

.. math::

   \Delta \mathbf{p} = - \left(\tilde{\mathbf{J}}^T \tilde{\mathbf{J}}\right)^{-1} \tilde{\mathbf{J}}^T \tilde{\mathbf{r}}

Where :

.. math::

   \tilde{\mathbf{J}} = \sqrt{W_J} \mathbf{J} \quad \tilde{\mathbf{r}} = \frac{W_R}{\sqrt{W_J}} \mathbf{r}

The :math:`W` factors are given by the following equations evaluated at :math:`\mathbf{p_k}`:

.. math::

   W_J = \text{diag}\left(\rho'(|\mathbf{r}_j|^2) + 2 \rho''(|\mathbf{r}_j|^2) |\mathbf{r}_j|^2\right)
 
.. math::

   W_R = \text{diag}\left(\rho'(|\mathbf{r}_j|^2)\right)

.. seealso::

   The class :class:`pysolvegn.Term` to represent a least square problem. 

   In addition to the ``Callable`` to compute the operators, this class stores
   the robust cost function to use for the optimization problem.
   This function returns the values of :math:`\rho`, :math:`\rho^{(1)}`
   and :math:`\rho^{(2)}` (as a three 1D-array with shape ``(n_residuals,)``)
   from the squared of the residual vector :math:`\|\mathbf{r}\|^2` (as a 1D-array with shape ``(n_residuals,)``).
   

Demonstration
~~~~~~~~~~~~~~

Consider a robust least squares problem of the form:

.. math::

   \min_{\mathbf{p}} \frac{1}{2} \sum_{j \in ( 1, n_r )}
   \rho \left(\| \mathbf{r}_j(\mathbf{p}) \|^2\right)

where :math:`\mathbf{r}_j(\mathbf{p})` is the :math:`j`-th scalar residual and
:math:`\rho` is the robust cost function.

Developing the residuals :math:`\mathbf{r}_j(\mathbf{p})` around the current parameters
:math:`\mathbf{p_k}` gives:

.. math::

   \mathbf{r}_j(\mathbf{p_k} + \Delta \mathbf{p})
   \approx
   \mathbf{r}_j
   + \mathbf{J}_j \Delta \mathbf{p}
   + \frac{1}{2}
   \Delta \mathbf{p}^T \mathbf{H}_j \Delta \mathbf{p}
   + \ldots

where :math:`\mathbf{J}_j(\mathbf{p_k}) = \nabla \mathbf{r}_j(\mathbf{p_k})` is the
Jacobian of the :math:`j`-th residual with respect to the parameters, and
:math:`\mathbf{H}_j(\mathbf{p_k})` is its Hessian. By convention, all
quantities are considered evaluated at :math:`\mathbf{p_k}`, such that
:math:`\mathbf{r}_j = \mathbf{r}_j(\mathbf{p_k})`.

Let :math:`Z_j = \| \mathbf{r}_j \|^2` be the squared residual at the current parameters.
The robust cost function can then be developed around :math:`Z_j`:

.. math::

   \rho(Z_j + \delta Z_j) \approx \rho(Z_j) + \rho'(Z_j)\delta Z_j + \frac{1}{2} \rho''(Z_j) \delta Z_j^2 + \ldots

where :math:`\delta Z_j` is the change in the squared residual resulting
from the parameter update:

.. math::

   \delta Z_j = \| \mathbf{r}_j(\mathbf{p_k} + \Delta \mathbf{p}) \|^2 - \| \mathbf{r}_j(\mathbf{p_k}) \|^2

Using the Taylor expansion of the residual and retaining terms up to
second order in :math:`\Delta \mathbf{p}`:

.. math::

   \delta Z_j \approx 2 \mathbf{r}_j^T \mathbf{J}_j \Delta \mathbf{p} + \Delta \mathbf{p}^T \left( \mathbf{J}_j^T \mathbf{J}_j + \mathbf{r}_j^T \mathbf{H}_j \right) \Delta \mathbf{p} + \ldots

In a similar way, the squared term :math:`\delta Z_j^2` can be approximated as:

.. math::

   \delta Z_j^2 \approx 4 \left( \mathbf{r}_j^T \mathbf{J}_j \Delta \mathbf{p} \right)^2 + \ldots

Substituting these expressions into the Taylor expansion of the robust
cost function gives:

.. math::

    \rho(Z_j + \delta Z_j) \approx
    \rho(Z_j) + 
    \rho'(Z_j) \Big[ \Delta \mathbf{p}^T \left( \mathbf{J}_j^T \mathbf{J}_j + \mathbf{r}_j^T \mathbf{H}_j \right) \Delta \mathbf{p} + 2 \mathbf{r}_j^T \mathbf{J}_j \Delta \mathbf{p} \Big] + 
    2 \rho''(Z_j) \left( \mathbf{r}_j^T \mathbf{J}_j \Delta \mathbf{p} \right)^2 + \ldots

By summing over all residuals, the gradient and the Hessian of the robust cost with respect to the parameters is therefore:

.. math::

   \mathbf{g}  \approx \sum_j 2 \rho'(Z_j) \mathbf{J}_j^T \mathbf{r}_j

.. math::

   \mathbf{H} \approx \sum_j \Big[2 \rho'(Z_j) \left( \mathbf{J}_j^T \mathbf{J}_j + \mathbf{r}_j^T \mathbf{H}_j \right) + 4 \rho''(Z_j) \mathbf{J}_j^T \mathbf{r}_j \mathbf{r}_j^T \mathbf{J}_j \Big]

The Gauss-Newton approximation consists in neglecting the second-order
derivatives of the residuals. The term involving :math:`r_j \mathbf{H}_j`
is neglected.

.. math::

   \mathbf{H} \approx \sum_j \Big[2 \rho'(Z_j) \mathbf{J}_j^T \mathbf{J}_j + 4 \rho''(Z_j) \mathbf{J}_j^T \mathbf{r}_j \mathbf{r}_j^T \mathbf{J}_j \Big]

.. math::

   \mathbf{H} \approx \sum_j 2 \mathbf{J}_j^T \Big[ \rho'(Z_j) + 2 \rho''(Z_j) Z_j \Big] \mathbf{J}_j

Finally, defining a diagonal matrix :math:`W_J` and a vector :math:`W_R`, the Hessian and gradient approximation
can be written as:

.. math::

   \mathbf{H} = \mathbf{J}^T \sqrt{W_J}^T \sqrt{W_J} \mathbf{J} \quad  \mathbf{g} = W_R \mathbf{J}^T \mathbf{r}

where:

.. math::

   W_J = \text{diag}\left(\rho'(|\mathbf{r}_j|^2) + 2 \rho''(|\mathbf{r}_j|^2) |\mathbf{r}_j|^2\right)
 
.. math::

   W_R = \text{diag}\left(\rho'(|\mathbf{r}_j|^2)\right)

This system can be expressed as a standard least squares system by
introducing a modified Jacobian and a modified residual:

.. math::

   \tilde{\mathbf{J}} = \sqrt{W_J}\mathbf{J} \quad \tilde{\mathbf{r}} = W_J^{-1/2} W_R \mathbf{r}.

Such that:

.. math::

   \tilde{\mathbf{J}}^T \tilde{\mathbf{J}} \Delta \mathbf{p} = -\tilde{\mathbf{J}}^T \tilde{\mathbf{r}}.



Non-positive curvature weights: fallback on :math:`\rho'`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The construction above requires :math:`\sqrt{W_J}`, so every diagonal element

.. math::

   W_{J,j} = \rho'(Z_j) + 2 \rho''(Z_j) Z_j

must be strictly positive. This always holds for a convex loss function, but
not for the loss functions that are designed to reject outliers. For these
loss functions, :math:`\rho''` is negative and dominates :math:`\rho'` for large
residuals:

.. list-table::
   :header-rows: 1
   :widths: 15 30 35 20

   * - Loss
     - :math:`\rho'(Z)`
     - :math:`W_J(Z) = \rho'(Z) + 2\rho''(Z) Z`
     - :math:`W_J \leq 0` when
   * - ``"linear"``
     - :math:`1`
     - :math:`1`
     - never
   * - ``"soft_l1"``
     - :math:`(1 + Z)^{-1/2}`
     - :math:`(1 + Z)^{-3/2}`
     - never
   * - ``"huber"``
     - :math:`1` if :math:`Z \leq 1`, :math:`Z^{-1/2}` otherwise
     - :math:`1` if :math:`Z \leq 1`, :math:`0` otherwise
     - :math:`Z > 1`
   * - ``"cauchy"``
     - :math:`(1 + Z)^{-1}`
     - :math:`(1 - Z)(1 + Z)^{-2}`
     - :math:`Z \geq 1`
   * - ``"arctan"``
     - :math:`(1 + Z^2)^{-1}`
     - :math:`(1 - 3Z^2)(1 + Z^2)^{-2}`
     - :math:`Z \geq 1/\sqrt{3}`
   * - ``"tukey"``
     - :math:`(1 - Z)^2` if :math:`Z \leq 1`, :math:`0` otherwise
     - :math:`(1 - Z)(1 - 5Z)` if :math:`Z \leq 1`, :math:`0` otherwise
     - :math:`Z \geq 1/5`

A negative :math:`W_{J,j}` means that the cost of the :math:`j`-th residual is
locally concave: the corresponding contribution
:math:`W_{J,j} \mathbf{J}_j^T \mathbf{J}_j` to the Hessian is negative
semi-definite, and the total Hessian may become indefinite (the step is then no
longer guaranteed to be a descent direction).

Simply clamping :math:`W_{J,j}` to a small value :math:`\varepsilon` is not a good
solution. The gradient stays exact, but the contribution of the residual to the
Hessian becomes :math:`\varepsilon \mathbf{J}_j^T \mathbf{J}_j \approx 0`. When
many residuals are in this regime (e.g. a poor initial guess, or residuals
expressed in a unit much larger than the threshold of the loss), the Hessian
becomes almost singular and the step explodes.

Instead, the second-order correction of the loss is dropped for these residuals
(as in Ceres Solver):

.. math::

   W_{J,j} =
   \begin{cases}
      \rho'(Z_j) + 2 \rho''(Z_j) Z_j & \text{if } \rho'(Z_j) + 2 \rho''(Z_j) Z_j > \varepsilon \\
      \max\left(\rho'(Z_j), 0\right) & \text{otherwise}
   \end{cases}

with :math:`\varepsilon = 10^{-12}`. In the fallback case, the modified residual and
Jacobian become:

.. math::

   \tilde{\mathbf{r}}_j = \frac{\rho'(Z_j)}{\sqrt{\rho'(Z_j)}} \mathbf{r}_j = \sqrt{\rho'(Z_j)} \, \mathbf{r}_j
   \quad
   \tilde{\mathbf{J}}_j = \sqrt{\rho'(Z_j)} \, \mathbf{J}_j

which is exactly the classical *Iteratively Reweighted Least Squares* (IRLS)
weighting: the residual is treated as a standard least squares residual with the
weight :math:`\rho'(Z_j)`. This choice has three properties:

- **The gradient is unchanged**: in both cases
  :math:`\tilde{\mathbf{J}}_j^T \tilde{\mathbf{r}}_j = \rho'(Z_j) \mathbf{J}_j^T \mathbf{r}_j`,
  so the stationary points of the problem are the same.
- **The Hessian approximation stays positive semi-definite**: each contribution
  :math:`W_{J,j} \mathbf{J}_j^T \mathbf{J}_j` uses :math:`W_{J,j} \geq 0`. When it is
  invertible, the step :math:`\Delta \mathbf{p} = -\mathbf{H}^{-1} \mathbf{g}` is a
  descent direction (:math:`\mathbf{g}^T \Delta \mathbf{p} < 0`).
- **The residual keeps a curvature proportional to its influence**: an outlier with
  a small :math:`\rho'(Z_j)` contributes little to both the gradient and the Hessian,
  instead of contributing to the gradient only.

If :math:`\rho'(Z_j) \leq 0` too (e.g. ``"tukey"`` beyond its threshold, where
:math:`\rho' = 0`), then :math:`W_{J,j} = 0` and the residual is simply ignored:
:math:`\tilde{\mathbf{r}}_j = 0` and :math:`\tilde{\mathbf{J}}_j = 0`.

.. warning::

   With a redescending loss such as ``"tukey"``, if all the residuals are beyond the
   threshold, the gradient and the Hessian are both zero and the system is singular.
   These loss functions require a good initial guess (e.g. the solution obtained
   with ``"cauchy"`` or ``"huber"``).

.. note::

   The fallback only modifies the Hessian approximation, never the gradient or the
   cost. The Levenberg-Marquardt damping (``damping="lm"`` or ``"lm-diag"``) remains
   useful on top of it to control the step length far from the solution.


Scaling a problem
-----------------

All the predefined loss functions have their transition between the quadratic
regime (:math:`\rho(Z) \approx Z`, inliers) and the robust regime (outliers) around
:math:`Z = 1`, that is :math:`|\mathbf{r}_j| \approx 1` **in the unit of the residuals**.

The result of a robust optimization therefore depends on the unit of the residuals.
For example, for residuals expressed in millimeters with a noise of a few
millimeters, all the residuals are in the robust regime of the loss function: every
residual is treated as an outlier and the optimization can fail or converge to a
poor solution. The same problem expressed in meters works as expected.

To control the threshold, a **soft threshold** :math:`C > 0` (``loss_scale``),
expressed in the unit of the residuals, is introduced. The loss function
:math:`\rho` is replaced by its scaled version:

.. math::

   \rho_C(Z) = C^2 \rho\left(\frac{Z}{C^2}\right)

The transition is now located at :math:`Z = C^2`, that is
:math:`|\mathbf{r}_j| \approx C`. This definition is the same as the
``f_scale`` argument of ``scipy.optimize.least_squares``.

Derivatives
~~~~~~~~~~~

Denoting :math:`u_j = Z_j / C^2` the dimensionless squared residual, the derivatives
of the scaled loss function are:

.. math::

   \rho_C'(Z_j) = \rho'(u_j)
   \quad
   \rho_C''(Z_j) = \frac{1}{C^2} \rho''(u_j)

The weights of the robust Gauss-Newton system become:

.. math::

   W_{R,j} = \rho'(u_j)
   \quad
   W_{J,j} = \rho'(u_j) + 2 \frac{\rho''(u_j)}{C^2} Z_j = \rho'(u_j) + 2 \rho''(u_j) u_j

Both weights are dimensionless and only depend on the ratio
:math:`|\mathbf{r}_j| / C`. The fallback on :math:`\rho'` described in the previous
section is applied in the same way on :math:`W_{J,j}`.

Properties
~~~~~~~~~~

- **Small residuals are unchanged.** Since all the predefined loss functions satisfy
  :math:`\rho(0) = 0` and :math:`\rho'(0) = 1`, for :math:`|\mathbf{r}_j| \ll C`:

  .. math::

     \rho_C(Z_j) \approx C^2 \frac{Z_j}{C^2} = Z_j

  The inliers are treated as in a standard least squares problem, whatever :math:`C`.

- **The linear loss is invariant.** For :math:`\rho(Z) = Z`,
  :math:`\rho_C(Z) = Z` for every :math:`C`.

- **Unit invariance.** If the residuals are multiplied by a factor :math:`\alpha`
  (change of unit) and :math:`C` is multiplied by the same factor, then
  :math:`u_j` is unchanged and the cost is multiplied by :math:`\alpha^2`:

  .. math::

     \rho_{\alpha C}(\alpha^2 Z_j) = \alpha^2 \rho_C(Z_j)

  The minimizer is therefore the same: the solution does not depend on the unit of
  the residuals, as long as :math:`C` is expressed in the same unit.

- **Equivalence with a weighted problem.** Since

  .. math::

     \rho_C\left(\|\mathbf{r}_j\|^2\right)
     = C^2 \rho\left(\left\|\frac{\mathbf{r}_j}{C}\right\|^2\right)

  a term with the loss :math:`\rho_C` and the weight :math:`w_i` is equivalent to a
  term with the loss :math:`\rho`, the residuals :math:`\mathbf{r}_i / C`, the Jacobian
  :math:`\mathbf{J}_i / C` and the weight :math:`w_i C^2`. ``loss_scale`` performs this
  normalization automatically, without modifying the residual functions or the weight.

- **Any loss function can be scaled.** The scaling only uses the values of
  :math:`\rho`, :math:`\rho'` and :math:`\rho''`, so it applies in the same way to
  the predefined loss functions and to a custom loss function.

.. tip::

   A good starting value for :math:`C` is the expected magnitude of the inlier
   residuals, e.g. two or three times the standard deviation of the measurement noise.
   Residuals larger than a few :math:`C` are then progressively considered as outliers.

.. seealso::

   The ``loss_scale`` argument of :class:`pysolvegn.Term` (and
   :class:`pysolvegn.BatchTerm`), which stores the soft threshold :math:`C` and
   replaces the loss function :math:`\rho` (predefined or custom) by :math:`\rho_C`.
   The same :math:`C` is used for all the residuals of a term, so terms with
   residuals in different units should use different values.


Adding regularization to the Gauss-Newton update
------------------------------------------------

In many optimization problems, the objective function is composed of several
least squares terms. The first term usually corresponds to the data fitting
problem, while the remaining terms can be used to regularize the solution.

The resulting objective function can be written as:

.. math::

   \min_{\mathbf{p}}
   \frac{1}{2}
   \sum_{i}
   w_i
   \sum_{j}
   \rho_i
   \left(
      \| \mathbf{r}_{i,j}(\mathbf{p}) \|^2
   \right)

where :math:`i` indexes the different least squares terms and :math:`j`
indexes the residuals within each term.

The scalar :math:`w_i` is the weight associated with the :math:`i`-th term,
and :math:`\rho_i` is its robust cost function. Each term can therefore have
its own residuals, weight, and robust cost function.

By convention, the first term (:math:`i=0`) is the main least squares term,
which usually contains the data residuals. The remaining terms
(:math:`i \geq 1`) are considered regularization terms.

.. note::

   A regularization term can be used to introduce additional constraints or
   prior information into the optimization problem. Its weight :math:`w_i`
   controls its influence relative to the other terms.

For each term, the robust Gauss-Newton approximation described above provides
a modified residual :math:`\tilde{\mathbf{r}}_i` and a modified Jacobian
:math:`\tilde{\mathbf{J}}_i`.

Since the objective function is the sum of all the terms, their contributions
to the gradient and Hessian approximation can be summed directly. The global
Gauss-Newton system is therefore:

.. math::

   \left(
      \sum_i
      w_i
      \tilde{\mathbf{J}}_i^T
      \tilde{\mathbf{J}}_i
   \right)
   \Delta \mathbf{p}
   =
   -
   \sum_i
   w_i
   \tilde{\mathbf{J}}_i^T
   \tilde{\mathbf{r}}_i.

.. seealso::

   The class :class:`pysolvegn.Term` to represent a term in a least square problem. 

   In addition to the ``Callable`` to compute the operators, this class stores
   the weight associated to the term.



Changing the parametrization
----------------------------

Consider a least squares problem of the form:

.. math::

   \min_{\mathbf{p}_{\mathrm{out}}}
   \frac{1}{2}
   \sum_i w_i \sum_j
   \rho_i
   \left(
      \| \mathbf{r}_{i,j}(\mathbf{p}_{\mathrm{out}}) \|^2
   \right)

In some applications, it is convenient to optimize the problem using a
different set of parameters than those expected by the residual functions.
This can be achieved by introducing a parametrization that maps the input
parameters :math:`\mathbf{p}_{\mathrm{in}}` to the output parameters
:math:`\mathbf{p}_{\mathrm{out}}`:

.. math::

   \mathbf{p}_{\mathrm{out}}
   =
   P(\mathbf{p}_{\mathrm{in}})

where :math:`\mathbf{p}_{\mathrm{in}} \in \mathbb{R}^{n_{p_{\mathrm{in}}}}`
are the parameters optimized by the Gauss-Newton algorithm, while
:math:`\mathbf{p}_{\mathrm{out}} \in \mathbb{R}^{n_{p_{\mathrm{out}}}}`
are the parameters passed to the residual functions.

The optimization problem can therefore be rewritten directly in terms of
the input parameters:

.. math::

   \min_{\mathbf{p}_{\mathrm{in}}}
   \frac{1}{2}
   \sum_i w_i \sum_j
   \rho_i
   \left(
      \left\|
      \mathbf{r}_{i,j}
      \left(
         P(\mathbf{p}_{\mathrm{in}})
      \right)
      \right\|^2
   \right)

At each iteration, the Gauss-Newton algorithm searches for an update
:math:`\Delta\mathbf{p}_{\mathrm{in}}` in the space of input parameters.
The corresponding output parameters are obtained through the parametrization:

.. math::

   \mathbf{p}_{\mathrm{out}}
   =
   P(\mathbf{p}_{\mathrm{in}}).

To compute the Jacobian of the residuals with respect to the parameters
being optimized, we introduce the Jacobian of the parametrization:

.. math::

   \mathbf{J}_P
   =
   \frac{\partial \mathbf{p}_{\mathrm{out}}}
        {\partial \mathbf{p}_{\mathrm{in}}}

Using the chain rule, the Jacobian of the :math:`j`-th residual of the
:math:`i`-th term with respect to the input parameters is:

.. math::

   \mathbf{J}_{i,j,\mathrm{in}}
   =
   \frac{\partial \mathbf{r}_{i,j}}
        {\partial \mathbf{p}_{\mathrm{in}}}
   =
   \frac{\partial \mathbf{r}_{i,j}}
        {\partial \mathbf{p}_{\mathrm{out}}}
   \frac{\partial \mathbf{p}_{\mathrm{out}}}
        {\partial \mathbf{p}_{\mathrm{in}}}

   =
   \mathbf{J}_{i,j}
   \mathbf{J}_P

Therefore, the parametrization only modifies the Jacobian used by the
Gauss-Newton algorithm. The robust weighting described in the previous
section is applied in exactly the same way, giving:

.. math::

   \tilde{\mathbf{J}}_i
   =
   \sqrt{W_{J,i}}
   \mathbf{J}_i
   \mathbf{J}_P

while the modified residual remains:

.. math::

   \tilde{\mathbf{r}}_i
   =
   W_{J,i}^{-1/2}
   W_{R,i}
   \mathbf{r}_i

The Gauss-Newton system is consequently solved in the input parameter
space:

.. math::

   \left(
      \sum_i
      w_i
      \tilde{\mathbf{J}}_i^T
      \tilde{\mathbf{J}}_i
   \right)
   \Delta\mathbf{p}_{\mathrm{in}}
   =
   -
   \sum_i
   w_i
   \tilde{\mathbf{J}}_i^T
   \tilde{\mathbf{r}}_i

The important point is that :math:`\Delta\mathbf{p}_{\mathrm{in}}` has the
dimension of the input parameter space. The parametrization is therefore
accounted for through :math:`\mathbf{J}_P` when computing the Jacobians.

Once the Gauss-Newton system has been solved, the input parameters are
updated as:

.. math::

   \mathbf{p}_{\mathrm{in}}^{k+1}
   =
   \mathbf{p}_{\mathrm{in}}^k
   +
   \Delta\mathbf{p}_{\mathrm{in}}

The corresponding output parameters are then obtained by applying the
parametrization:

.. math::

   \mathbf{p}_{\mathrm{out}}^{k+1}
   =
   P
   \left(
      \mathbf{p}_{\mathrm{in}}^{k+1}
   \right)

This formulation separates the parameters used by the optimization
algorithm from those used by the residual functions. The Gauss-Newton
algorithm operates entirely in the input parameter space, while the
parametrization maps these parameters to the output space required by
the optimization terms.

.. seealso::

   The class :class:`pysolvegn.Parametrization` represents a parameter
   transformation by storing the functions required to compute the output
   parameters and the Jacobian of the transformation.

   Given the input parameters :math:`\mathbf{p}_{\mathrm{in}}` as a 1D-array
   with shape ``(n_parameters,)``, the parametrization computes the output
   parameters :math:`\mathbf{p}_{\mathrm{out}}` as a 1D-array with shape
   ``(n_p_outputs,)`` and the Jacobian :math:`\mathbf{J}_P` as a 2D-array
   with shape ``(n_p_outputs, n_parameters)``

Solving a batch of independent problems
---------------------------------------

Many applications require solving the same least squares problem for a large
number of independent datasets (e.g. one fit per pixel, per point or per image).
Consider :math:`K` independent problems sharing the same terms, indexed by
:math:`k \in (1, K)`:

.. math::

   \min_{\mathbf{p}_{\mathrm{in},k}}
   \frac{1}{2}
   \sum_i w_{i,k} \sum_j
   \rho_i
   \left(
      \left\| \mathbf{r}_{i,j}\left(P(\mathbf{p}_{\mathrm{in},k}), k\right) \right\|^2
   \right)
   \quad \forall k \in (1, K)

The residual functions, the robust cost functions :math:`\rho_i` and the
parametrization :math:`P` have the same form for all the problems (:math:`P` is
strictly identical for all of them), but each problem
has its own parameters :math:`\mathbf{p}_{\mathrm{in},k} \in \mathbb{R}^{n_p}`, its
own data (the residuals depend on :math:`k`) and possibly its own weights
:math:`w_{i,k}`.

Block-diagonal structure
~~~~~~~~~~~~~~~~~~~~~~~~

These :math:`K` problems could be stacked into a single problem with
:math:`K n_p` parameters. Since the residuals of the problem :math:`k` only depend
on :math:`\mathbf{p}_{\mathrm{in},k}`, the stacked Jacobian is block-diagonal, and so
is the Gauss-Newton system:

.. math::

   \begin{bmatrix}
      \mathbf{H}_1 & & \\
      & \ddots & \\
      & & \mathbf{H}_K
   \end{bmatrix}
   \begin{bmatrix}
      \Delta \mathbf{p}_{\mathrm{in},1} \\
      \vdots \\
      \Delta \mathbf{p}_{\mathrm{in},K}
   \end{bmatrix}
   =
   -
   \begin{bmatrix}
      \mathbf{g}_1 \\
      \vdots \\
      \mathbf{g}_K
   \end{bmatrix}

where, for each problem :math:`k`, the Hessian approximation and the gradient are
built exactly as in the previous sections (robust weighting, fallback on
:math:`\rho'`, loss scaling and parametrization):

.. math::

   \mathbf{H}_k = \sum_i w_{i,k} \, \tilde{\mathbf{J}}_{i,k}^T \tilde{\mathbf{J}}_{i,k}
   \quad
   \mathbf{g}_k = \sum_i w_{i,k} \, \tilde{\mathbf{J}}_{i,k}^T \tilde{\mathbf{r}}_{i,k}
   \quad
   \tilde{\mathbf{J}}_{i,k} = \sqrt{W_{J,i,k}} \, \mathbf{J}_{i,k} \mathbf{J}_{P,k}

The global system therefore decouples into :math:`K` small independent systems of
size :math:`n_p \times n_p`:

.. math::

   \mathbf{H}_k \Delta \mathbf{p}_{\mathrm{in},k} = -\mathbf{g}_k
   \quad \forall k \in (1, K)

Instead of one system of size :math:`K n_p`, the batch solver solves :math:`K`
systems of size :math:`n_p` in a single vectorized operation. The cost of an
iteration is linear in :math:`K`, and all the quantities are stored as arrays with
a leading batch dimension:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Quantity
     - Shape
   * - Parameters :math:`\mathbf{p}_{\mathrm{in}}`
     - ``(K, n_parameters)``
   * - Residuals :math:`\mathbf{r}_i`
     - ``(K, n_residuals_i)``
   * - Jacobians :math:`\mathbf{J}_i`
     - ``(K, n_residuals_i, n_p_outputs)``
   * - Jacobian of the parametrization :math:`\mathbf{J}_P`
     - ``(K, n_p_outputs, n_parameters)``
   * - Gradients :math:`\mathbf{g}`
     - ``(K, n_parameters)``
   * - Hessians :math:`\mathbf{H}`
     - ``(K, n_parameters, n_parameters)``
   * - Weights :math:`w_i`
     - scalar, or ``(K,)`` for one weight per problem

Independent convergence
~~~~~~~~~~~~~~~~~~~~~~~

Since the problems are independent, each one converges at its own pace: an easy
problem may converge in a few iterations, while a difficult one requires many more.
The stopping criteria (``ftol``, ``atol``, ``gtol``, ``xtol``, ``ptol``) are therefore
evaluated **for each problem separately**, using its own cost :math:`F_k`, gradient
:math:`\mathbf{g}_k` and update :math:`\Delta \mathbf{p}_{\mathrm{in},k}`. For instance,
the problem :math:`k` satisfies ``ftol`` when:

.. math::

   |F_k^{(n)} - F_k^{(n-1)}| < \mathrm{ftol} \cdot F_k^{(n)}

The solver maintains the set :math:`\mathcal{A}^{(n)}` of the problems still in
processing at iteration :math:`n` (the *active set*), with
:math:`m = |\mathcal{A}^{(n)}| \leq K`:

.. math::

   \mathcal{A}^{(0)} = (1, K)
   \quad
   \mathcal{A}^{(n+1)} = \mathcal{A}^{(n)} \setminus \left\{ k \text{ stopped at iteration } n \right\}

At each iteration, only the :math:`m` active problems are evaluated, and their
parameters are updated:

.. math::

   \mathbf{p}_{\mathrm{in},k}^{(n+1)} =
   \begin{cases}
      \mathbf{p}_{\mathrm{in},k}^{(n)} + \Delta \mathbf{p}_{\mathrm{in},k} & \text{if } k \in \mathcal{A}^{(n+1)} \\
      \mathbf{p}_{\mathrm{in},k}^{(n)} & \text{otherwise (frozen)}
   \end{cases}

This has two consequences:

- **The converged problems are not modified anymore.** Their parameters are frozen
  at the iteration at which they satisfied a criterion, exactly as if each problem
  had been solved alone with :func:`pysolvegn.solve`.
- **The work decreases during the optimization.** The cost of an iteration is
  proportional to the number :math:`m` of active problems, not to :math:`K`.

A problem can also be stopped without being converged: when its system is singular,
when NaN or Inf values appear in its parameters, or when a global limit is reached
(``max_iteration``, ``max_time`` or the callback), which stops all the remaining
active problems. The solver reports, for each problem, whether it has converged.

.. note::

   The global cost :math:`\sum_k F_k` has no particular meaning for independent
   problems (a single badly fitted problem dominates it). The logged values are
   therefore averaged over the active problems.

Evaluating a subset of problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because of the active set, the callables of the terms are called with the parameters
of the :math:`m` active problems only, together with their indices in the batch:

.. math::

   \left(
      \begin{bmatrix}
         \mathbf{p}_{k_1} \\
         \vdots \\
         \mathbf{p}_{k_m}
      \end{bmatrix},
      \begin{bmatrix}
         k_1 \\
         \vdots \\
         k_m
      \end{bmatrix}
   \right)
   \longmapsto
   \begin{bmatrix}
      \mathbf{r}(\mathbf{p}_{k_1}, k_1) \\
      \vdots \\
      \mathbf{r}(\mathbf{p}_{k_m}, k_m)
   \end{bmatrix}

The row :math:`a` of every output must only depend on the row :math:`a` of the
parameters and on the data of the problem :math:`k_a` (selected with the indices,
e.g. ``observations[indices]``). The number :math:`m` changes from one call to the
next, so the callables must never assume :math:`m = K`.

The parametrization :math:`P` is the same for all the problems: it does not depend on
:math:`k`, so its callables only receive the parameters of the :math:`m` active problems
(without indices) and are applied row by row:

.. math::

   \begin{bmatrix}
      \mathbf{p}_{\mathrm{in},k_1} \\
      \vdots \\
      \mathbf{p}_{\mathrm{in},k_m}
   \end{bmatrix}
   \longmapsto
   \begin{bmatrix}
      P(\mathbf{p}_{\mathrm{in},k_1}) \\
      \vdots \\
      P(\mathbf{p}_{\mathrm{in},k_m})
   \end{bmatrix}

The same convention is used for the vector weights: the weights of the active
problems are :math:`(w_{i,k_1}, \ldots, w_{i,k_m})`. The weights must be strictly
positive: a zero weight would cancel all the contributions of a problem and make
its system :math:`\mathbf{H}_k` singular.

.. note::

   The loss scale :math:`C_i` (``loss_scale``) of a term is shared by all the
   problems of the batch. Problems whose residuals have very different magnitudes
   should be normalized in the residual functions (e.g. by dividing by a
   per-problem noise level selected with the indices).

.. seealso::

   The function :func:`pysolvegn.solve_batch` solves a batch of independent problems.

   The class :class:`pysolvegn.BatchTerm` represents a term of the batch. Its
   ``Callable`` take the parameters as a 2D-array with shape ``(m, n_parameters)``
   and the indices as a 1D-array with shape ``(m,)``, and return the residuals as
   a 2D-array with shape ``(m, n_residuals)`` and the Jacobians as a 3D-array with
   shape ``(m, n_residuals, n_parameters)``. Its weight is either a scalar or a
   1D-array with shape ``(K,)``.

   The class :class:`pysolvegn.BatchParametrization` represents the parameter
   transformation :math:`P` of the batch. The same transformation is applied to all
   the problems: its ``Callable`` only take the input parameters as a 2D-array with
   shape ``(m, n_parameters)`` (no indices) and return the output parameters with shape
   ``(m, n_p_outputs)`` and the Jacobians :math:`\mathbf{J}_{P,k}` with shape
   ``(m, n_p_outputs, n_parameters)``.