Mathematical Background
=======================

.. contents:: Table of Contents
   :local:
   :depth: 2


Robust Least Squares Optimization by the Gauss-Newton Method
-------------------------------------------------------------

Consider a least squares problem of the form:

.. math::

   \min_{\mathbf{p}} \frac{1}{2} \sum_j \rho\left(\| \mathbf{r}_j(\mathbf{p}) \|^2\right)

where :math:`\mathbf{r}_j(\mathbf{p})` is a residual function (:math:`\mathbb{R}^{n_p} \rightarrow \mathbb{R}`)
depending on the parameters :math:`\mathbf{p} \in \mathbb{R}^{n_p}`, and :math:`\rho` is a robust 
cost function (:math:`\mathbb{R} \rightarrow \mathbb{R}`) that reduces the influence of outliers.

At each iteration of the Gauss-Newton method, we search for an update :math:`\Delta p` 
to the current parameters :math:`\mathbf{p_k}` that minimizes the robust cost function.

The solution is given by:

.. math::

   \Delta p = - \left(\tilde{\mathbf{J}}^T \tilde{\mathbf{J}}\right)^{-1} \tilde{\mathbf{J}}^T \tilde{\mathbf{r}}

Where :

.. math::

   \tilde{\mathbf{J}} = \sqrt{W_J} \mathbf{J} \quad \tilde{\mathbf{r}} = \frac{W_R}{\sqrt{W_J}} \mathbf{r}

Where :math:`\mathbf{r}` is the full residual vector in :math:`\mathbb{R}^{n_r}` containing each :math:`\mathbf{r}_j` evaluated at :math:`\mathbf{p_k}`
and :math:`\mathbf{J}` is the full jacobian matrix in :math:`\mathbb{M}_{n_r, n_p}(\mathbb{R})` containing each :math:`\mathbf{J}_j = \nabla \mathbf{r}_j`
evaluated at :math:`\mathbf{p_k}`. The :math:`W` factors are given by the following equations evaluated at :math:`\mathbf{p_k}`:

.. math::

   W_J = \text{diag}\left(\rho'(|\mathbf{r}_j|^2) + 2 \rho''(|\mathbf{r}_j|^2) |\mathbf{r}_j|^2\right)
 
.. math::

   W_R = \text{diag}\left(\rho'(|\mathbf{r}_j|^2)\right)

.. seealso::

   The class :class:`pysolvegn.Term` to represent each term in the least squares problem by storing 
   the functions to compute from :math:`\mathbf{p}` (as a 1D-array with shape ``(n_parameters,)``) 
   the residual vector :math:`\mathbf{r}` (as a 1D-array with shape ``(n_residual,)``) and 
   jacobian matrix :math:`\mathbf{J}` (as a 2D-array with shape ``(n_residual, n_parameters)``).


Demonstration
~~~~~~~~~~~~~~

Consider a least squares problem of the form:

.. math::

   \min_{\mathbf{p}} \frac{1}{2} \sum_j \rho\left(\| \mathbf{r}_j(\mathbf{p}) \|^2\right)

Developing the residuals :math:`\mathbf{r}_j(\mathbf{p})` around the current parameters :math:`\mathbf{p_k}`:

.. math::

   \mathbf{r}_j(\mathbf{p_k} + \Delta p) \approx \mathbf{r}_j + \mathbf{J}_j \Delta p + 0.5 \Delta p^T \mathbf{H}_j \Delta p + \ldots

where :math:`\mathbf{J}_j(\mathbf{p_k}) = \nabla \mathbf{r}_j(\mathbf{p_k})` is the Jacobian 
of the residuals with respect to the parameters, and :math:`\mathbf{H}_j(\mathbf{p_k})` 
is the Hessian of the residuals. By convention, the quantities are considered evaluated 
at :math:`\mathbf{p_k}` such that :math:`\mathbf{r}_j = \mathbf{r}_j(\mathbf{p_k})`

Let :math:`Z_j = \| \mathbf{r}_j \|^2` be the squared 
norm of the residuals at the current parameters.
The robust cost function can be developed around :math:`Z_j`:

.. math::

   \rho(Z_j + \delta Z_j) \approx \rho(Z_j) + \rho'(Z_j)\delta Z_j + 0.5 \rho''(Z_j) \delta Z_j^2 + \ldots

where :math:`\delta Z_j` is the change in the squared norm due to the parameter update:

.. math::

   \delta Z_j = \| \mathbf{r}_j(\mathbf{p_k} + \Delta p) \|^2 - \| \mathbf{r}_j(\mathbf{p_k}) \|^2

By only keeping up to the second order terms, we can approximate :math:`\delta Z_j` as:

.. math::

   \delta Z_j \approx \Delta p^T \left( \mathbf{J}_j^T \mathbf{J}_j + \mathbf{r}_j^T \mathbf{H}_j \right) \Delta p + 2 \mathbf{r}_j^T \mathbf{J}_j \Delta p + ...

In a similar way, the squared term :math:`\delta Z_j^2` can be approximated as:

.. math::

   \delta Z_j^2 \approx 4 \left( \mathbf{r}_j^T \mathbf{J}_j \Delta p \right)^2 + ...

Thus, by injecting all the approximations into the robust cost function, 
we optain :

.. math::

    \rho(Z_j + \delta Z_j) \approx
    \rho(Z_j) + 
    \rho'(Z_j) \Big[ \Delta p^T \left( \mathbf{J}_j^T \mathbf{J}_j + \mathbf{r}_j^T \mathbf{H}_j \right) \Delta p + 2 \mathbf{r}_j^T \mathbf{J}_j \Delta p \Big] + 
    2 \rho''(Z_j) \left( \mathbf{r}_j^T \mathbf{J}_j \Delta p \right)^2 + ...

With Gauss-Newton, we shearch for the update :math:`\Delta p` that minimizes 
the robust cost function, which is equivalent to solving the zero of the gradient
of the robust cost function with respect to the parameters.

.. math::

    \nabla_{\Delta p} \rho(Z_j + \delta Z_j) = 0

The Gauss-Newton update can be obtained by solving the following linear system:

.. math::

    \mathbf{H} \, \Delta p = - \mathbf{g}

Where :math:`\mathbf{g}` is the gradient of the robust cost function with respect to the 
parameters, and :math:`\mathbf{H}` is the Hessian approximation of the robust cost function 
with respect to the parameters.

By summing over all residuals, the gradient and Hessian approximation are:

.. math::

   \mathbf{g}  \approx \sum_j 2 \rho'(Z_j) \mathbf{J}_j^T \mathbf{r}_j

.. math::

   \mathbf{H} \approx \sum_j \Big[2 \rho'(Z_j) \left( \mathbf{J}_j^T \mathbf{J}_j + \mathbf{r}_j^T \mathbf{H}_j \right) + 4 \rho''(Z_j) \mathbf{J}_j^T \mathbf{r}_j \mathbf{r}_j^T \mathbf{J}_j \Big]

Finally, the Gauss-Newton suggested to ignore the second order term of the residuals, 
as at convergence, the residuals should be small, and thus the second order term 
:math:`\mathbf{r}_j^T \mathbf{H}_j` should be negligible compared to the first order 
term :math:`\mathbf{J}_j^T \mathbf{J}_j`.

.. math::

   \mathbf{H} \approx \sum_j \Big[2 \rho'(Z_j) \mathbf{J}_j^T \mathbf{J}_j + 4 \rho''(Z_j) \mathbf{J}_j^T \mathbf{r}_j \mathbf{r}_j^T \mathbf{J}_j \Big]

.. math::

   \mathbf{H} \approx \sum_j 2 \mathbf{J}_j^T \Big[ \rho'(Z_j) + 2 \rho''(Z_j) Z_j \Big] \mathbf{J}_j

Finally, we can build a diagonal matrix :math:`W_J` and a vector :math:`W_R` that depend on the 
robust cost function and the squared norm of the residuals to solve the complete linear system
as:

.. math::

   \mathbf{J}^T \sqrt{W_J}^T \sqrt{W_J} \mathbf{J} \Delta p = - W_R \mathbf{J}^T \mathbf{r}

.. math::

   W_J = \text{diag}\left(\rho'(|\mathbf{r}_j|^2) + 2 \rho''(|\mathbf{r}_j|^2) |\mathbf{r}_j|^2\right)
 
.. math::

   W_R = \text{diag}\left(\rho'(|\mathbf{r}_j|^2)\right)

We observe that the system is similar to solve:

.. math::

   \tilde{\mathbf{J}}^T \tilde{\mathbf{J}} \Delta p = - \tilde{\mathbf{J}}^T \tilde{\mathbf{r}}

With a modified Jacobian and a modified residual :

.. math::

   \tilde{\mathbf{J}} = \sqrt{W_J} \mathbf{J} \quad \tilde{\mathbf{r}} = \frac{W_R}{\sqrt{W_J}} \mathbf{r}


Adding some regularization to the Gauss-Newton update
------------------------------------------------------

Sometimes, we may also have regularization terms. In this case the problem can be
written as a sum of a sub least squares problem:

.. math::

   \min_{\mathbf{p}} \frac{1}{2} \sum_{i} w_i \sum_j \rho_i\left(\| \mathbf{r}_{i,j}(\mathbf{p}) \|^2\right)

Where :math:`w_i` is a weight for each sub least squares problem, 
and :math:`\rho_i` is a robust cost function for each sub least squares problem.

.. note::

   By nomenclature, we assume that the first sub least squares problem (i.e. :math:`i=0`)
   is the main least squares problem containing the data residuals, and the other sub 
   least squares problems (i.e. :math:`i \geq 1`) are regularization terms.

In this case, the Gauss-Newton update can be written as:

.. math::

   \sum_{i} w_i \tilde{\mathbf{J}}_i^T \tilde{\mathbf{J}}_i \Delta p = -\sum_{i} w_i \tilde{\mathbf{J}}_i^T \tilde{\mathbf{r}}_i


Changing the parametrization
----------------------------

Consider a least squares problem of the form:

.. math::

   \min_{\mathbf{p}_{\mathrm{out}}} \frac{1}{2} \sum_{i} w_i \sum_j
   \rho_i\left(\| \mathbf{r}_{i,j}(\mathbf{p}_{\mathrm{out}}) \|^2\right)

A parametrization can be introduced to optimize the problem using a different set
of input parameters :math:`\mathbf{p}_{\mathrm{in}}`. The parametrization is defined
by a transformation

.. math::

   \mathbf{p}_{\mathrm{out}} = P(\mathbf{p}_{\mathrm{in}})

where :math:`\mathbf{p}_{\mathrm{in}} \in \mathbb{R}^{n_{p_{\mathrm{in}}}}` are the
parameters actually optimized by the Gauss-Newton algorithm, and
:math:`\mathbf{p}_{\mathrm{out}} \in \mathbb{R}^{n_{p_{\mathrm{out}}}}` are the
parameters passed to the residual functions.

The original optimization problem can therefore be rewritten as

.. math::

   \min_{\mathbf{p}_{\mathrm{in}}} \frac{1}{2} \sum_{i} w_i \sum_j
   \rho_i\left(
      \left\|
      \mathbf{r}_{i,j}\left(P(\mathbf{p}_{\mathrm{in}})\right)
      \right\|^2
   \right)

At each iteration, the Gauss-Newton algorithm searches for an update
:math:`\Delta\mathbf{p}_{\mathrm{in}}` to the input parameters
:math:`\mathbf{p}_{\mathrm{in}}`.

The output parameters are obtained from the current input parameters as

.. math::

   \mathbf{p}_{\mathrm{out}} = P(\mathbf{p}_{\mathrm{in}}).

Using

.. math::

   \mathbf{J}_P(\mathbf{p}_{\mathrm{in}})
   =
   \frac{\partial \mathbf{p}_{\mathrm{out}}}
        {\partial \mathbf{p}_{\mathrm{in}}}

to denote the Jacobian matrix of the parametrization, the chain rule gives the
Jacobian of each residual with respect to the parameters being optimized:

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
   \mathbf{J}_{i,j}\mathbf{J}_P.

Consequently, the modified Jacobian used by the Gauss-Newton algorithm is

.. math::

   \tilde{\mathbf{J}}_i
   =
   \sqrt{W_{J,i}}\,
   \mathbf{J}_i\,
   \mathbf{J}_P

while the modified residual remains

.. math::

   \tilde{\mathbf{r}}_i
   =
   \frac{W_{R,i}}{\sqrt{W_{J,i}}}
   \mathbf{r}_i

The Gauss-Newton system is therefore solved for the update
:math:`\Delta\mathbf{p}_{\mathrm{in}}`:

.. math::

   \sum_i w_i
   \tilde{\mathbf{J}}_i^T
   \tilde{\mathbf{J}}_i
   \Delta\mathbf{p}_{\mathrm{in}}
   =
   -\sum_i w_i
   \tilde{\mathbf{J}}_i^T
   \tilde{\mathbf{r}}_i.

After solving this system, the input parameters are updated as

.. math::

   \mathbf{p}_{\mathrm{in}}^{k+1}
   =
   \mathbf{p}_{\mathrm{in}}^k
   +
   \Delta\mathbf{p}_{\mathrm{in}},

and the corresponding output parameters are obtained by applying the
parametrization:

.. math::

   \mathbf{p}_{\mathrm{out}}^{k+1}
   =
   P(\mathbf{p}_{\mathrm{in}}^{k+1}).

This formulation makes explicit that the parametrization changes the coordinates
of the optimization problem: the Gauss-Newton algorithm optimizes
:math:`\mathbf{p}_{\mathrm{in}}`, while the residual functions operate on
:math:`\mathbf{p}_{\mathrm{out}}`.

.. seealso::

   The class :class:`pysolvegn.Parametrization` represents a parametric
   transformation by storing the functions to compute from
   :math:`\mathbf{p}_{\mathrm{in}}` (as a 1D-array with shape
   ``(n_parameters,)``) the output parameters
   :math:`\mathbf{p}_{\mathrm{out}}` (as a 1D-array with shape
   ``(n_p_outputs,)``) and the Jacobian matrix
   :math:`\mathbf{J}_P` (as a 2D-array with shape
   ``(n_p_outputs, n_parameters)``).
