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
