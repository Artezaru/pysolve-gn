Mathematical Background
=======================

.. contents:: Table of Contents
   :local:
   :depth: 2


Robust Least Squares Optimization by the Gauss-Newton Method
-------------------------------------------------------------


Consider a least squares problem of the form:

.. math::

   \min_{\mathbf{p}} \frac{1}{2} \sum_j \rho\left(\| \mathbf{R}_j(\mathbf{p}) \|^2\right)

where :math:`\mathbf{R}_j(\mathbf{p})` is a residual function (:math:`\mathbb{R}^{n} \rightarrow \mathbb{R}`)
depending on the parameters :math:`\mathbf{p} \in \mathbb{R}^{n}`, and :math:`\rho` is a robust 
cost function (:math:`\mathbb{R} \rightarrow \mathbb{R}`) that reduces the influence of outliers.

At each iteration of the Gauss-Newton method, we search for an update :math:`\Delta p` 
to the current parameters :math:`\mathbf{p_k}` that minimizes the robust cost function.
Developing the residuals around the current parameters :math:`\mathbf{p_k}`:

.. math::

   \mathbf{R}_j(\mathbf{p} + \Delta p) \approx \mathbf{R}_j + \mathbf{J}_j \Delta p + 0.5 \Delta p^T \mathbf{H}_j \Delta p + \ldots

where :math:`\mathbf{J}_j(\mathbf{p_k}) = \nabla \mathbf{R}_j(\mathbf{p_k})` is the Jacobian 
of the residuals with respect to the parameters, and :math:`\mathbf{H}_j(\mathbf{p_k})` 
is the Hessian of the residuals.

Let :math:`Z_j = \| \mathbf{R}_j(\mathbf{p_k}) \|^2` be the squared 
norm of the residuals at the current parameters.
The robust cost function can be developed around :math:`Z_j`:

.. math::

   \rho(Z_j + \delta Z_j) \approx \rho(Z_j) + \rho'(Z_j)\delta Z_j + 0.5 \rho''(Z_j) \delta Z_j^2 + \ldots

where :math:`\delta Z_j` is the change in the squared norm due to the parameter update:

.. math::

   \delta Z_j = \| \mathbf{R}_j(\mathbf{p_k} + \Delta p) \|^2 - \| \mathbf{R}_j(\mathbf{p_k}) \|^2

By only keeping up to the second order terms, we can approximate :math:`\delta Z_j` as:

.. math::

   \delta Z_j \approx \Delta p^T \left( \mathbf{J}_j^T \mathbf{J}_j + \mathbf{R}_j^T \mathbf{H}_j \right) \Delta p + 2 \mathbf{R}_j^T \mathbf{J}_j \Delta p + ...

In a similar way, the squared term :math:`\delta Z_j^2` can be approximated as:

.. math::

   \delta Z_j^2 \approx 4 \left( \mathbf{R}_j^T \mathbf{J}_j \Delta p \right)^2 + ...

Thus, by injecting all the approximations into the robust cost function, 
we optain :

.. math::

    \rho(Z_j + \delta Z_j) \approx
    \rho(Z_j) + 
    \rho'(Z_j) \Big[ \Delta p^T \left( \mathbf{J}_j^T \mathbf{J}_j + \mathbf{R}_j^T \mathbf{H}_j \right) \Delta p + 2 \mathbf{R}_j^T \mathbf{J}_j \Delta p \Big] + 
    2 \rho''(Z_j) \left( \mathbf{R}_j^T \mathbf{J}_j \Delta p \right)^2 + ...

With Gauss-Newton, we shearch for the update :math:`\Delta p` that minimizes 
the robust cost function, which is equivalent to solving the zero of the gradient
of the robust cost function with respect to the parameters.

.. math::

    \nabla_{\Delta p} \rho(Z_j + \delta Z_j) = 0

The Gauss-Newton update can be obtained by solving the following linear system:

.. math::

    H \, \Delta p = - g

Where :math:`g` is the gradient of the robust cost function with respect to the 
parameters, and :math:`H` is the Hessian approximation of the robust cost function 
with respect to the parameters.

By summing over all residuals, the gradient and Hessian approximation are:

.. math::

   g \approx \sum_j 2 \rho'(Z_j) \mathbf{J}_j^T \mathbf{R}_j

.. math::

   H \approx \sum_j \Big[2 \rho'(Z_j) \left( \mathbf{J}_j^T \mathbf{J}_j + \mathbf{R}_j^T \mathbf{H}_j \right) + 4 \rho''(Z_j) \mathbf{J}_j^T \mathbf{R}_j \mathbf{R}_j^T \mathbf{J}_j \Big]

Finally, the Gauss-Newton suggested to ignore the second order term of the residuals, 
as at convergence, the residuals should be small, and thus the second order term 
:math:`\mathbf{R}_j^T \mathbf{H}_j` should be negligible compared to the first order 
term :math:`\mathbf{J}_j^T \mathbf{J}_j`.

.. math::

   H \approx \sum_j \Big[2 \rho'(Z_j) \mathbf{J}_j^T \mathbf{J}_j + 4 \rho''(Z_j) \mathbf{J}_j^T \mathbf{R}_j \mathbf{R}_j^T \mathbf{J}_j \Big]

.. math::

   H \approx \sum_j 2 \mathbf{J}_j^T \Big[ \rho'(Z_j) + 2 \rho''(Z_j) Z_j \Big] \mathbf{J}_j

Finally, we can build a diagonal matrix :math:`W_J` and a vector :math:`W_R` that depend on the 
robust cost function and the squared norm of the residuals to solve the complete linear system
as:

.. math::

   \mathbf{J}^T \sqrt{W_J}^T \sqrt{W_J} \mathbf{J} \Delta p = - W_R \mathbf{J}^T \mathbf{R}

.. math::

   W_J = \text{diag}\left(\rho'(|\mathbf{R}_j|^2) + 2 \rho''(|\mathbf{R}_j|^2) |\mathbf{R}_j|^2\right)
 
.. math::

   W_R = \text{diag}\left(\rho'(|\mathbf{R}_j|^2)\right)

Wo observe that the system is similar to solve:

.. math::

   \tilde{\mathbf{J}}^T \tilde{\mathbf{J}} \Delta p = - \tilde{\mathbf{J}}^T \tilde{\mathbf{R}}

With a modified Jacobian :math:`\tilde{\mathbf{J}} = \sqrt{W_J} \mathbf{J}` and a modified residual :math:`\tilde{\mathbf{R}} = \frac{W_R}{\sqrt{W_J}} \mathbf{R}`.


Adding some regularization to the Gauss-Newton update
------------------------------------------------------

Sometimes, we may also have regularization terms. In this case the problem can be
written as a sum of a sub least squares problem:

.. math::

   \min_{\mathbf{p}} \frac{1}{2} \sum_{i} w_i \sum_j \rho_i\left(\| \mathbf{R}_{i,j}(\mathbf{p}) \|^2\right)

Where :math:`w_i` is a weight for each sub least squares problem, 
and :math:`\rho_i` is a robust cost function for each sub least squares problem.

.. note::

   By nomenclature, we assume that the first sub least squares problem (i.e. :math:`i=0`)
   is the main least squares problem containing the data residuals, and the other sub 
   least squares problems (i.e. :math:`i \geq 1`) are regularization terms.

In this case, the Gauss-Newton update can be written as:

.. math::

   \sum_{i} w_i \tilde{\mathbf{J}}_i(\mathbf{p_k})^T \tilde{\mathbf{J}}_i(\mathbf{p_k}) \Delta p = -\sum_{i} w_i \tilde{\mathbf{J}}_i(\mathbf{p_k})^T \tilde{\mathbf{R}}_i(\mathbf{p_k})

