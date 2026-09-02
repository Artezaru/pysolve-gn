"""
.. _sphx_glr__gallery_curve_fitting.py:

Curve fitting
==================================

This example shows how to use pysolve-gn package.

We will fit an exponential curve to noisy data.
"""

# %%
# Define the model function
# --------------------------
#
# Lets consider the following exponential model.
#
# .. math::
#
#     y = a \cdot e^{b x}
#
# We generate a small and noisy dataset. In this situation, the fitted
# parameters can deviate significantly from the true parameters.
#
# The parameters :math:`a` and :math:`b` will be estimated from the noisy data.


import numpy as np
import matplotlib.pyplot as plt
import pysolvegn

np.random.seed(0)


def model(params, x):
    a, b = params
    return a * np.exp(b * x)


n_points = 100
x_data = np.linspace(0, 3, n_points)

true_params = np.array([2.5, 0.5])

y_true = model(true_params, x_data)

# Add relatively strong noise to make the effect of regularization visible.
y_data = y_true + 1.0 * np.random.normal(size=y_true.shape)


# %%
# Create the data term
# --------------------
#
# The data term represents the residuals between the observations and
# the model predictions.
#
# .. math::
#
#     r_j(\mathbf{p}) =
#     a e^{b x_j} - y_j
#
# The Jacobian is:
#
# .. math::
#
#     \frac{\partial r_j}{\partial a} = e^{b x_j}
#
# .. math::
#
#     \frac{\partial r_j}{\partial b}
#     = a x_j e^{b x_j}
#
# .. seealso::
#
#     The class :class:`pysolvegn.Term` to store the residual and jacobian functions.
#


def residual_function(params):
    return model(params, x_data) - y_data


def jacobian_function(params):
    a, b = params

    J = np.zeros((len(x_data), len(params)))

    J[:, 0] = np.exp(b * x_data)
    J[:, 1] = a * x_data * np.exp(b * x_data)

    return J


data_term = pysolvegn.Term.from_rJ(
    residual_func=residual_function,
    jacobian_func=jacobian_function,
    loss="linear",
    weight=1.0,
)


# %%
# Performing optimization
# ---------------------------
#
# We solve the problem using only the :func:`pysolvegn.solve` function.
# This function takes the data term, an initialisation and convergence criterion.
#

initial_params = np.array([1.0, 1.0])

optimized_parameters = pysolvegn.solve(
    terms=[data_term],
    p0=initial_params,
    max_iteration=100,
    xtol=1e-8,
    ftol=1e-8,
    verbosity=2,
)

# %%
# Display the fitted parameters
# ------------------------------

print("True parameters:")
print(true_params)

print("\nEstimated parameters:")
print(optimized_parameters)

x_plot = np.linspace(0, 3, 300)
y_plot = model(optimized_parameters, x_plot)
y_true_plot = model(true_params, x_plot)

plt.figure(figsize=(10, 6))

plt.scatter(
    x_data,
    y_data,
    label="Data Points",
    color="red",
    s=35,
)

plt.plot(
    x_plot,
    y_true_plot,
    label="True Curve",
    color="blue",
    linewidth=2,
)

plt.plot(
    x_plot,
    y_plot,
    label="Optimized Curve",
    color="orange",
    linestyle="--",
    linewidth=2,
)

plt.title("Curve Fitting")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()

plt.show()
