"""
.. _sphx_glr__gallery_adding_regularization.py:

Adding regularization (curve fitting)
======================================

This example shows how to use regularization with pysolve-gn.
This example is the continuation of the basic curve fitting example,
demonstrating how to incorporate regularization.

"""

# %%
# Copy the basic curve fitting example
# ---------------------------------------
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
#
# A Gaussian prior will then be used to regularize the solution.


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
# Define a Gaussian regularization
# ---------------------------------
#
# We can add a Gaussian prior on the parameters.
# The regularization will be added as an additional term (:class:`pysolvegn.Term`) in the optimization problem.
#
# The regularization residuals are:
#
# .. math::
#
#     r_{\mathrm{reg},i}
#     =
#     \frac{p_i - \mu_i}{\sigma_i}
#
# where :math:`\mu_i` is the expected value of the parameter and
# :math:`\sigma_i` controls the strength of the prior.
#
# The corresponding contribution to the cost is:
#
# .. math::
#
#     \frac{1}{2}
#     \sum_i
#     \left(
#         \frac{p_i-\mu_i}{\sigma_i}
#     \right)^2
#
# Here we assume that the parameters are expected to be close to
# :math:`[2.5, 0.5]`, but we allow some uncertainty around these values.
#
# .. note::
#
#    This regularization term can be define similarly to the data term, given a residual function and a Jacobian function
#    or directly using the provided helper function `pysolvegn.build_squared_regularization`.

prior_means = np.array([2.5, 0.5])
prior_stds = np.array([0.5, 0.2])

regularization_term = pysolvegn.build_squared_regularization(
    means=prior_means,
    stds=prior_stds,
    weight=1.0,  # Can be change to update the regularization strength
)

# %%
# Fit with regularization
# -----------------------
#
# The regularization is simply added as another term in the optimization
# problem.
#
# The solver therefore minimizes:
#
# .. math::
#
#     C(\mathbf{p})
#     =
#     \frac{1}{2}\sum_j r_j(\mathbf{p})^2
#     +
#     \frac{1}{2}w \sum_i
#     \left(
#         \frac{p_i-\mu_i}{\sigma_i}
#     \right)^2
#
# Both the data term and the regularization term contribute to the
# Gauss-Newton system.

result_without_regularization = pysolvegn.solve(
    terms=[
        data_term,
    ],
    p0=np.array([1.0, 1.0]),
    max_iteration=100,
    xtol=1e-8,
    ftol=1e-8,
    verbosity=0,
)

result_with_regularization = pysolvegn.solve(
    terms=[
        data_term,
        regularization_term,
    ],
    p0=np.array([1.0, 1.0]),
    max_iteration=100,
    xtol=1e-8,
    ftol=1e-8,
    verbosity=0,
)


# %%
# Display the fitted parameters
# ------------------------------

print("True parameters:")
print(true_params)

print("\nEstimated parameters without regularization:")
print(result_without_regularization)

print("\nEstimated parameters with regularization:")
print(result_with_regularization)

# Use a dense set of points to display the fitted curves smoothly.
x_plot = np.linspace(0, 3, 300)

y_true_plot = model(true_params, x_plot)

y_without_regularization_plot = model(
    result_without_regularization,
    x_plot,
)
y_with_regularization_plot = model(
    result_with_regularization,
    x_plot,
)

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
    y_without_regularization_plot,
    label="Without Regularization",
    color="orange",
    linestyle="--",
    linewidth=2,
)

plt.plot(
    x_plot,
    y_with_regularization_plot,
    label="With Regularization",
    color="green",
    linestyle="-.",
    linewidth=2,
)

plt.title("Curve Fitting with Gaussian Regularization")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()

plt.show()
