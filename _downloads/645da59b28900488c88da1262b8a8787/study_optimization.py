"""
.. _sphx_glr__gallery_study_optimization.py:

Study of camera calibration optimization
========================================

This example shows how to use :func:`pysolvegn.study_optimization`
to study the Gauss-Newton Hessian of a camera calibration problem.

We consider a simplified camera calibration problem in which the camera
intrinsic parameters and the radial/tangential distortion parameters are
estimated from synthetic image observations.

The complete camera parameter vector is:

    ``[fx, fy, cx, cy, k1, k2, p1, p2, k3]``

Unlike the affine parametrization example, all nine parameters are
independent optimization variables.

No parametrization is used.

The purpose of this example is therefore to study the observability,
conditioning, covariance, and weak parameter combinations of the complete
nine-dimensional camera calibration problem at a given initial point.

No optimization step is performed. ``study_optimization`` evaluates the
initial state by internally calling ``solve`` with ``max_iteration=0``.
"""

# %%
# Camera model
# ------------
#
# We use the Brown-Conrady camera distortion model.
#
# For normalized coordinates ``(x, y)``, the distorted coordinates are:
#
# .. math::
#
#     x_d =
#         x(1 + k_1 r^2 + k_2 r^4 + k_3 r^6)
#         + 2p_1xy
#         + p_2(r^2 + 2x^2)
#
# .. math::
#
#     y_d =
#         y(1 + k_1 r^2 + k_2 r^4 + k_3 r^6)
#         + p_1(r^2 + 2y^2)
#         + 2p_2xy
#
# where:
#
# .. math::
#
#     r^2 = x^2 + y^2.
#
# The pixel coordinates are:
#
# .. math::
#
#     u = f_x x_d + c_x
#
# .. math::
#
#     v = f_y y_d + c_y


import numpy as np
import pysolvegn

np.random.seed(0)


def camera_model(params, points):
    """
    Compute distorted image coordinates.

    Parameters
    ----------
    params : array-like
        Complete camera parameter vector:

        [fx, fy, cx, cy, k1, k2, p1, p2, k3]

    points : ndarray
        Normalized image points with shape ``(n_points, 2)``.

    Returns
    -------
    ndarray
        Image coordinates with shape ``(n_points, 2)``.
    """
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = params

    x = points[:, 0]
    y = points[:, 1]

    r2 = x**2 + y**2

    radial = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3

    x_distorted = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x**2)

    y_distorted = y * radial + p1 * (r2 + 2.0 * y**2) + 2.0 * p2 * x * y

    u = fx * x_distorted + cx
    v = fy * y_distorted + cy

    return np.column_stack((u, v))


# %%
# Generate synthetic calibration observations
# --------------------------------------------
#
# We use the same synthetic camera parameters as in the affine
# parametrization example.
#
# Here, however, all nine parameters are considered independent.


true_fx = 850.0
true_fy = 850.0
true_cx = 640.0
true_cy = 480.0

true_k1 = -0.20
true_k2 = 0.05
true_p1 = 0.001
true_p2 = -0.002
true_k3 = -0.01


true_parameters = np.array(
    [
        true_fx,
        true_fy,
        true_cx,
        true_cy,
        true_k1,
        true_k2,
        true_p1,
        true_p2,
        true_k3,
    ]
)


# Generate normalized calibration points.
x = np.linspace(-0.7, 0.7, 15)
y = np.linspace(-0.5, 0.5, 11)

xx, yy = np.meshgrid(x, y)

normalized_points = np.column_stack(
    (
        xx.ravel(),
        yy.ravel(),
    )
)


image_points_true = camera_model(
    true_parameters,
    normalized_points,
)


# Add measurement noise.
image_points = image_points_true + np.random.normal(
    scale=0.5,
    size=image_points_true.shape,
)


# %%
# Define the residual
# -------------------
#
# The residual function directly receives the complete nine-dimensional
# parameter vector.
#
# There is no parametrization between the solver and the camera model.


def residual_function(params):
    predicted_points = camera_model(
        params,
        normalized_points,
    )

    return (predicted_points - image_points).ravel()


data_term = pysolvegn.Term.from_rJ(
    residual_func=residual_function,
    loss="linear",
    weight=1.0,
)


# %%
# Define the initial parameters
# -----------------------------
#
# All nine camera parameters are independent optimization variables:
#
#     ``[fx, fy, cx, cy, k1, k2, p1, p2, k3]``


initial_parameters = np.array(
    [
        800.0,  # fx
        800.0,  # fy
        640.0,  # cx
        480.0,  # cy
        0.0,  # k1
        0.0,  # k2
        0.0,  # p1
        0.0,  # p2
        0.0,  # k3
    ]
)


# %%
# Study the optimization problem
# ------------------------------
#
# ``study_optimization`` evaluates the least-squares problem at the
# initial parameter vector.
#
# It internally calls ``solve`` with:
#
#     ``max_iteration=0``
#
# Consequently, no optimization step is performed.
#
# The Gauss-Newton Hessian is computed directly in the complete
# nine-dimensional parameter space.


pysolvegn.study_optimization(
    terms=[data_term],
    p0=initial_parameters,
    title="Camera calibration at initialization - complete parameter vector",
)


# %%
# Study the optimization problem at convergence
# -----------------------------------------------
#
# ``study_optimization`` evaluates the least-squares problem at the
# initial parameter vector.
#
# It internally calls ``solve`` with:
#
#     ``max_iteration=0``
#
# Consequently, no optimization step is performed.
#
# The Gauss-Newton Hessian is computed directly in the complete
# nine-dimensional parameter space.

converged_parameters = pysolvegn.solve(
    terms=[data_term],
    p0=initial_parameters,
    max_iteration=100,
    ftol=1e-6,
    xtol=1e-6,
    gtol=1e-6,
)

pysolvegn.study_optimization(
    terms=[data_term],
    p0=converged_parameters,
    title="Camera calibration at convergence - complete parameter vector",
)
