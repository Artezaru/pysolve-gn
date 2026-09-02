"""
.. _sphx_glr__gallery_affine_parametrization.py:

Camera calibration with affine parametrization
===============================================

This example shows how to use a parametrization with pysolve-gn.

We consider a simplified camera calibration problem in which the camera
intrinsic parameters and the radial/tangential distortion parameters are
estimated from synthetic image observations.

The complete camera parameter vector is:

    ``[fx, fy, cx, cy, k1, k2, p1, p2, k3]``

However, we explicitly impose the following constraints:

    ``fx = fy``
    ``cx = fixed``
    ``cy = fixed``

Therefore, only six parameters are actually optimized:

    ``[f, k1, k2, p1, p2, k3]``

The example also shows that these constraints can be expressed using
:func:`pysolvegn.build_affine_parametrization`.
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
import matplotlib.pyplot as plt
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
# We assume square pixels:
#
# .. math::
#
#     f_x = f_y = f
#
# and a known principal point:
#
# .. math::
#
#     c_x = c_{x,\mathrm{fixed}}
#
# .. math::
#
#     c_y = c_{y,\mathrm{fixed}}
#
# Consequently, the unknown parameters are:
#
# .. math::
#
#     [f, k_1, k_2, p_1, p_2, k_3]


cx_fixed = 640.0
cy_fixed = 480.0

true_f = 850.0
true_k1 = -0.20
true_k2 = 0.05
true_p1 = 0.001
true_p2 = -0.002
true_k3 = -0.01

true_parameters = np.array(
    [
        true_f,
        true_f,
        cx_fixed,
        cy_fixed,
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
# Define the affine parametrization
# ---------------------------------
#
# The solver optimizes only six parameters:
#
# .. math::
#
#     \mathbf{p}_{in}
#     =
#     \begin{bmatrix}
#     f & k_1 & k_2 & p_1 & p_2 & k_3
#     \end{bmatrix}^T.
#
# The camera model expects nine parameters:
#
# .. math::
#
#     \mathbf{p}_{out}
#     =
#     \begin{bmatrix}
#     f_x & f_y & c_x & c_y & k_1 & k_2 & p_1 & p_2 & k_3
#     \end{bmatrix}^T.
#
# We define the transformation:
#
# .. math::
#
#     \mathbf{p}_{out}
#     =
#     M\mathbf{p}_{in} + \mathbf{p}_0
#
# with:
#
# .. math::
#
#     M =
#     \begin{bmatrix}
#     1 & 0 & 0 & 0 & 0 & 0 \\
#     1 & 0 & 0 & 0 & 0 & 0 \\
#     0 & 0 & 0 & 0 & 0 & 0 \\
#     0 & 0 & 0 & 0 & 0 & 0 \\
#     0 & 1 & 0 & 0 & 0 & 0 \\
#     0 & 0 & 1 & 0 & 0 & 0 \\
#     0 & 0 & 0 & 1 & 0 & 0 \\
#     0 & 0 & 0 & 0 & 1 & 0 \\
#     0 & 0 & 0 & 0 & 0 & 1
#     \end{bmatrix}
#
# and:
#
# .. math::
#
#     \mathbf{p}_0 =
#     \begin{bmatrix}
#     0 & 0 & c_x & c_y & 0 & 0 & 0 & 0 & 0
#     \end{bmatrix}^T.
#
# Therefore:
#
#     fx = f
#     fy = f
#     cx = cx_fixed
#     cy = cy_fixed
#
# while the distortion parameters are unchanged.


modes = np.zeros((9, 6), dtype=float)

# fx = f
modes[0, 0] = 1.0

# fy = f
modes[1, 0] = 1.0

# cx and cy are fixed.
# Their rows are therefore zero.

# k1 = k1
modes[4, 1] = 1.0

# k2 = k2
modes[5, 2] = 1.0

# p1 = p1
modes[6, 3] = 1.0

# p2 = p2
modes[7, 4] = 1.0

# k3 = k3
modes[8, 5] = 1.0


offset = np.array(
    [
        0.0,
        0.0,
        cx_fixed,
        cy_fixed,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
)


# Build the parametrization using the helper function.
parametrization = pysolvegn.build_affine_parametrization(
    modes=modes,
    offset=offset,
)


# %%
# Define the residual
# -------------------
#
# The residual function receives the complete output parameter vector
# generated by the parametrization.
#
# The solver itself only manipulates the six input parameters.


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
# Perform the calibration
# -----------------------
#
# The initial guess contains only the six optimized parameters:
#
#     ``[f, k1, k2, p1, p2, k3]``
#
# The fixed values ``cx`` and ``cy`` are not included in ``p0``.


initial_parameters = np.array(
    [
        800.0,  # f
        0.0,  # k1
        0.0,  # k2
        0.0,  # p1
        0.0,  # p2
        0.0,  # k3
    ]
)


result = pysolvegn.solve(
    terms=[data_term],
    p0=initial_parameters,
    parametrization=parametrization,
    max_iteration=30,
    xtol=1e-8,
    ftol=1e-8,
    verbosity=2,
)


print("Optimized parameters:")
print("f  =", result[0])
print("k1 =", result[1])
print("k2 =", result[2])
print("p1 =", result[3])
print("p2 =", result[4])
print("k3 =", result[5])


# %%
# Recover the complete camera parameter vector
# ---------------------------------------------
#
# ``result`` contains the optimized input parameters.
#
# The complete camera parameters are obtained by applying the
# parametrization:
#
#     ``p_out = P(p_in)``


estimated_parameters = parametrization.p_func(result)

print()
print("Complete camera parameters:")
print("fx =", estimated_parameters[0])
print("fy =", estimated_parameters[1])
print("cx =", estimated_parameters[2])
print("cy =", estimated_parameters[3])
print("k1 =", estimated_parameters[4])
print("k2 =", estimated_parameters[5])
print("p1 =", estimated_parameters[6])
print("p2 =", estimated_parameters[7])
print("k3 =", estimated_parameters[8])


# %%
# Verify the constraints
# ----------------------
#
# The constraints are hard constraints imposed by the parametrization.
# They are therefore satisfied exactly, rather than approximately.


assert np.isclose(
    estimated_parameters[0],
    estimated_parameters[1],
)

assert np.isclose(
    estimated_parameters[2],
    cx_fixed,
)

assert np.isclose(
    estimated_parameters[3],
    cy_fixed,
)


# %%
# Equivalence with the explicit affine transformation
# ----------------------------------------------------
#
# ``build_affine_parametrization`` implements:
#
#     ``p_out = modes @ p_in + offset``
#
# Therefore applying the parametrization is equivalent to directly evaluating
# this affine transformation.


estimated_parameters_direct = modes @ result + offset


assert np.allclose(
    estimated_parameters,
    estimated_parameters_direct,
)


print()
print("The parametrization is equivalent to " "modes @ parameters + offset.")


# %%
# Visualize the calibration result
# --------------------------------


image_points_estimated = camera_model(
    estimated_parameters,
    normalized_points,
)


plt.figure(figsize=(10, 6))

plt.scatter(
    image_points[:, 0],
    image_points[:, 1],
    label="Observed points",
    color="red",
    s=15,
)

plt.scatter(
    image_points_true[:, 0],
    image_points_true[:, 1],
    label="True points",
    color="blue",
    s=15,
)

plt.scatter(
    image_points_estimated[:, 0],
    image_points_estimated[:, 1],
    label="Fitted points",
    color="green",
    marker="x",
    s=20,
)

plt.title("Camera calibration using an affine parametrization")

plt.xlabel("u [pixels]")
plt.ylabel("v [pixels]")

plt.legend()
plt.grid()
plt.axis("equal")
plt.show()
