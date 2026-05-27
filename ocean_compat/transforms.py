"""Rotation-matrix and coordinate-frame transform utilities.

Pure PyTorch, no Isaac Sim / torchrl dependencies.
"""

from __future__ import annotations

from torch import Tensor

from oceanscale.marinegym_compat.math import (
    euler_to_quaternion,
    quaternion_to_euler,
    quaternion_to_rotation_matrix,
)


def euler_to_rotation_matrix(euler: Tensor) -> Tensor:
    """Convert Euler angles (roll, pitch, yaw) to a 3x3 rotation matrix.

    Args:
        euler: (..., 3) tensor of Euler angles in radians.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    q = euler_to_quaternion(euler)
    return quaternion_to_rotation_matrix(q)


def rotation_matrix_to_euler(R: Tensor) -> Tensor:
    """Extract Euler angles (roll, pitch, yaw) from rotation matrices.

    ZYX intrinsic convention.  Calls through to the quaternion path for
    numerical stability.

    Args:
        R: (..., 3, 3) rotation matrices.

    Returns:
        (..., 3) Euler angles in radians.
    """
    # Recover quaternion from rotation matrix via Shepperd's method.
    # Batch-flatten, compute, then restore shape.
    batch_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)

    trace = R_flat[:, 0, 0] + R_flat[:, 1, 1] + R_flat[:, 2, 2]

    # When trace > 0 (general case)
    s = (trace + 1.0).clamp(min=1e-10).sqrt() * 2.0
    w = 0.25 * s
    x = (R_flat[:, 2, 1] - R_flat[:, 1, 2]) / s
    y = (R_flat[:, 0, 2] - R_flat[:, 2, 0]) / s
    z = (R_flat[:, 1, 0] - R_flat[:, 0, 1]) / s

    q = __import__("torch").stack([w, x, y, z], dim=-1)
    q = q.reshape(*batch_shape, 4)

    # Normalize to unit quaternion
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-10)
    return quaternion_to_euler(q)
