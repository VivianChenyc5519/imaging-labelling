"""
bezier_simvc.py

Utility functions for cubic Bezier curve completion and SIMVC-like fairness
energy in 2D or 3D.

This file is intentionally independent from the skeleton/graph code.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


# ============================================================
# Basic vector utilities
# ============================================================

def unit_vector(v, eps: float = 1e-12):
    """Return v / ||v||, or None if v is too small."""
    if v is None:
        return None

    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))

    if n < eps:
        return None

    return v / n


def angle_deg(v1, v2, eps: float = 1e-12) -> float:
    """
    Directed angle in degrees between two vectors.

    0 deg   = same direction
    90 deg  = perpendicular
    180 deg = opposite direction
    """
    u1 = unit_vector(v1, eps=eps)
    u2 = unit_vector(v2, eps=eps)

    if u1 is None or u2 is None:
        return 0.0

    c = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


# ============================================================
# Cubic Bezier curve and derivatives
# ============================================================

def bezier_cubic(P0, P1, P2, P3, t):
    """
    Sample cubic Bezier curve in 2D or 3D.

    B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t)t^2 P2 + t^3 P3
    """
    P0 = np.asarray(P0, dtype=np.float64)
    P1 = np.asarray(P1, dtype=np.float64)
    P2 = np.asarray(P2, dtype=np.float64)
    P3 = np.asarray(P3, dtype=np.float64)

    t = np.asarray(t, dtype=np.float64)
    u = 1.0 - t

    return (
        (u ** 3)[:, None] * P0
        + (3.0 * u ** 2 * t)[:, None] * P1
        + (3.0 * u * t ** 2)[:, None] * P2
        + (t ** 3)[:, None] * P3
    )


def bezier_cubic_d1(P0, P1, P2, P3, t):
    """First derivative B'(t) of cubic Bezier curve."""
    P0 = np.asarray(P0, dtype=np.float64)
    P1 = np.asarray(P1, dtype=np.float64)
    P2 = np.asarray(P2, dtype=np.float64)
    P3 = np.asarray(P3, dtype=np.float64)

    t = np.asarray(t, dtype=np.float64)
    u = 1.0 - t

    return (
        (3.0 * u ** 2)[:, None] * (P1 - P0)
        + (6.0 * u * t)[:, None] * (P2 - P1)
        + (3.0 * t ** 2)[:, None] * (P3 - P2)
    )


def bezier_cubic_d2(P0, P1, P2, P3, t):
    """Second derivative B''(t) of cubic Bezier curve."""
    P0 = np.asarray(P0, dtype=np.float64)
    P1 = np.asarray(P1, dtype=np.float64)
    P2 = np.asarray(P2, dtype=np.float64)
    P3 = np.asarray(P3, dtype=np.float64)

    t = np.asarray(t, dtype=np.float64)
    u = 1.0 - t

    return (
        (6.0 * u)[:, None] * (P2 - 2.0 * P1 + P0)
        + (6.0 * t)[:, None] * (P3 - 2.0 * P2 + P1)
    )


# ============================================================
# Curvature and fairness / SIMVC-like energy
# ============================================================

def curvature_from_derivatives_nd(d1, d2, eps: float = 1e-12):
    """
    Curvature for a 2D or 3D parametric curve.

    2D:
        kappa = |x' y'' - y' x''| / ||B'||^3

    3D:
        kappa = ||B' x B''|| / ||B'||^3
    """
    d1 = np.asarray(d1, dtype=np.float64)
    d2 = np.asarray(d2, dtype=np.float64)

    if d1.ndim != 2 or d2.ndim != 2:
        raise ValueError("d1 and d2 must be arrays of shape (n_samples, dim).")

    if d1.shape != d2.shape:
        raise ValueError("d1 and d2 must have the same shape.")

    dim = d1.shape[1]
    speed = np.linalg.norm(d1, axis=1)

    if dim == 2:
        x1, y1 = d1[:, 0], d1[:, 1]
        x2, y2 = d2[:, 0], d2[:, 1]
        numerator = np.abs(x1 * y2 - y1 * x2)
    elif dim == 3:
        numerator = np.linalg.norm(np.cross(d1, d2), axis=1)
    else:
        raise ValueError("Only 2D and 3D curves are supported.")

    denominator = (speed + eps) ** 3
    return numerator / denominator


def simvc_energy_for_handles_nd(
    A,
    B,
    TA,
    TB,
    c1: float,
    c2: float,
    n_samples: int = 120,
    handle_regularization: float = 1e-3,
):
    """
    SIMVC-like fairness energy for cubic Bezier completion in 2D or 3D.

    P0 = A
    P1 = A + c1 * TA
    P2 = B + c2 * TB
    P3 = B

    Energy:
        E = (L^5 / chord^2) * integral((dkappa/ds)^2 ds)
            + handle_regularization * (c1^2 + c2^2)
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    TA = unit_vector(TA)
    TB = unit_vector(TB)

    if TA is None or TB is None:
        return np.inf

    P0 = A
    P1 = A + c1 * TA
    P2 = B + c2 * TB
    P3 = B

    t = np.linspace(0.0, 1.0, int(n_samples))
    dt = float(t[1] - t[0])

    d1 = bezier_cubic_d1(P0, P1, P2, P3, t)
    d2 = bezier_cubic_d2(P0, P1, P2, P3, t)

    speed = np.linalg.norm(d1, axis=1) + 1e-12
    ds = speed * dt
    length = float(np.sum(ds))

    kappa = curvature_from_derivatives_nd(d1, d2)

    dk_dt = np.gradient(kappa, dt)
    dk_ds = dk_dt / speed

    integral = float(np.sum((dk_ds ** 2) * ds))

    chord = float(np.linalg.norm(B - A)) + 1e-12
    energy = (length ** 5) / (chord ** 2) * integral

    energy += handle_regularization * (float(c1) ** 2 + float(c2) ** 2)

    return float(energy)


def optimize_bezier_simvc_nd(
    A,
    B,
    TA,
    TB,
    n_samples: int = 120,
    x0_fraction: float = 0.35,
    min_handle_fraction: float = 0.03,
    max_handle_fraction: float = 1.50,
):
    """
    Optimize cubic Bezier handle lengths for SIMVC-like completion.

    Parameters
    ----------
    A, B:
        Endpoints, shape (2,) or (3,).
    TA, TB:
        Endpoint tangent directions. They do not need to be normalized.
        TA should point out of A into the completion gap.
        TB should point out of B into the completion gap.
    n_samples:
        Number of samples used in the SIMVC objective.
    x0_fraction:
        Initial handle length as fraction of chord length.
    min_handle_fraction, max_handle_fraction:
        Bounds on handle lengths as fraction of chord length.

    Returns
    -------
    control_points:
        Tuple (P0, P1, P2, P3), or None if optimization is invalid.
    info:
        Dict containing c1, c2, fun, success, nit, reason.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    TA = unit_vector(TA)
    TB = unit_vector(TB)

    if TA is None or TB is None:
        return None, {
            "success": False,
            "fun": np.inf,
            "reason": "invalid tangent",
        }

    chord = float(np.linalg.norm(B - A))

    if chord < 1e-8:
        return None, {
            "success": False,
            "fun": np.inf,
            "reason": "zero chord",
        }

    min_handle = float(min_handle_fraction * chord)
    max_handle = float(max_handle_fraction * chord)
    x0 = np.array([x0_fraction * chord, x0_fraction * chord], dtype=np.float64)

    def objective(c):
        return simvc_energy_for_handles_nd(
            A,
            B,
            TA,
            TB,
            float(c[0]),
            float(c[1]),
            n_samples=n_samples,
        )

    res = minimize(
        objective,
        x0,
        method="Powell",
        bounds=[(min_handle, max_handle), (min_handle, max_handle)],
        options={"maxiter": 120, "xtol": 1e-4, "ftol": 1e-4},
    )

    c1_opt = float(res.x[0])
    c2_opt = float(res.x[1])

    P0 = A
    P1 = A + c1_opt * TA
    P2 = B + c2_opt * TB
    P3 = B

    return (P0, P1, P2, P3), {
        "success": bool(res.success),
        "c1": c1_opt,
        "c2": c2_opt,
        "fun": float(res.fun),
        "nit": int(getattr(res, "nit", -1)),
        "reason": str(getattr(res, "message", "")),
    }


# ============================================================
# Polyline diagnostics for bridge scoring
# ============================================================

def max_polyline_turn_angle(points) -> float:
    """
    Maximum local turn angle along a sampled polyline.

    0 deg = locally straight.
    Larger values indicate sharper local turns.
    """
    pts = np.asarray(points, dtype=np.float64)

    if len(pts) < 3:
        return 0.0

    max_angle = 0.0

    for i in range(1, len(pts) - 1):
        v_prev = pts[i] - pts[i - 1]
        v_next = pts[i + 1] - pts[i]
        max_angle = max(max_angle, angle_deg(v_prev, v_next))

    return float(max_angle)


def second_derivative_smoothness(points) -> float:
    """
    Discrete second-derivative smoothness for sampled points.

    This is not exact curvature. It measures average second finite difference:
        P[i+1] - 2P[i] + P[i-1]
    """
    pts = np.asarray(points, dtype=np.float64)

    if len(pts) < 3:
        return 0.0

    second = pts[2:] - 2.0 * pts[1:-1] + pts[:-2]
    return float(np.mean(np.linalg.norm(second, axis=1)))


def sample_bezier_bridge(control_points, n: int = 32):
    """Sample a Bezier bridge from control points (P0, P1, P2, P3)."""
    P0, P1, P2, P3 = control_points
    t = np.linspace(0.0, 1.0, int(n))
    return bezier_cubic(P0, P1, P2, P3, t).astype(np.float32)
