from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Helpful alias for readability
FloatArray = NDArray[np.float64]


def _as_float64(a: list[float] | FloatArray) -> FloatArray:
    """Ensure float64 NDArray for consistent typing and math."""
    return np.asarray(a, dtype=np.float64)


def kabsch(
    P: FloatArray,
    Q: FloatArray,
    weights: NDArray[np.float64] | None = None,
    allow_reflection: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """
    Compute the optimal rigid transformation that aligns P onto Q using the Kabsch algorithm.

    This implementation assumes **row-vector points** of shape ``(N, D)`` and solves for
    rotation ``R`` and translation ``t`` in the mapping:

        Q ≈ P @ R + t

    The solution minimizes the root-mean-square deviation (RMSD) between P transformed
    and Q, optionally with per-point weights.

    The algorithm:

      1. Compute centroids of P and Q (weighted if `weights` provided).
      2. Subtract centroids to get centered coordinates P0, Q0.
      3. Compute the cross-covariance matrix::

             C = P0.T @ Q0          # (D, D) for row-vector convention

      4. Perform singular value decomposition::

             U, S, Vt = np.linalg.svd(C)

      5. Compute rotation::

             R = Vt.T @ U.T

         If `allow_reflection` is False and det(R) < 0, flip the sign of the last row
         of Vt before recomputing R to ensure a proper rotation (det(R) = +1).

      6. Compute translation::

             t = cQ - cP @ R

    Args:
        P (ndarray of shape (N, D)): Source point coordinates.
        Q (ndarray of shape (N, D)): Target point coordinates, corresponding 1-to-1 with P.
        weights (ndarray of shape (N,), optional): Nonnegative weights for each correspondence.
            If provided, centroids and covariance are computed with these weights.
        allow_reflection (bool, default=False): If False, the solution will have det(R) >= 0
            (proper rotation). If True, improper rotations (reflections) are allowed.

    Returns:
        Tuple[ndarray, ndarray]:
            - R (ndarray of shape (D, D)): Optimal rotation matrix.
            - t (ndarray of shape (D,)): Translation vector.

    Raises:
        ValueError: If P and Q have mismatched shapes, fewer than D points are provided,
            or if weights are invalid (negative, wrong shape, or zero sum).

    Notes:
        - Works for any dimensionality D >= 2.
        - For column-vector convention (``R @ P + t``), the covariance and multiplication
          order must be adjusted.
        - The returned transform is optimal in the least-squares sense and preserves distances
          (no scaling or shearing).

    """

    P = _as_float64(P)
    Q = _as_float64(Q)

    if P.ndim != 2 or Q.ndim != 2 or P.shape != Q.shape:
        msg = "P and Q must have the same shape (N, D)."
        raise ValueError(msg)

    N: int = P.shape[0]
    D: int = P.shape[1]

    if N < D:
        msg = "Need at least D points to determine a D-D rotation."
        raise ValueError(msg)

    # Center the points and build covariance
    if weights is not None:
        w = _as_float64(weights).reshape(-1)
        if w.shape != (N,):
            msg = "weights must be shape (N,)."
            raise ValueError(msg)
        if np.any(w < 0.0):
            msg = "weights must be nonnegative."
            raise ValueError(msg)
        w_sum = float(w.sum())
        if not np.isfinite(w_sum) or w_sum <= 0.0:
            msg = "sum(weights) must be positive and finite."
            raise ValueError(msg)

        Pw = (w[:, None] * P) / w_sum
        Qw = (w[:, None] * Q) / w_sum
        cP = Pw.sum(axis=0)  # shape: (D,)
        cQ = Qw.sum(axis=0)  # shape: (D,)
        P0 = P - cP  # shape: (N, D)
        Q0 = Q - cQ  # shape: (N, D)
        H: FloatArray = P0.T @ (Q0 * w[:, None])  # shape: (D, D)
    else:
        cP = P.mean(axis=0)  # (D,)
        cQ = Q.mean(axis=0)  # (D,)
        P0 = P - cP
        Q0 = Q - cQ
        H = P0.T @ Q0  # (D, D)

    # SVD: H = U S Vt
    U, _, Vt = np.linalg.svd(H)

    # Rotation mapping P -> Q
    R = U @ Vt

    # Enforce right-handedness unless reflections are allowed
    if not allow_reflection and float(np.linalg.det(R)) < 0.0:
        U_flipped = U.copy()
        U_flipped[:, -1] *= -1.0
        R = U_flipped @ Vt

    # Translation
    t = cQ - (cP @ R)

    # Ensure float64 outputs
    return _as_float64(R), _as_float64(t)


def apply_transform(P: FloatArray, R: FloatArray, t: FloatArray) -> FloatArray:
    """Apply affine transform defined by rotation R and translation t to points P."""
    P = _as_float64(P)
    R = _as_float64(R)
    t = _as_float64(t)
    return P @ R + t


def rmsd(
    A: FloatArray,
    B: FloatArray,
    weights: NDArray[np.float64] | None = None,
) -> float:
    """Root mean square deviation between two point sets A and B."""
    A = _as_float64(A)
    B = _as_float64(B)
    if A.shape != B.shape:
        msg = "A and B must have the same shape."
        raise ValueError(msg)
    diff2 = np.sum((A - B) ** 2, axis=1)  # (N,)
    if weights is None:
        return float(np.sqrt(diff2.mean()))
    w = _as_float64(weights).reshape(-1)
    if w.shape[0] != A.shape[0]:
        msg = "weights must have shape (N,)."
        raise ValueError(msg)
    w_sum = float(w.sum())
    if w_sum <= 0.0:
        msg = "sum(weights) must be positive."
        raise ValueError(msg)
    return float(np.sqrt(np.sum(w * diff2) / w_sum))


# --- Minimal example for manual run ---
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 20
    P = rng.normal(size=(N, 3))

    # Create a random proper rotation
    U_rand, _, Vt_rand = np.linalg.svd(rng.normal(size=(3, 3)))
    R_true = U_rand @ Vt_rand
    if np.linalg.det(R_true) < 0:
        U_rand[:, -1] *= -1
        R_true = U_rand @ Vt_rand

    t_true = rng.normal(size=3)
    Q = P @ R_true + t_true

    R_est, t_est = kabsch(P, Q)
    P_aligned = apply_transform(P, R_est, t_est)

    print("RMSD before:", rmsd(P, Q))
    print("RMSD after: ", rmsd(P_aligned, Q))
    print("Rotation error:", np.linalg.norm(R_est - R_true))
    print("Translation error:", np.linalg.norm(t_est - t_true))
