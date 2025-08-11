from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from chemfit.kabsch import apply_transform, kabsch, rmsd

FloatArray = NDArray[np.float64]


def random_proper_rotation(D: int, rng: np.random.Generator) -> FloatArray:
    """
    Helper: generate a random orthogonal matrix with det=+1 (a proper rotation).

    We use QR and, if needed, flip a column to enforce a right-handed rotation.
    """
    A: FloatArray = np.asarray(rng.normal(size=(D, D)), dtype=np.float64)
    Q, _ = np.linalg.qr(A)  # type: ignore[assignment]
    # Ensure det=+1
    if float(np.linalg.det(Q)) < 0.0:
        Q[:, -1] *= -1.0
    return np.asarray(Q, dtype=np.float64)


def make_correspondences(
    N: int, D: int, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """
    Helper: synthesize paired point sets P and Q with a known ground-truth.

    Rotation R and translation t such that Q = P @ R + t.
    """
    P: FloatArray = np.asarray(rng.normal(size=(N, D)), dtype=np.float64)
    R: FloatArray = random_proper_rotation(D, rng)
    t: FloatArray = np.asarray(rng.normal(size=(D,)), dtype=np.float64)
    Q: FloatArray = P @ R + t
    return P, Q, R, t


def test_identity_alignment() -> None:
    """
    When P == Q, the optimal transform is the identity rotation and zero translation.

    RMSD after alignment should be essentially zero.
    """
    rng = np.random.default_rng(0)
    P: FloatArray = np.asarray(rng.normal(size=(30, 3)), dtype=np.float64)
    Q: FloatArray = P.copy()
    R, t = kabsch(P, Q)
    assert np.allclose(R, np.eye(3), atol=1e-12)
    assert np.allclose(t, np.zeros(3), atol=1e-12)
    assert rmsd(apply_transform(P, R, t), Q) < 1e-12


def test_random_3d_alignment_no_reflection() -> None:
    """
    For a random 3D rigid transform with a proper rotation (det=+1).

    Kabsch should recover R and t (within numerical tolerance) when reflections are disallowed.
    """
    rng = np.random.default_rng(1)
    P, Q, R_true, t_true = make_correspondences(50, 3, rng)
    R, t = kabsch(P, Q, allow_reflection=False)
    P_aligned = apply_transform(P, R, t)
    # Rotation must be proper (right-handed)
    assert np.isclose(float(np.linalg.det(R)), 1.0, atol=1e-8)
    # Alignment should be extremely tight
    assert rmsd(P_aligned, Q) < 1e-10
    # Parameter recovery up to tiny numerical error
    assert np.linalg.norm(R - R_true) < 1e-8
    assert np.linalg.norm(t - t_true) < 1e-8


def test_noise_robustness_rmsd_decreases() -> None:
    """
    With small Gaussian noise added to Q, Kabsch should still significantly reduce RMSD.

    We don't expect exact recovery, but the post-fit RMSD should be much smaller than pre-fit.
    """
    rng = np.random.default_rng(2)
    P, Q_clean, _, _ = make_correspondences(200, 3, rng)
    noise: FloatArray = np.asarray(
        0.01 * rng.normal(size=Q_clean.shape), dtype=np.float64
    )
    Q_noisy: FloatArray = Q_clean + noise
    before = rmsd(P, Q_noisy)
    R, t = kabsch(P, Q_noisy)
    after = rmsd(apply_transform(P, R, t), Q_noisy)
    # Expect a drop by ~two orders of magnitude (heuristic but strong)
    assert after < before * 1e-2


def test_translation_only() -> None:
    """
    If Q differs from P by translation only.

    The optimal rotation is identity and the recovered translation should match exactly.
    """
    rng = np.random.default_rng(3)
    P: FloatArray = np.asarray(rng.normal(size=(40, 3)), dtype=np.float64)
    t_true: FloatArray = np.asarray([0.5, -1.2, 3.3], dtype=np.float64)
    Q: FloatArray = P + t_true
    R, t = kabsch(P, Q)
    assert np.allclose(R, np.eye(3), atol=1e-12)
    assert np.allclose(t, t_true, atol=1e-12)


def test_rotation_only() -> None:
    """
    If Q differs from P by rotation only.

    The optimal translation is zero and the recovered rotation should match exactly.
    """

    rng = np.random.default_rng(4)
    P: FloatArray = np.asarray(rng.normal(size=(60, 3)), dtype=np.float64)
    R_true: FloatArray = random_proper_rotation(3, rng)
    Q: FloatArray = P @ R_true
    R, t = kabsch(P, Q)

    print(R)
    print(R_true)

    assert np.allclose(t, np.zeros(3), atol=1e-12)
    assert np.allclose(R, R_true, atol=1e-10)


def test_2d_alignment() -> None:
    """
    The algorithm is dimension-agnostic.

    In 2D it should still perfectly recover a proper rotation and the exact translation.
    """
    rng = np.random.default_rng(5)
    P, Q, R_true, t_true = make_correspondences(25, 2, rng)
    R, t = kabsch(P, Q)
    assert np.isclose(float(np.linalg.det(R)), 1.0, atol=1e-8)
    assert np.linalg.norm(R - R_true) < 1e-10
    assert np.linalg.norm(t - t_true) < 1e-10


def test_reflection_behavior_toggle() -> None:
    """
    Test behaviour under reflection.

    If Q is a reflected version of P, then:
      - When allow_reflection=False, the returned rotation must be proper (det=+1).
        A proper rotation cannot realize a mirror flip, so alignment error will not be ~0.
      - When allow_reflection=True, an improper rotation (det=-1) is allowed and
        the alignment error should be ~0.
    """
    rng = np.random.default_rng(6)
    P: FloatArray = np.asarray(rng.normal(size=(50, 3)), dtype=np.float64)
    reflect: FloatArray = np.asarray(np.diag([1.0, 1.0, -1.0]), dtype=np.float64)
    t_true: FloatArray = np.asarray([0.2, -0.3, 0.4], dtype=np.float64)
    Q_reflected: FloatArray = P @ reflect + t_true

    # Proper-only fit
    R_no_reflect, t_nr = kabsch(P, Q_reflected, allow_reflection=False)
    assert np.isclose(float(np.linalg.det(R_no_reflect)), 1.0, atol=1e-8)

    # Reflection-allowed fit
    R_yes_reflect, t_yr = kabsch(P, Q_reflected, allow_reflection=True)
    assert np.isclose(float(np.linalg.det(R_yes_reflect)), -1.0, atol=1e-8)

    rms_after_nr = rmsd(apply_transform(P, R_no_reflect, t_nr), Q_reflected)
    rms_after_yr = rmsd(apply_transform(P, R_yes_reflect, t_yr), Q_reflected)

    # With reflection allowed, alignment should be essentially perfect.
    assert rms_after_yr < 1e-10

    # With reflection disallowed, alignment cannot be perfect; it should be
    # clearly worse than the reflection-allowed case.
    assert rms_after_nr > rms_after_yr * 1e3  # ratio guard
    # and not numerically tiny:
    assert rms_after_nr > 1e-6


def test_equal_weights_matches_unweighted() -> None:
    """
    Using uniform weights should yield exactly the same result as the unweighted fit.

    This guards the weighted path against numerical or implementation drift.
    """
    rng = np.random.default_rng(7)
    P, Q, _, _ = make_correspondences(30, 3, rng)
    w: FloatArray = np.ones(P.shape[0], dtype=np.float64)
    Rw, tw = kabsch(P, Q, weights=w)
    Ru, tu = kabsch(P, Q, weights=None)
    assert np.allclose(Rw, Ru, atol=1e-12)
    assert np.allclose(tw, tu, atol=1e-12)


def test_weighted_skews_toward_heavy_points() -> None:
    """
    With two clusters where one has much larger weights, the fitted transform should preferentially minimize error on the heavy-weight cluster.

    We compare RMSD on the heavy cluster between weighted and unweighted fits.
    """
    rng = np.random.default_rng(8)
    P1: FloatArray = np.asarray(
        rng.normal(loc=0.0, scale=0.5, size=(50, 3)), dtype=np.float64
    )
    P2: FloatArray = np.asarray(
        rng.normal(loc=5.0, scale=0.5, size=(10, 3)), dtype=np.float64
    )
    P: FloatArray = np.vstack([P1, P2])

    R_true: FloatArray = random_proper_rotation(3, rng)
    t_true: FloatArray = np.asarray([1.0, -2.0, 0.5], dtype=np.float64)
    Q: FloatArray = P @ R_true + t_true

    # Heavy weights for P1, very light for P2
    w: FloatArray = np.hstack([np.full(len(P1), 10.0), np.full(len(P2), 0.1)]).astype(
        np.float64
    )

    Rw, tw = kabsch(P, Q, weights=w)
    P_aligned_w = apply_transform(P, Rw, tw)
    R, t = kabsch(P, Q)
    P_aligned_u = apply_transform(P, R, t)

    err_heavy_w = rmsd(P_aligned_w[: len(P1)], Q[: len(P1)])
    err_heavy_u = rmsd(P_aligned_u[: len(P1)], Q[: len(P1)])
    # Weighted fit should do noticeably better on the heavy cluster
    assert err_heavy_w <= err_heavy_u * 0.8


def test_mismatched_shapes_error() -> None:
    """
    Guardrail: P and Q must be the same shape (N, D).

    Otherwise the function should raise ValueError rather than proceeding with undefined behavior.
    """
    P: FloatArray = np.zeros((5, 3), dtype=np.float64)
    Q: FloatArray = np.zeros((6, 3), dtype=np.float64)
    with pytest.raises(ValueError):  # noqa: PT011
        kabsch(P, Q)


def test_bad_weights_errors() -> None:
    """
    Guardrail: weights must be nonnegative, length N, and have a positive sum.

    Each of the cases below should raise ValueError.
    """
    P: FloatArray = np.zeros((5, 3), dtype=np.float64)
    Q: FloatArray = np.zeros((5, 3), dtype=np.float64)
    with pytest.raises(ValueError):  # noqa: PT011
        kabsch(P, Q, weights=np.asarray([1, 1, -1, 1, 1], dtype=np.float64))
    with pytest.raises(ValueError):  # noqa: PT011
        kabsch(P, Q, weights=np.asarray([1, 1, 1], dtype=np.float64))
    with pytest.raises(ValueError):  # noqa: PT011
        kabsch(P, Q, weights=np.zeros(5, dtype=np.float64))


def test_too_few_points_for_dimension() -> None:
    """
    Guardrail: you need at least D points in D dimensions to determine a D-D rotation.

    For N < D the function should raise ValueError.
    """
    P: FloatArray = np.zeros((2, 3), dtype=np.float64)
    Q: FloatArray = np.zeros((2, 3), dtype=np.float64)
    with pytest.raises(ValueError):  # noqa: PT011
        kabsch(P, Q)


def test_rmsd_weighted_matches_definition() -> None:
    """
    Sanity check for the RMSD helper.

    Verify that the weighted RMSD produced by
    the helper matches the manual formula sqrt(sum(w * ||a_i - b_i||^2) / sum(w)).
    """

    rng = np.random.default_rng(9)
    A: FloatArray = np.asarray(rng.normal(size=(100, 3)), dtype=np.float64)
    B: FloatArray = A + 0.1
    w: FloatArray = np.asarray(rng.random(100) + 0.01, dtype=np.float64)
    diff2: FloatArray = np.sum((A - B) ** 2, axis=1)
    manual = float(np.sqrt(np.sum(w * diff2) / float(np.sum(w))))
    assert np.isclose(manual, rmsd(A, B, weights=w))
