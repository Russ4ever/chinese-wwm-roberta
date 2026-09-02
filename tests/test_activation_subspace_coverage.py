"""Tests for attention subspace coverage analysis."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.activation_subspace_coverage import (
    HIDDEN_DIM,
    compute_projector,
    coverage_summary_row,
    effective_rank,
    load_coverage_policy,
    numerical_invariants,
    pairwise_overlap,
    run_coverage_stage,
)

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Policy tests
# --------------------------------------------------------------------------- #
def test_policy_loads_and_validates():
    policy = load_coverage_policy(
        ROOT / "configs" / "activation_subspace_coverage.yaml"
    )
    assert policy["schema_version"] == "activation_subspace_coverage_policy_v1.0"
    assert 256 in policy["parameters"]["k_grid"]
    assert policy["parameters"]["primary_k"] == 256
    assert policy["parameters"]["n_layers"] == 12
    assert "label" not in json.dumps(policy)


# --------------------------------------------------------------------------- #
# Linear algebra: identical subspaces
# --------------------------------------------------------------------------- #
def test_identical_subspaces_have_overlap_1():
    rng = np.random.default_rng(42)
    V = rng.standard_normal((HIDDEN_DIM, 128))
    V, _ = np.linalg.qr(V)  # orthonormalise
    stats = pairwise_overlap(V, V, 128)
    assert stats["overlap"] == pytest.approx(1.0, abs=1e-10)
    assert stats["excess_overlap"] == pytest.approx(1.0 - 128 / HIDDEN_DIM, abs=1e-8)
    assert stats["principal_cos2_min"] == pytest.approx(1.0, abs=1e-10)


# --------------------------------------------------------------------------- #
# Linear algebra: orthogonal subspaces
# --------------------------------------------------------------------------- #
def test_orthogonal_subspaces_have_overlap_0():
    V1 = np.eye(HIDDEN_DIM, 64)          # first 64 standard basis vectors
    V2 = np.eye(HIDDEN_DIM, 64, 64)      # next 64
    stats = pairwise_overlap(V1, V2, 64)
    assert stats["overlap"] == pytest.approx(0.0, abs=1e-10)
    assert stats["excess_overlap"] == pytest.approx(-64 / HIDDEN_DIM, abs=1e-8)


# --------------------------------------------------------------------------- #
# Random baseline
# --------------------------------------------------------------------------- #
def test_random_subspaces_approximate_k_over_768():
    rng = np.random.default_rng(123)
    k = 128
    overlaps = []
    for _ in range(200):
        V1 = rng.standard_normal((HIDDEN_DIM, k))
        V1, _ = np.linalg.qr(V1)
        V2 = rng.standard_normal((HIDDEN_DIM, k))
        V2, _ = np.linalg.qr(V2)
        overlaps.append(pairwise_overlap(V1, V2, k)["overlap"])
    mean_overlap = float(np.mean(overlaps))
    assert mean_overlap == pytest.approx(k / HIDDEN_DIM, abs=0.02)


# --------------------------------------------------------------------------- #
# Rotation invariance
# --------------------------------------------------------------------------- #
def test_rotation_invariance():
    rng = np.random.default_rng(7)
    V = rng.standard_normal((HIDDEN_DIM, 128))
    V, _ = np.linalg.qr(V)
    # random rotation within the subspace
    R = rng.standard_normal((128, 128))
    R, _ = np.linalg.qr(R)
    V_rot = V @ R
    stats_orig = pairwise_overlap(V, V, 128)
    stats_rot = pairwise_overlap(V_rot, V_rot, 128)
    assert stats_orig["overlap"] == pytest.approx(stats_rot["overlap"], abs=1e-8)

    P_orig = compute_projector(V)
    P_rot = compute_projector(V_rot)
    assert np.allclose(P_orig, P_rot, atol=1e-8)


# --------------------------------------------------------------------------- #
# Numerical invariants
# --------------------------------------------------------------------------- #
def test_projector_invariants():
    rng = np.random.default_rng(99)
    V = rng.standard_normal((HIDDEN_DIM, 256))
    V, _ = np.linalg.qr(V)
    P = compute_projector(V)
    tol = {"orthogonal": 1e-5, "symmetry": 1e-5, "projector": 1e-4, "trace": 1e-6, "psd": 1e-8}
    inv = numerical_invariants(V, P, 256, tol)
    assert inv["orthogonal_passed"]
    assert inv["symmetry_passed"]
    assert inv["projector_passed"]
    assert inv["trace_passed"]
    assert abs(np.trace(P) - 256) < 1e-6
    assert np.allclose(P @ P, P, atol=1e-4)


# --------------------------------------------------------------------------- #
# Coverage spectrum: identical projectors
# --------------------------------------------------------------------------- #
def test_identical_projectors_coverage():
    rng = np.random.default_rng(55)
    V = rng.standard_normal((HIDDEN_DIM, 128))
    V, _ = np.linalg.qr(V)
    P = compute_projector(V)
    # 3 identical projectors
    from src.activation_subspace_coverage import coverage_spectrum
    eigvals, _ = coverage_spectrum([P, P, P])
    summary = coverage_summary_row(eigvals)
    assert summary["effective_coverage_rank"] == pytest.approx(128.0, abs=0.5)
    assert summary["directions_coverage_above_090"] == 128
    assert summary["directions_coverage_below_005"] == HIDDEN_DIM - 128


# --------------------------------------------------------------------------- #
# Cumulative coverage: orthogonal layers grow linearly
# --------------------------------------------------------------------------- #
def test_orthogonal_layers_cumulative_growth():
    # 3 disjoint 64-dim subspaces in 768-dim space
    V1 = np.eye(HIDDEN_DIM, 64)
    V2 = np.eye(HIDDEN_DIM, 64, 64)
    V3 = np.eye(HIDDEN_DIM, 64, 128)
    projectors = [compute_projector(V) for V in (V1, V2, V3)]
    eigvals_m1 = np.linalg.eigvalsh(projectors[0])[::-1]
    eigvals_m3 = np.linalg.eigvalsh(np.mean(projectors, axis=0))[::-1]
    summary_m1 = coverage_summary_row(eigvals_m1)
    summary_m3 = coverage_summary_row(eigvals_m3)
    # 3 disjoint 64-dim subspaces: 192 directions covered, effective rank should be ~192
    assert summary_m3["effective_coverage_rank"] > summary_m1["effective_coverage_rank"]
    assert summary_m3["directions_coverage_above_090"] == 192
