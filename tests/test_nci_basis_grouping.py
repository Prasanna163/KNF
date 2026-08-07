"""Regression test for the grouped/vectorized basis evaluation in nci_torch.

``engine._evaluate_basis_chunk`` used to loop in pure Python over every basis
function, launching a handful of tiny tensor ops per function per grid chunk.
It was rewritten to bucket basis functions by (lx, ly, lz) power triple and
evaluate each bucket with one batched op covering every basis function (and
every zero-padded primitive) in that bucket at once. This test locks in that
the rewrite is numerically identical to the original per-function formula by
recomputing the same basis values with a naive reference loop and comparing.
"""

import torch

from knf_core.nci_torch.engine import _group_prepared_basis, _evaluate_basis_chunk
from knf_core.nci_torch.types import PreparedBasisFunction


def _reference_evaluate(points: torch.Tensor, prepared_basis: list[PreparedBasisFunction]) -> torch.Tensor:
    """Pre-rewrite formula: one Python-level iteration per basis function."""
    n_points = points.shape[0]
    n_basis = len(prepared_basis)
    out = torch.empty((n_points, n_basis), device=points.device, dtype=points.dtype)

    for col, bf in enumerate(prepared_basis):
        dx = points[:, 0] - bf.center[0]
        dy = points[:, 1] - bf.center[1]
        dz = points[:, 2] - bf.center[2]
        r2 = dx * dx + dy * dy + dz * dz

        lx, ly, lz = bf.powers
        poly = torch.ones_like(r2)
        if lx:
            poly = poly * (dx**lx)
        if ly:
            poly = poly * (dy**ly)
        if lz:
            poly = poly * (dz**lz)

        prim = torch.exp(-r2[:, None] * bf.exponents[None, :]) * bf.coefficients[None, :]
        out[:, col] = poly * prim.sum(dim=1)

    return out


def _make_synthetic_basis(seed: int = 0) -> list[PreparedBasisFunction]:
    """A handful of s/p/d basis functions, on different centers, with
    deliberately *different* primitive counts (2, 3, and 5) so zero-padding
    across a group is actually exercised."""
    gen = torch.Generator().manual_seed(seed)
    dtype = torch.float64  # tight tolerance for the equivalence check

    powers_pool = [
        (0, 0, 0),  # s
        (1, 0, 0), (0, 1, 0), (0, 0, 1),  # p
        (2, 0, 0), (1, 1, 0), (0, 1, 1),  # d (subset)
    ]
    prim_counts = [2, 3, 5, 3, 2, 4, 3]

    basis = []
    for powers, n_prim in zip(powers_pool, prim_counts):
        center = torch.rand(3, generator=gen, dtype=dtype) * 4.0 - 2.0
        exponents = torch.rand(n_prim, generator=gen, dtype=dtype) * 2.0 + 0.2
        coefficients = torch.rand(n_prim, generator=gen, dtype=dtype) * 2.0 - 1.0
        basis.append(
            PreparedBasisFunction(
                center=center,
                powers=powers,
                exponents=exponents,
                coefficients=coefficients,
            )
        )

    # Duplicate the pool with a second, independent center per power triple so
    # groups actually contain more than one basis function (the case the
    # rewrite optimizes for).
    for powers, n_prim in zip(powers_pool, prim_counts):
        center = torch.rand(3, generator=gen, dtype=dtype) * 4.0 - 2.0
        exponents = torch.rand(n_prim + 1, generator=gen, dtype=dtype) * 2.0 + 0.2
        coefficients = torch.rand(n_prim + 1, generator=gen, dtype=dtype) * 2.0 - 1.0
        basis.append(
            PreparedBasisFunction(
                center=center,
                powers=powers,
                exponents=exponents,
                coefficients=coefficients,
            )
        )

    return basis


def test_grouped_evaluation_matches_naive_reference():
    prepared_basis = _make_synthetic_basis(seed=42)
    n_basis = len(prepared_basis)

    gen = torch.Generator().manual_seed(7)
    points = torch.rand(500, 3, generator=gen, dtype=torch.float64) * 6.0 - 3.0

    expected = _reference_evaluate(points, prepared_basis)

    basis_groups = _group_prepared_basis(prepared_basis)
    actual = _evaluate_basis_chunk(points, basis_groups, n_basis)

    assert actual.shape == expected.shape == (points.shape[0], n_basis)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)


def test_grouped_evaluation_covers_points_on_top_of_a_center():
    # r2 == 0 exercises the zero-padding path (exp(0) * 0 must stay exactly 0)
    # and integer powers of zero for p/d functions.
    prepared_basis = _make_synthetic_basis(seed=1)
    n_basis = len(prepared_basis)
    centers = torch.stack([bf.center for bf in prepared_basis], dim=0)

    expected = _reference_evaluate(centers, prepared_basis)
    basis_groups = _group_prepared_basis(prepared_basis)
    actual = _evaluate_basis_chunk(centers, basis_groups, n_basis)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)


def test_grouping_buckets_by_power_triple_and_preserves_column_order():
    prepared_basis = _make_synthetic_basis(seed=3)
    basis_groups = _group_prepared_basis(prepared_basis)

    # Every basis function must appear in exactly one group.
    all_indices = sorted(idx for group in basis_groups for idx in group.indices)
    assert all_indices == list(range(len(prepared_basis)))

    # Groups are keyed by (and only by) the power triple.
    for group in basis_groups:
        for idx in group.indices:
            assert prepared_basis[idx].powers == group.powers

    # There are 7 distinct power triples in the synthetic set (built with two
    # basis functions per triple), regardless of there being 14 basis
    # functions total -- this is the whole point of the rewrite.
    assert len(basis_groups) == 7
