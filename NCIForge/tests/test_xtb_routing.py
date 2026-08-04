"""Table-driven unit tests for the pure xTB CPU/GPU routing policy.

Covers ``knf_core.engine.xtb_routing.route_xtb`` in isolation (no I/O, no torch,
no subprocess) so the decision layer is locked down independently of the
pipeline plumbing that consumes it.
"""

from __future__ import annotations

import pytest

from knf_core.engine.xtb_routing import route_xtb

CUTOFF = 350


def _route(
    engine="xtbx",
    atom_count=6,
    batch_size=1,
    explicit_gpu=False,
    gpu_available=True,
):
    return route_xtb(
        engine=engine,
        atom_count=atom_count,
        batch_size=batch_size,
        explicit_gpu=explicit_gpu,
        gpu_available=gpu_available,
        large_atom_cutoff=CUTOFF,
    )


# (kwargs, expected_launcher, expected_use_gpu, reason_substring)
CASES = [
    # Explicit stock xtb: never GPU, regardless of size / gpu availability.
    (dict(engine="xtb"), "xtb", False, "stock"),
    (dict(engine="xtb", atom_count=400), "xtb", False, "stock"),
    (dict(engine="xtb", explicit_gpu=True, atom_count=400), "xtb", False, "stock"),
    # No GPU in play for the run: always CPU.
    (dict(engine="xtbx", gpu_available=False), "xtbx", False, "no GPU"),
    (dict(engine="xtbx", gpu_available=False, atom_count=400), "xtbx", False, "no GPU"),
    (dict(engine="auto", gpu_available=False), "xtb", False, "no GPU"),
    (dict(engine="auto", gpu_available=False, atom_count=400), "xtbx", False, "no GPU"),
    # Large molecule with a GPU present: GPU wins big.
    (dict(engine="xtbx", atom_count=400), "xtbx", True, "large"),
    (dict(engine="auto", atom_count=400), "xtbx", True, "large"),
    (dict(engine="xtbx", atom_count=CUTOFF), "xtbx", True, "large"),  # boundary: == cutoff
    (dict(engine="xtbx", atom_count=CUTOFF - 1), "xtbx", False, "single small"),  # just below
    # Single small molecule: CPU (cold start dominates)...
    (dict(engine="xtbx", atom_count=6, batch_size=1), "xtbx", False, "single small"),
    (dict(engine="auto", atom_count=6, batch_size=1), "xtb", False, "single small"),
    (dict(engine="xtbx", atom_count=0, batch_size=1), "xtbx", False, "single small"),  # unknown -> small
    # ...unless the user explicitly prefers GPU for that one molecule.
    (dict(engine="xtbx", atom_count=6, batch_size=1, explicit_gpu=True), "xtbx", True, "single small molecule"),
    (dict(engine="auto", atom_count=6, batch_size=1, explicit_gpu=True), "xtbx", True, "single small molecule"),
    # Many small molecules: stay CPU (parallel); GPU reserved for the NCI stage.
    (dict(engine="xtbx", atom_count=6, batch_size=50), "xtbx", False, "reserved for NCI"),
    (dict(engine="auto", atom_count=6, batch_size=50), "xtb", False, "reserved for NCI"),
    # Many small molecules under explicit --gpu: batch-aware, cold-start won't amortize.
    (dict(engine="xtbx", atom_count=6, batch_size=50, explicit_gpu=True), "xtbx", False, "does not amortize"),
]


@pytest.mark.parametrize("kwargs,launcher,use_gpu,reason_sub", CASES)
def test_route_table(kwargs, launcher, use_gpu, reason_sub):
    decision = _route(**kwargs)
    assert decision.launcher == launcher
    assert decision.use_gpu is use_gpu
    assert reason_sub in decision.reason
    assert decision.reason  # always populated for logging/metadata


def test_large_molecule_beats_batch_shape():
    # A large molecule routes to GPU even inside a big batch and without --gpu.
    decision = _route(engine="xtbx", atom_count=800, batch_size=200, explicit_gpu=False)
    assert decision.use_gpu is True


def test_large_molecule_on_cpu_only_run_stays_cpu():
    # Large, but no GPU configured for the run -> CPU (xtbx CPU build).
    decision = _route(engine="xtbx", atom_count=800, gpu_available=False)
    assert decision.launcher == "xtbx"
    assert decision.use_gpu is False
