"""Throughput-aware CPU/GPU routing for the per-molecule xTB stage.

This is a pure decision layer (no I/O, no subprocess, no torch) so it can be
exhaustively unit-tested in isolation. It replaces the old blunt rule
``xtb_force_gpu = (nci_backend == 'torch' and nci_device == 'cuda')`` which
forced ``xtbx --gpu`` on *every* molecule whenever the NCI grid happened to run
on CUDA -- a cold CUDA context start (~1s, per the measured xtbx numbers) on
every small molecule, and N concurrent cold starts fighting the NCI stage for
the GPU in batch mode.

The core insight (see ``nciforge_xtbx/xtbx_run.sh`` header): the GPU wins big on
LARGE systems and on high throughput *when a CUDA context is reused*, but KNF
launches xtb as a fresh subprocess per molecule and needs per-compound
molden/wbo/hess artifacts that xtbx's shared-context ``--gpu-batch`` pool does
not produce. So per-molecule forced GPU never amortizes here. Routing policy:

    explicit stock 'xtb'          -> CPU (stock build has no GPU path)
    no GPU in play for this run   -> CPU
    large molecule (>= cutoff)    -> GPU (measured ~10-32x win; CPU impractical)
    single small molecule         -> CPU (GPU cold-start dominates)
    many small molecules          -> CPU, run in parallel across workers; the
                                     GPU is reserved for the NCI grid stage
                                     (which is where the GPU win + a warm CUDA
                                     context, via NCIDeviceRouter, actually live)
    explicit --gpu, single small  -> GPU (honor the user override)

A molecule's atom count is the same for its opt and single-point calls, so both
stages route identically for a given input.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class XtbRoutingDecision:
    """Resolved routing for one xTB invocation."""

    launcher: str  # "xtb" (stock CPU build) | "xtbx" (unified CPU/GPU front-end)
    use_gpu: bool  # whether to append --gpu to the xtbx invocation
    reason: str    # human-readable, always populated (logged + stored in metadata)


def _auto_engine_launcher(engine: str, is_large: bool) -> str:
    """Launcher for the size-gated 'auto' engine; 'xtbx' otherwise.

    'auto' preserves KNF's historical behavior of running small systems on the
    stock CPU ``xtb`` build and routing large systems to ``xtbx``. The default
    'xtbx' engine always uses the ``xtbx`` launcher (its CPU build handles small
    systems in ~50ms with no CUDA dependency).
    """
    if engine == "auto":
        return "xtbx" if is_large else "xtb"
    return "xtbx"


def route_xtb(
    *,
    engine: str,
    atom_count: int,
    batch_size: int,
    explicit_gpu: bool,
    gpu_available: bool,
    large_atom_cutoff: int,
) -> XtbRoutingDecision:
    """Decide launcher + GPU use for one molecule's xTB stage.

    Parameters
    ----------
    engine:
        ``xtb`` (stock CPU), ``xtbx`` (default unified front-end), or ``auto``
        (size-gate the launcher between the two).
    atom_count:
        Atom count of the molecule about to run (0 = unknown -> treated small).
    batch_size:
        Number of molecules in the enclosing workload (1 = single-file run).
    explicit_gpu:
        The user asked to prefer the GPU (the ``--gpu`` shortcut). Distinct from
        merely running the NCI grid on CUDA.
    gpu_available:
        A GPU is in play for this run (i.e. the NCI device is CUDA, which is only
        set when a CUDA-capable GPU was detected/validated).
    large_atom_cutoff:
        ``--xtb-gpu-atoms`` (default 350) -- systems at/above this route to GPU.
    """
    engine = (engine or "xtbx").strip().lower()
    n = int(atom_count or 0)
    is_large = bool(n and n >= int(large_atom_cutoff))
    is_batch = int(batch_size or 1) > 1

    # Explicit stock CPU build: never GPU (it has no GPU code path at all).
    if engine == "xtb":
        return XtbRoutingDecision("xtb", False, "explicit stock xtb -> CPU")

    # No GPU configured for this run: stay on CPU regardless of size.
    if not gpu_available:
        return XtbRoutingDecision(
            _auto_engine_launcher(engine, is_large),
            False,
            "no GPU in play for this run -> CPU",
        )

    # Large system: GPU is a large, unambiguous win; CPU would be impractical.
    if is_large:
        return XtbRoutingDecision(
            "xtbx",
            True,
            f"large system ({n} atoms >= {large_atom_cutoff}) -> GPU",
        )

    # Small system, GPU present.
    if explicit_gpu and not is_batch:
        return XtbRoutingDecision(
            "xtbx", True, "user --gpu on a single small molecule -> GPU"
        )

    if explicit_gpu and is_batch:
        # Batch-aware honoring of --gpu: each xtb call is a fresh process, so a
        # forced-GPU small run pays a full CUDA cold start that never amortizes
        # for full-artifact (molden/wbo/hess) runs. Keep small molecules on the
        # CPU (parallel across workers); the GPU serves the NCI grid stage.
        return XtbRoutingDecision(
            _auto_engine_launcher(engine, is_large),
            False,
            "user --gpu but many small molecules -> CPU per molecule "
            "(cold-start does not amortize); GPU reserved for NCI stage",
        )

    # auto/xtbx, small, no explicit GPU preference.
    if is_batch:
        return XtbRoutingDecision(
            _auto_engine_launcher(engine, is_large),
            False,
            "many small molecules -> CPU in parallel; GPU reserved for NCI stage",
        )
    return XtbRoutingDecision(
        _auto_engine_launcher(engine, is_large),
        False,
        "single small molecule -> CPU",
    )
