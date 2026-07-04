"""Array backends for GeoInit's vectorised compute layer.

A :class:`Backend` exposes the small set of array primitives the energy/gradient
kernels need, implemented identically for NumPy and PyTorch.  Kernels are written
once against this interface and run unchanged on CPU (NumPy / Torch) or CUDA
(Torch).

The primitives are deliberately minimal:

``asarray, to_numpy, zeros, sqrt, sum, clip, norm_rows, gather_rows, scatter_add``

Everything else (``+``, ``-``, ``*``, ``/``, ``**``, row indexing ``X[idx]``) uses
native operators that behave the same on ``np.ndarray`` and ``torch.Tensor``.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
#  Optional PyTorch — imported *lazily* so the NumPy fallback never pays the
#  ~1-2 s ``import torch`` cost.  We only check that the package exists up front
#  (cheap, no import); the heavy module load happens when a Torch backend is
#  actually constructed.
# --------------------------------------------------------------------------- #
_TORCH_INSTALLED = importlib.util.find_spec("torch") is not None
torch = None  # type: ignore[assignment]  # populated lazily by _ensure_torch()


def _ensure_torch():
    """Import PyTorch on first real use and cache the module globally."""
    global torch
    if torch is None:
        import torch as _t  # type: ignore[import-not-found]

        torch = _t
    return torch


def torch_available() -> bool:
    """Return ``True`` if PyTorch is installed (does not import it)."""
    return _TORCH_INSTALLED


_CUDA_PROBE: bool | None = None


def cuda_available() -> bool:
    """Return ``True`` if PyTorch is present *and* a CUDA device is usable.

    The probe (``torch.cuda.is_available()``) can initialise a CUDA context,
    which costs seconds the first time in a process — so the result is cached.
    Tiny problems avoid calling this at all (see :func:`select_backend`).
    """
    global _CUDA_PROBE
    if _CUDA_PROBE is not None:
        return _CUDA_PROBE
    if not _TORCH_INSTALLED:
        _CUDA_PROBE = False
        return False
    try:  # pragma: no cover - depends on hardware
        _CUDA_PROBE = bool(_ensure_torch().cuda.is_available())
    except Exception:  # pragma: no cover
        _CUDA_PROBE = False
    return _CUDA_PROBE


# --------------------------------------------------------------------------- #
#  Backend interface
# --------------------------------------------------------------------------- #

class Backend:
    """Abstract array backend.  Subclasses implement the primitives below."""

    name: str = "abstract"
    device: str = "cpu"
    #: True when the gradient is obtained by reverse-mode autodiff rather than
    #: the hand-written analytic kernel.  Used by the engine to pick a path.
    supports_autograd: bool = False

    # -- construction / transfer ------------------------------------------- #
    def asarray(self, x: Any, dtype: Any = None) -> Any:
        raise NotImplementedError

    def to_numpy(self, x: Any) -> np.ndarray:
        raise NotImplementedError

    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        raise NotImplementedError

    # -- elementwise / reductions ------------------------------------------ #
    def sqrt(self, x: Any) -> Any:
        raise NotImplementedError

    def sum(self, x: Any, axis: int | None = None) -> Any:
        raise NotImplementedError

    def clip(self, x: Any, lo: float | None, hi: float | None) -> Any:
        raise NotImplementedError

    def norm_rows(self, x: Any) -> Any:
        """Euclidean norm along the last axis of an ``(M, 3)`` array → ``(M,)``."""
        return self.sqrt(self.sum(x * x, axis=-1))

    # -- scatter / gather --------------------------------------------------- #
    def scatter_add(self, out: Any, idx: Any, vals: Any) -> None:
        """In-place ``out[idx[k]] += vals[k]`` with duplicate-safe accumulation."""
        raise NotImplementedError

    # -- float scalar extraction ------------------------------------------- #
    def item(self, x: Any) -> float:
        return float(self.to_numpy(x))


# --------------------------------------------------------------------------- #
#  NumPy backend (always available — the single-threaded fallback)
# --------------------------------------------------------------------------- #

class NumpyBackend(Backend):
    name = "numpy"
    device = "cpu"
    supports_autograd = False

    def __init__(self, dtype: Any = np.float64) -> None:
        self.dtype = dtype

    def asarray(self, x: Any, dtype: Any = None) -> np.ndarray:
        return np.asarray(x, dtype=dtype or self.dtype)

    def to_numpy(self, x: Any) -> np.ndarray:
        return np.asarray(x)

    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> np.ndarray:
        return np.zeros(shape, dtype=dtype or self.dtype)

    def sqrt(self, x: Any) -> np.ndarray:
        return np.sqrt(x)

    def sum(self, x: Any, axis: int | None = None) -> Any:
        return np.sum(x, axis=axis)

    def clip(self, x: Any, lo: float | None, hi: float | None) -> np.ndarray:
        return np.clip(x, lo, hi)

    def scatter_add(self, out: np.ndarray, idx: np.ndarray, vals: np.ndarray) -> None:
        # np.add.at is the duplicate-safe unbuffered accumulator.
        np.add.at(out, idx, vals)


# --------------------------------------------------------------------------- #
#  PyTorch backend (CPU multi-thread or CUDA GPU)
# --------------------------------------------------------------------------- #

class TorchBackend(Backend):
    name = "torch"
    supports_autograd = True

    def __init__(self, device: str = "cpu", dtype: Any = None) -> None:
        if not _TORCH_INSTALLED:  # pragma: no cover
            raise RuntimeError("PyTorch is not available; cannot create TorchBackend.")
        _ensure_torch()
        if device == "cuda" and not cuda_available():  # pragma: no cover
            raise RuntimeError("CUDA requested but not available.")
        self.device = device
        self.dtype = dtype if dtype is not None else torch.float64
        self._index_dtype = torch.long

    def asarray(self, x: Any, dtype: Any = None) -> Any:
        if isinstance(x, torch.Tensor):
            t = x
            if dtype is not None:
                t = t.to(dtype)
            return t.to(self.device)
        return torch.as_tensor(np.asarray(x), dtype=dtype or self.dtype, device=self.device)

    def as_index(self, x: Any) -> Any:
        """Return an integer index tensor on the backend device."""
        if isinstance(x, torch.Tensor):
            return x.to(self._index_dtype).to(self.device)
        return torch.as_tensor(np.asarray(x), dtype=self._index_dtype, device=self.device)

    def to_numpy(self, x: Any) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().to("cpu").numpy()
        return np.asarray(x)

    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        return torch.zeros(shape, dtype=dtype or self.dtype, device=self.device)

    def sqrt(self, x: Any) -> Any:
        return torch.sqrt(x)

    def sum(self, x: Any, axis: int | None = None) -> Any:
        if axis is None:
            return torch.sum(x)
        return torch.sum(x, dim=axis)

    def clip(self, x: Any, lo: float | None, hi: float | None) -> Any:
        return torch.clamp(x, min=lo, max=hi)

    def scatter_add(self, out: Any, idx: Any, vals: Any) -> None:
        # index_add_ accumulates rows; idx must be a long tensor on-device.
        out.index_add_(0, self.as_index(idx), vals)

    def item(self, x: Any) -> float:
        if isinstance(x, torch.Tensor):
            return float(x.detach().to("cpu").item())
        return float(x)


# --------------------------------------------------------------------------- #
#  Selection / factory
# --------------------------------------------------------------------------- #

#: Below this many *interactions* (bonds+angles+nonbonded+rigid pairs) the GPU
#: host↔device transfer overhead dwarfs the compute, so NumPy is faster per call.
#: Measured crossover on a laptop RTX 3050: NumPy wins up to ~5e4 interactions; CUDA
#: pulls clearly ahead (>2x) by ~1e5.  Override via ``GEOINIT_GPU_MIN_WORK``.
_GPU_MIN_WORK = int(os.environ.get("GEOINIT_GPU_MIN_WORK", "50000"))

#: Above this many interactions, a multi-threaded Torch-CPU backend beats single
#: -threaded NumPy even without a GPU.  Override via ``GEOINIT_TORCH_CPU_MIN_WORK``.
_TORCH_CPU_MIN_WORK = int(os.environ.get("GEOINIT_TORCH_CPU_MIN_WORK", "60000"))


def backend_available(name: str) -> bool:
    """Return whether a named backend (``'numpy'``/``'torch'``/``'cuda'``) can run."""
    if name == "numpy":
        return True
    if name in ("torch", "torch-cpu"):
        return torch_available()
    if name in ("cuda", "torch-cuda", "gpu"):
        return cuda_available()
    return False


def get_backend(name: str, dtype: Any = None) -> Backend:
    """Instantiate a backend by name.

    Names: ``'numpy'``, ``'torch'``/``'torch-cpu'``, ``'cuda'``/``'torch-cuda'``/``'gpu'``.
    Falls back to NumPy if the requested backend is unavailable.
    """
    if name == "numpy":
        return NumpyBackend(dtype=dtype or np.float64)
    if name in ("torch", "torch-cpu"):
        if not torch_available():
            return NumpyBackend(dtype=dtype or np.float64)
        return TorchBackend(device="cpu", dtype=dtype)
    if name in ("cuda", "torch-cuda", "gpu"):
        if not cuda_available():
            # Graceful degradation: prefer torch-cpu, else numpy.
            if torch_available():
                return TorchBackend(device="cpu", dtype=dtype)
            return NumpyBackend(dtype=dtype or np.float64)
        return TorchBackend(device="cuda", dtype=dtype)
    raise ValueError(f"Unknown backend '{name}'.")


def select_backend(
    work_size: int,
    prefer: str = "auto",
    dtype: Any = None,
) -> Backend:
    """Choose a backend for a problem of the given interaction count.

    Parameters
    ----------
    work_size : int
        Total number of pairwise/triplet interactions (proxy for compute cost).
    prefer : str
        ``'auto'`` (size-aware), or a forced backend name handled by
        :func:`get_backend`, or ``'cpu'`` (numpy).
    dtype :
        Optional floating dtype override.

    Returns
    -------
    Backend
        The instantiated backend.  Always returns a usable backend; never raises
        for hardware reasons (degrades to NumPy).
    """
    if prefer in ("cpu",):
        return NumpyBackend(dtype=dtype or np.float64)
    if prefer != "auto":
        return get_backend(prefer, dtype=dtype)

    # Auto: tiered escalation by problem size.
    #   small  → NumPy        (single-thread, zero launch/transfer overhead)
    #   medium → Torch-CPU    (multi-threaded BLAS, no device transfer)
    #   large  → CUDA         (GPU wins once transfer is amortised)
    # The size thresholds are checked *before* probing the hardware so that small
    # problems (the overwhelming majority) never pay the CUDA-context init cost.
    if work_size >= _GPU_MIN_WORK and cuda_available():
        return TorchBackend(device="cuda", dtype=dtype)
    if work_size >= _TORCH_CPU_MIN_WORK and torch_available():
        return TorchBackend(device="cpu", dtype=dtype)
    return NumpyBackend(dtype=dtype or np.float64)
