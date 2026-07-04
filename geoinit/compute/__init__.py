"""GeoInit compute layer: backend-agnostic array execution.

This package provides a thin abstraction over NumPy (the always-available
single-threaded fallback) and PyTorch (optional CPU multi-threaded / CUDA GPU
acceleration).  The same vectorised energy/gradient kernels in
:mod:`geoinit.energy.engine` run unchanged on whichever backend is selected.

Design goals
------------
* **Single fallback** — NumPy is always available and is the default.  Nothing
  in the core library requires PyTorch.
* **GPU / multi-thread** — when PyTorch is installed, the *same* kernels run on
  CPU (multi-threaded BLAS) or CUDA with no change to the math.
* **Auto-select** — :func:`select_backend` chooses NumPy for tiny problems
  (where host↔device transfer dominates) and the GPU only when the work is
  large enough to amortise the overhead.
"""

from geoinit.compute.backends import (
    Backend,
    NumpyBackend,
    TorchBackend,
    backend_available,
    get_backend,
    select_backend,
)

__all__ = [
    "Backend",
    "NumpyBackend",
    "TorchBackend",
    "backend_available",
    "get_backend",
    "select_backend",
]
