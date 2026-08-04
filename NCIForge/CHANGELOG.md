# Changelog

All notable changes to NCIForge are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-07-30

### Packaging

- Added a reproducible Windows bundle containing a private CPython 3.11
  runtime, the NCIForge API, scientific dependencies, and CPU-only PyTorch
  `2.11.0+cpu`.
- Added an NSIS installer check for NVIDIA hardware. Interactive GPU systems
  can optionally install and validate PyTorch `2.11.0+cu128` in an isolated
  per-user runtime; all other installs retain CPU Torch.
- Added source fingerprinting so unchanged backend runtimes are reused while
  backend code changes force a rebuild.
- Added branded application, header, and sidebar assets generated from the
  approved NCIForge logo directory.
- Added automatic cleanup of the optional CUDA runtime during uninstall.
- Replaced the missing legacy `freeze_backend.ps1` packaging dependency.

### Fixed

- **Problem:** CUDA jobs submitted through FastAPI failed immediately with
  `EOF when reading a line` when PyTorch was absent.
- **Cause:** API preflight called the CLI-oriented Torch resolver, which can
  prompt to install CUDA PyTorch. Uvicorn worker threads have no interactive
  stdin.
- **Changed:** CUDA API preflight now calls the runtime check with
  `allow_prompt=False`, producing a clear API error instead of prompting.
- Added regression tests covering the non-interactive CUDA path and missing
  runtime error.
- Installed and live-verified CUDA PyTorch `2.11.0+cu128` in the F-F
  development environment; an end-to-end 29-atom test completed successfully
  on the RTX 3050.

- **Problem:** Batch runs with live progress callbacks falsely failed after a
  successful xTB single-point calculation. The raw `xtb.log` contained
  `normal termination of xtb` and the expected outputs (`molden.input`, WBO,
  Hessian, and polarizability), but the failure manifest recorded
  `CalledProcessError: ... returned non-zero exit status 0`.
- **Cause:** In `run_xtb_sp()`'s streamed execution path, the
  `CalledProcessError` raise was accidentally dedented outside the
  `if return_code != 0` guard. It therefore ran for both successful and failed
  xTB subprocesses.
- **Changed:** Restored the raise to the non-zero-return-code branch, so a
  streamed xTB single-point calculation that returns `0` proceeds to
  descriptor and NCI processing normally.
- Added a regression test that mocks a streamed xTB single-point return code
  of `0` and verifies that no exception is raised.

### Performance

- Serialize real CUDA use between GPU-routed `xtbx` subprocesses and the
  Torch NCI CUDA stage to avoid device contention in mixed-size batches.
- Vectorized Torch NCI basis evaluation by grouping basis functions with the
  same Cartesian power tuple, reducing small GPU kernel launches.
- Reuse a parsed Molden wavefunction between native WBO and Torch NCI stages,
  avoiding a duplicate parser pass.
- Preserve the CUDA allocator cache after successful NCI runs; allocator
  cleanup remains on failure paths.
- Load each NCI grid payload once when computing both SNCI and NCI statistics.

### Validation

- Added numerical-equivalence coverage for grouped basis evaluation and
  reusable wavefunctions.
- Verified baseline-equivalent NCI grid outputs on the included Molden test
  system and exercised CUDA performance on an RTX 3050.

## [1.0.8]

Current released package version prior to the unreleased changes above.
