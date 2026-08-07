# Changelog

All notable changes to NCIForge are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-07-12

### Fixed

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
