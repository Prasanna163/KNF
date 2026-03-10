# KNF-Core v1.0.5 (KNF-GPU)

## Highlights
- Native Molden-based WBO is now the default path for `f3` (`--wbo-mode native`).
- Added optional `--wbo-mode xtb` fallback to parse xTB `wbo` files.
- Added H-bond interaction seeding for two-fragment systems before UFF/xTB stages.
- Added WBO diagnostics in output metadata (`wbo_max_global`, inter-pair diagnostics, overlap model).
- Updated Docker/version naming references and release checklist.

## Compatibility
- Default behavior is now `--wbo-mode native`.
- Existing xTB-WBO behavior is available via `--wbo-mode xtb`.
