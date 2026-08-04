# NCIForge v1

## Highlights
- Rebranded CLI and runtime surfaces to `NCIForge`.
- Added `nciforge` as primary command while keeping `knf` as compatibility alias.
- Removed `kuid` as a callable CLI entry point.
- Updated live CLI headers and status panels to reflect new branding.
- Added compact batch status line with done/failed/running/jobs-per-minute.
- Updated output path rendering in live UI to avoid long-path overflow.
- Updated Docker image/service naming and entrypoint defaults to `nciforge`.
- Updated documentation and release workflow to align with the new naming.

## Compatibility
- Existing command `knf` remains valid.
- KUID/KUID-Intensive output artifacts and schema remain supported.

