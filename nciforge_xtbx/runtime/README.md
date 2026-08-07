Bundled xtbx runtime
====================

This directory is the NCIForge-owned xtbx runtime slot.

The repository includes a compact native Windows runtime:

- `bin/xtb-cpu.exe`
- `bin/xtb.exe`
- xTB parameter files
- small MinGW/OpenBLAS/Fortran runtime DLLs

The very large NVIDIA CUDA redistributable DLLs are not committed here because
several individual files are hundreds of megabytes and exceed normal GitHub/PyPI
package limits. For explicit GPU execution, `xtbx` scans configured paths,
`PATH`, common xTB runtime locations, CUDA Toolkit folders, and the active
Python/Torch environment. If it finds the needed CUDA DLLs, it creates a managed
runtime under the NCIForge config directory. If local assembly is not possible,
it downloads the pinned runtime archive from the Prasanna163/xtb GitHub release
and verifies its SHA256 before extraction. You can also configure a full
`xtb-win-release` folder directly:

```powershell
xtbx --setup-gpu
```

or:

```powershell
xtbx --setup-gpu "C:\path\to\xtb-win-release"
```

For one process without saving config, set `NCIFORGE_XTBX_RUNTIME` or pass
`--gpu-runtime`.

Normal small-molecule `nciforge` runs use the bundled CPU path through `xtbx`.
