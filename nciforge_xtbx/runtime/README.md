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
package limits. For explicit GPU execution, configure a full `xtb-win-release`
folder that contains the CUDA DLL payload:

```powershell
xtbx --setup-gpu
```

or:

```powershell
xtbx --setup-gpu "E:\Prasanna\xTB\xtb\xtb-win-release"
```

For one process without saving config, set `NCIFORGE_XTBX_RUNTIME` or pass
`--gpu-runtime`.

Normal small-molecule `nciforge` runs use the bundled CPU path through `xtbx`.
