# Post-Optimization Speed + Accuracy Report

Input: `E:\Prasanna\KNF-CORE\molden.input`

## Scope
- NCI stage only (xTB not included in timing).
- Torch CPU/GPU timings are measured after one warm-up run.

## Speed
- Multiwfn avg (3 runs): **22.429s**
- Torch CPU avg (5 measured): **0.804s**
- Torch GPU avg (5 measured): **0.301s**
- CPU speedup vs Multiwfn: **27.89x**
- GPU speedup vs Multiwfn: **74.52x**
- GPU speedup vs CPU: **2.67x**

## Accuracy (vs Multiwfn grid)
- CPU low-RDG Pearson (sl2rho/rdg): **0.990095 / 0.998957**
- GPU low-RDG Pearson (sl2rho/rdg): **0.990095 / 0.998957**

## Torch Timing Breakdown (last measured run)
- CPU: `{'parse_molden': 0.008698699995875359, 'build_grid': 5.080003757029772e-05, 'compute_fields': 0.7955341999186203, 'export_grid': 0.004819300025701523}`
- GPU: `{'parse_molden': 0.009525700006633997, 'build_grid': 5.779997445642948e-05, 'compute_fields': 0.2833902999991551, 'export_grid': 0.0038501000963151455}`
