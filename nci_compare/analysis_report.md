# NCI Time-Correlation Analysis Report

- Generated (UTC): `2026-02-18T09:01:43.147066+00:00`
- Input: `e:\Prasanna\KNF-CORE\\molden.input`
- Grid points: `1762490` (`[158, 97, 115]`)
- Device: `cpu` (CUDA available: `False`)

## Timing
- Multiwfn avg: **21.162 s**
- Custom avg (total): **8.503 s**
- Custom avg (compute): `4.456 s`
- Custom avg (write): `4.047 s`
- Speedup (Custom vs Multiwfn): **2.49x**

## Correlation
- All points: Pearson RDG `0.4038`, Pearson sign(lambda2)rho `0.0717`
- Trimmed 99.9%: Pearson RDG `0.7303`, Pearson sign(lambda2)rho `0.6462`
- Low RDG (<= 2.0): Pearson RDG **0.9990**, Pearson sign(lambda2)rho **0.9901**
- Attractive region: Pearson RDG `0.9723`

## Interpretation
- The custom method is significantly faster in this environment.
- Correlation is strongest in the chemically relevant low-RDG region.
- Full-range metrics are affected by outlier/nuclear-region behavior.

## Recommended Files
- `nci_compare/correlation_summary.json`
- `nci_compare/correlation_subsets.json`
- `nci_compare/time_analysis.json`
- `nci_compare/scatter_overlay_low_rdg_le_2.png`
