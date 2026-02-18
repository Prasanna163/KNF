# Comprehensive NCI Speed + Accuracy Report

- Generated: `2026-02-18T10:01:36.553655+00:00`
- Input: `e:\Prasanna\KNF-CORE\molden.input`
- Grid: `[158, 97, 115]` (`1762490` points)
- Torch: `2.10.0+cu128` (CUDA build `12.8`)
- CUDA available: `True`
- CUDA device: `NVIDIA GeForce RTX 3050 6GB Laptop GPU`

## Speed
- Multiwfn avg total: **22.398s**
- Torch CPU estimated total (compute+write): **8.914s**
- Torch GPU estimated total (compute+write): **7.451s**
- Speedup CPU vs Multiwfn: **2.51x**
- Speedup GPU vs Multiwfn: **3.01x**
- Speedup GPU vs CPU: **1.20x**

## Accuracy (vs Multiwfn)
### Torch CPU
- All points: Pearson RDG `0.4038`, Pearson sign(lambda2)rho `0.0717`
- Trimmed 99.9%: Pearson RDG `0.7303`, Pearson sign(lambda2)rho `0.6462`
- Low RDG: Pearson RDG `0.9990`, Pearson sign(lambda2)rho `0.9901`
- Attractive: Pearson RDG `0.9723`, Pearson sign(lambda2)rho `0.0459`

### Torch GPU
- All points: Pearson RDG `0.4038`, Pearson sign(lambda2)rho `0.0717`
- Trimmed 99.9%: Pearson RDG `0.7303`, Pearson sign(lambda2)rho `0.6462`
- Low RDG: Pearson RDG `0.9990`, Pearson sign(lambda2)rho `0.9901`
- Attractive: Pearson RDG `0.9723`, Pearson sign(lambda2)rho `0.0459`
