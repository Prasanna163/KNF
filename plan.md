# KNF-Core v0.1
## Automated Descriptor Engine for SNCI, SCDI, and 9D KNF

Author: Prasanna Kulkarni  
Language: Python 3.11+  
Design Goal: Automated, resumable, deterministic scientific engine  

---

# 1. OBJECTIVE

Build a fully automated CLI tool named `knf` that:

- Accepts a molecular file (.mol, .xyz, etc.)
- Automatically detects file type
- Detects whether system is:
  - Single molecule
  - Two-fragment complex
- Runs required quantum and analysis steps
- Computes:
  - SNCI (raw integral)
  - SCDI (area-weighted variance)
  - 9D KNF vector
- Writes:
  - output.txt
  - knf.json
- Is resumable
- Is non-interactive
- Is deterministic

No interactive menus.
No prompts.
No manual intervention.

---

# 2. DEPENDENCIES

External Programs (must be available in PATH):

- xTB
- Multiwfn

Python Dependencies:

- numpy
- scipy
- rdkit
- matplotlib
- json
- subprocess
- pathlib

---

# 3. CLI DESIGN

Primary usage:

    knf input_file.mol

Optional flags:

    --charge INT (default 0)
    --spin INT (default 1)
    --force (recompute all steps)
    --clean (delete previous results)
    --debug (verbose logging)

The CLI must be implemented using argparse.

---

# 4. DIRECTORY STRUCTURE

Given input:

    complex.mol

Create:

    complex_knf/
        input/
        results/

Behavior:

- Copy original file into input/
- All generated files go into results/
- Never modify original file

---

# 5. STATE MANAGEMENT (RESUMABLE DESIGN)

Each step must check for required output files.

If file exists and --force is NOT used:
    skip step

If --force:
    recompute step

State is determined purely by file existence.

No databases.
No external state tracking.

---

# 6. PIPELINE ORDER

Strict execution order:

1. Convert input to XYZ
2. Detect fragments
3. xTB geometry optimization
4. xTB single point (molden + wbo + cosmo)
5. Extract electronic descriptors
6. Run Multiwfn NCI grid
7. Compute SNCI
8. Compute NCI statistical descriptors
9. Parse COSMO
10. Compute SCDI variance
11. Compute geometry descriptors
12. Assemble KNF vector
13. Write output.txt
14. Write knf.json

Each step must be implemented as a pure function.

---

# 7. FILE TYPE DETECTION

The program must detect file extension automatically:

Supported:
- .mol
- .xyz
- .molden

Conversion rules:
- .mol → RDKit → xyz
- .xyz → direct use
- .molden → skip single-point if valid

If unsupported file:
    raise explicit error

---

# 8. FRAGMENT DETECTION

Use RDKit connectivity graph.

If:
- 1 fragment → treat as molecule
- 2 fragments → treat as complex
- >2 fragments → raise error

Store fragment atom indices for:
- COM calculation
- Intermolecular WBO detection
- HB angle detection

---

# 9. GEOMETRY DESCRIPTORS

## f1 — Center of Mass Distance

For each fragment:

    COM = sum(m_i * r_i) / sum(m_i)

Compute Euclidean distance between fragment COMs.

Units: Å

---

## f2 — Hydrogen Bond Angle

Detect D–H···A geometry:

- H attached to electronegative atom (O, N, F)
- Acceptors from other fragment

If detected:
    compute angle

If not:
    set f2 = 180.0

---

# 10. XTB INTEGRATION

## Optimization

Command:

    xtb input.xyz --opt --charge X --uhf Y

Output:
- optimized.xyz
- xtb.log

Must detect convergence failure.

---

## Single Point

Command:

    xtb optimized.xyz --wbo --molden --cosmo

Extract:

- molden.input
- wbo file
- cosmo file
- dipole moment
- polarizability

Parse dipole magnitude:

    mu = sqrt(mx^2 + my^2 + mz^2)

Parse isotropic polarizability:

    alpha = (axx + ayy + azz)/3

---

# 11. ELECTRONIC DESCRIPTORS

## f3 — Maximum Intermolecular WBO

From WBO file:

- Only bonds connecting fragment A and B
- Return maximum value

## f4 — Dipole magnitude

## f5 — Isotropic polarizability

---

# 12. MULTIWFN AUTOMATION

Must run Multiwfn non-interactively using scripted input.

Steps:

- Load molden.input
- The procedure for doing the NCIs analysis is as follows:
        20, 1, 3, 1, 2, 0, 0, q 
- Generate RDG grid
- Compute sign(lambda2) * rho
- Output grid data file
- the file will be saved as output.txt
- The file will have 5 columns
- Column 1/2/3: X/Y/Z in Angstrom
- Column 4/5: sign(lambda2)rho and RDG in a.u.


Save:

    output.txt

---

# 13. SNCI COMPUTATION

From grid:
Column 1/2/3: X/Y/Z in Angstrom
Column 4/5: sign(lambda2)rho and RDG in a.u.

Filter:

    lambda2 < 0

Compute:

    SNCI = sum( -sign(lambda2_i) * rho_i * deltaV )

Return raw value.

No normalization.

---

# 14. NCI STATISTICS

From attractive points:

Define:

    xi = sign(lambda2) * rho

Compute:

- f6 = count
- f7 = mean(xi)
- f8 = std(xi)
- f9 = skewness(xi)

---

# 15. SCDI COMPUTATION

From COSMO file:

Extract:
- surface area elements a_i
- surface charge q_i

Compute area-weighted mean:

    mu_A = sum(a_i q_i) / sum(a_i)

Compute variance:

    SCDI = sum(a_i (q_i - mu_A)^2) / sum(a_i)

Return raw variance.

No normalization.

---

# 16. KNF VECTOR

Assemble:

    KNF = [
        f1, f2,
        f3, f4, f5,
        f6, f7, f8, f9
    ]

---

# 17. OUTPUT FILES

## output.txt

Human-readable:

    SNCI_raw:
    SCDI_variance:
    f1:
    f2:
    ...
    f9:

## knf.json

Machine-readable:

{
  "SNCI": float,
  "SCDI_variance": float,
  "KNF_vector": [ ... ],
  "metadata": {
    "charge": int,
    "spin": int,
    "fragments": int
  }
}

---

# 18. ERROR HANDLING

If:
- xTB fails → stop execution
- Multiwfn fails → stop execution
- No attractive NCI points → SNCI = 0
- Single fragment but KNF requested → geometry-only mode

No silent failures.
All errors explicit.

---

# 19. CODE STRUCTURE

Minimal but modular:

    knf_core/
        main.py
        pipeline.py
        geometry.py
        xtb.py
        multiwfn.py
        snci.py
        scdi.py
        knf_vector.py
        utils.py

Core computation functions must:
- Accept file paths
- Return values
- Not print

CLI layer handles printing.

---

# 20. VERSIONING

v0.1 Requirements:

- Fully automated
- Resumable
- Deterministic
- Single-system mode
- No normalization
- No GUI

---

# END OF SPECIFICATION
