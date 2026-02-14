# KNF-CORE: Automated Descriptor Engine

**KNF-CORE** is a specialized computational chemistry pipeline designed to automate the extraction of key molecular descriptors for KNF (Kinetic and Non-covalent Feature) analysis. It integrates powerful tools like **xTB** (Extended Tight-Binding) and **Multiwfn** (Multifunctional Wavefunction Analyzer) to generate a unique 9-dimensional vector representation of molecular interactions.

## Features

- **Automated Workflow**: From a single input structure (XYZ, SDF, MOL, etc.) to a final descriptor vector with minimal user intervention.
- **Robust Geometry Handling**: Automatically detects and converts input formats, perceives molecular fragments, and handles single molecules, two-fragment complexes, and multi-fragment systems.
- **xTB Integration**: Performs semi-empirical quantum mechanical optimization (Geometry Optimization) and single-point energy calculations to derive electronic properties.
- **Multiwfn Analysis**: Automates Non-Covalent Interaction (NCI) analysis to capture weak interactions critical for binding affinity and stability.
- **COSMO Solvation Model**: Computes Sigma-Profile descriptors (SCDI) to account for solvation effects.
- **Docker Support**: Fully containerized environment for consistent execution across different platforms.

## Branch Notes (`Multiple-Molecules`)

This branch includes explicit multi-fragment behavior:

- `1` fragment: `f1 = 0.0`, `f2 = 180.0`
- `2` fragments: `f1` = COM distance between fragments, `f2` = detected H-bond angle
- `>2` fragments:
  - `f1` = average COM distance across all unique fragment pairs
  - `f2` = fixed to `180.0`

This branch also includes an updated Docker workflow (`Dockerfile`, `docker-compose.yml`, and container entrypoint script).

## Installation

### Install from PyPI

```bash
pip install KNF
```

### Prerequisites

- **Python 3.8+**
- **xTB**: Must be installed and accessible in your system PATH (`xtb` command).
- **Multiwfn**: Must be installed and accessible (`Multiwfn` command), or configured via `settings.ini`.
- **OpenBabel**: Required for file format conversions (`obabel` command).

### Install from Source

 Clone the repository and install using pip:

```bash
git clone https://github.com/yourusername/KNF-CORE.git
cd KNF-CORE
pip install .
```

### Docker Usage

A Dockerfile and `docker-compose.yml` are provided for a complete, pre-configured environment
with Python, xTB, OpenBabel, and Multiwfn.

1. **Build the image:**
   ```bash
   docker build -t knf-core .
   ```

2. **Run a KNF calculation:**
   ```bash
   docker run --rm -v $(pwd):/work -w /work knf-core input.sdf --charge 0 --force
   ```

3. **Run with Docker Compose:**
   ```bash
   docker compose up --build
   ```
   Edit `docker-compose.yml` command to target your input and options.

4. **Open a shell in the container (debugging):**
   ```bash
   docker run --rm -it -v $(pwd):/work -w /work knf-core bash
   ```

5. **Verify toolchain inside container:**
   ```bash
   docker run --rm knf-core --help
   docker run --rm -it knf-core bash -lc "xtb --version && obabel -V && Multiwfn < /dev/null || true"
   ```

## Usage

### Command Line Interface (CLI)

The package provides a `knf` command for easy execution.

**Basic Usage:**
```bash
knf input_molecule.sdf
```

**Options:**
- `--charge <int>`: Net charge of the system (default: 0).
- `--spin <int>`: Spin multiplicity (default: 1).
- `--force`: Force recalculation of existing steps.
- `--clean`: Clean up previous run directories before starting.
- `--debug`: Enable verbose debug logging.

**Example with Test File:**
The repository includes a test file `example.mol` (diethyl sulfate) for verification.

```bash
knf example.mol --charge 0 --force
```

**General Example:**
```bash
knf drug_molecule.sdf --charge 1 --force
```

### Python API

You can also use KNF-CORE within your own Python scripts:

```python
from knf_core.pipeline import KNFPipeline

# Initialize the pipeline
pipeline = KNFPipeline(
    input_file='test.sdf',
    charge=0,
    spin=1
)

# Run the analysis
pipeline.run()
```

## Output

The pipeline generates a `results` folder containing:
- `knf.json`: The final 9D KNF vector and metadata.
- `output.txt`: Human-readable summary of the results.
- `xtbopt.xyz`: Optimized geometry.
- Intermediate files from xTB and Multiwfn.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
