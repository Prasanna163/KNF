# KNF-CORE: Automated Descriptor Engine

**KNF-CORE** is a specialized computational chemistry pipeline designed to automate the extraction of key molecular descriptors for KNF (Kinetic and Non-covalent Feature) analysis. It integrates powerful tools like **xTB** (Extended Tight-Binding) and **Multiwfn** (Multifunctional Wavefunction Analyzer) to generate a unique 9-dimensional vector representation of molecular interactions.

## Features

- **Automated Workflow**: From a single input structure (XYZ, SDF, MOL, etc.) to a final descriptor vector with minimal user intervention.
- **Robust Geometry Handling**: Automatically detects and converts input formats, perceives molecular fragments, and handles single molecules, two-fragment complexes, and multi-fragment systems.
- **xTB Integration**: Performs semi-empirical quantum mechanical optimization (Geometry Optimization) and single-point energy calculations to derive electronic properties.
- **Multiwfn Analysis**: Automates Non-Covalent Interaction (NCI) analysis to capture weak interactions critical for binding affinity and stability.
- **COSMO Solvation Model**: Computes Sigma-Profile descriptors (SCDI) to account for solvation effects.
- **Docker Support**: Fully containerized environment for consistent execution across different platforms.

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

A Dockerfile is provided for a complete, pre-configured environment.

1. **Build the image:**
   ```bash
   docker build -t knf-core .
   ```

2. **Run the container:**
   ```bash
   docker run --rm -v $(pwd):/data knf-core /data/input.sdf
   ```

## Usage

### Command Line Interface (CLI)

The package provides a `knf` command for easy execution.

**Interactive Guided Mode:**
```bash
knf
```
This launches a step-by-step interface with:
- Input file loading and fragment detection
- Mode selection (`Automated Pipeline` or `Interactive xTB Explorer`)
- Selectable xTB calculations with command preview/confirmation
- Post-optimization next-step menu (including Multiwfn NCI handoff)

### Streamlit GUI

Run the graphical interface:

```bash
knf-gui
```

Or directly with Streamlit:

```bash
streamlit run streamlit_app.py
```

GUI capabilities include:
- Full automated KNF pipeline execution
- xTB explorer with selectable operations and command preview
- Optional solvent flag injection (`--alpb`)
- Post-optimization NCI handoff to Multiwfn
- Result file browser and previews

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
