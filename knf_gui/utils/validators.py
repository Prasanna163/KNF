import os

SUPPORTED_EXTENSIONS = {".xyz", ".sdf", ".mol", ".pdb", ".mol2"}
MAX_FILE_SIZE_MB = 50


def validate_molecule_file(filepath: str):
    if not filepath:
        return False, "No file path provided.", []
    if not os.path.exists(filepath):
        return False, f"File not found: {filepath}", []

    ext = os.path.splitext(filepath)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return False, f"Unsupported extension: {ext}", []

    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"File is too large ({size_mb:.1f} MB). Max is {MAX_FILE_SIZE_MB} MB.", []

    warnings = []
    if size_mb > 20:
        warnings.append("Large file detected. Calculations may take longer.")

    return True, None, warnings


def validate_parameters(charge: int, spin: int):
    if spin < 1:
        return False, "Spin multiplicity must be >= 1."
    if charge < -5 or charge > 5:
        return False, "Charge must be between -5 and +5."
    return True, None

