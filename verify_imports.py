import sys
import os
import subprocess

def test_imports():
    print("Testing imports...")
    try:
        import knf_core
        from knf_core import utils, geometry, xtb, multiwfn, snci, scdi, knf_vector, pipeline, main
        print("Imports successful.")
    except ImportError as e:
        print(f"Import failed: {e}")
        sys.exit(1)

def test_cli_help():
    print("Testing CLI help...")
    cmd = [sys.executable, '-m', 'knf_core.main', '--help']
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("CLI help command ran successfully.")
        print(result.stdout)
    else:
        print("CLI help command failed.")
        print(result.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # Ensure current directory is in sys.path
    sys.path.insert(0, os.getcwd())
    test_imports()
    test_cli_help()
