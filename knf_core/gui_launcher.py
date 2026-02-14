import os
import sys
import subprocess
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent.parent
    app_path = repo_root / "knf_gui" / "app.py"
    if not app_path.exists():
        raise FileNotFoundError(f"Streamlit app not found: {app_path}")

    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    subprocess.run(cmd, cwd=str(repo_root), check=True)


if __name__ == "__main__":
    main()
