from pathlib import Path

from typer.testing import CliRunner

from knf_core import api
from knf_core.cli.app import app
from knf_core.engine.constants import CLI_VERSION


ROOT = Path(__file__).resolve().parents[1]


def test_public_version_is_consistent():
    assert CLI_VERSION == "1.0.9"
    assert api.API_VERSION == CLI_VERSION
    assert "version='1.0.9'" in (ROOT / "setup.py").read_text(encoding="utf-8")
    assert "release-1.0.9" in (ROOT / "README.md").read_text(encoding="utf-8")


def test_cli_version_flag_does_not_require_input():
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "NCIForge 1.0.9"


def test_public_readme_uses_repository_owned_brand_asset():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "assets/branding/nciforge-horizontal.png" in readme
    assert "github.com/user-attachments" not in readme
    assert (ROOT / "assets" / "branding" / "nciforge-horizontal.png").is_file()
