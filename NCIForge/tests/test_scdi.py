from pathlib import Path

import pytest

from knf_core import scdi


def _write_cosmo(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "$segment_information",
                "# segment atom x y z charge area potential",
                "1 1 -1.0 0.0 0.0 -0.1 1.0 -0.5",
                "2 1  1.0 0.0 0.0  0.1 3.0  0.5",
                "$end",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_scdi_computes_area_weighted_variance(tmp_path):
    metrics = scdi.compute_scdi_metrics(str(_write_cosmo(tmp_path / "xtb.cosmo")))

    assert metrics.valid is True
    assert metrics.reason is None
    assert metrics.variance == pytest.approx(0.0075)
    assert metrics.scdi is None


def test_scdi_applies_optional_fixed_bounds(tmp_path):
    metrics = scdi.compute_scdi_metrics(
        str(_write_cosmo(tmp_path / "xtb.cosmo")),
        var_min=0.0,
        var_max=0.01,
    )

    assert metrics.valid is True
    assert metrics.scdi == pytest.approx(0.25)


def test_scdi_missing_or_empty_input_is_unavailable_not_zero(tmp_path):
    missing = scdi.compute_scdi_metrics(str(tmp_path / "missing.cosmo"))
    assert missing.valid is False
    assert missing.variance is None
    assert missing.reason == "cosmo_file_not_found"

    empty_path = tmp_path / "empty.cosmo"
    empty_path.write_text("$segment_information\n$end\n", encoding="utf-8")
    empty = scdi.compute_scdi_metrics(str(empty_path))
    assert empty.valid is False
    assert empty.variance is None
    assert empty.reason == "no_segment_rows"
