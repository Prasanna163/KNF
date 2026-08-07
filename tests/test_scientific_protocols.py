from pathlib import Path

import pytest

from knf_core import knf_vector, scdi
from knf_core.engine.kuid_ops import _require_single_f3_protocol
from knf_core.engine.types import RunOptions


def _entry(protocol: str) -> dict:
    return {
        "knf": {
            "metadata": {"f3_definition": protocol},
            "KNF_vector": [1.0] * 9,
        }
    }


def test_production_defaults_are_strict_geometry_and_parsed_xtb_wbo():
    options = RunOptions()
    assert options.sp is False
    assert options.seed_contact is False
    assert options.wbo_mode == "xtb"


def test_missing_scdi_remains_null(tmp_path):
    metrics = scdi.compute_scdi_metrics(str(tmp_path / "missing.cosmo"))
    assert metrics.variance is None
    assert metrics.scdi is None

    result = knf_vector.KNFResult(
        SNCI=0.1,
        SCDI=None,
        SCDI_variance=None,
        KNF_vector=[1.0, 2.0, 0.3, 1.2, 3.4, 5.0, 0.01, 0.02, 0.03],
        metadata={},
    )
    output = tmp_path / "output.txt"
    knf_vector.write_output_txt(str(output), result)
    assert "SCDI_variance:  n/a" in output.read_text(encoding="utf-8")


def test_kuid_calibration_rejects_mixed_f3_definitions():
    with pytest.raises(ValueError, match="mix incompatible f3 definitions"):
        _require_single_f3_protocol(
            [
                _entry("parsed_xtb_interfragment_wiberg_bond_order"),
                _entry("identity_overlap_interfragment_density_coupling"),
            ]
        )


def test_kuid_calibration_accepts_one_f3_definition():
    protocol = "parsed_xtb_interfragment_wiberg_bond_order"
    assert _require_single_f3_protocol([_entry(protocol), _entry(protocol)]) == protocol
