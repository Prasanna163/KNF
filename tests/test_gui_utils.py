from knf_gui.utils.validators import validate_parameters
from knf_gui.utils.formatters import format_seconds


def test_validate_parameters_ok():
    ok, err = validate_parameters(0, 1)
    assert ok is True
    assert err is None


def test_validate_parameters_bad_spin():
    ok, err = validate_parameters(0, 0)
    assert ok is False
    assert ">= 1" in err


def test_format_seconds():
    assert format_seconds(5.0).endswith("s")
    assert "m" in format_seconds(65.0)

