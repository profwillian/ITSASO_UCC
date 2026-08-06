import math

import pytest

from ucc.config import (
    ConfigurationError,
    db_to_linear,
    dbm_per_hz_to_w_per_hz,
    ghz_to_cycles_per_second,
    mbit_to_bits,
    mbps_to_bps,
    mhz_to_hz,
)


def test_mbit_to_bits() -> None:
    assert mbit_to_bits(2.0) == 2_000_000.0


def test_mbps_to_bps() -> None:
    assert mbps_to_bps(5.0) == 5_000_000.0


def test_mhz_to_hz() -> None:
    assert mhz_to_hz(10.0) == 10_000_000.0


@pytest.mark.parametrize(
    ("value_ghz", "expected"),
    [
        (0.2, 200_000_000.0),
        (2.0, 2_000_000_000.0),
        (500.0, 500_000_000_000.0),
    ],
)
def test_ghz_to_cycles_per_second(
    value_ghz: float,
    expected: float,
) -> None:
    assert ghz_to_cycles_per_second(value_ghz) == expected


def test_db_to_linear() -> None:
    assert db_to_linear(-60.0) == pytest.approx(
        1.0e-6,
        rel=1.0e-12,
    )


def test_dbm_per_hz_to_w_per_hz() -> None:
    expected = 3.981071705534986e-21

    assert dbm_per_hz_to_w_per_hz(-174.0) == pytest.approx(
        expected,
        rel=1.0e-12,
    )


@pytest.mark.parametrize(
    "invalid_value",
    [-1.0, -math.inf, math.inf, math.nan],
)
def test_non_negative_conversions_reject_invalid_values(
    invalid_value: float,
) -> None:
    with pytest.raises(ConfigurationError):
        mbit_to_bits(invalid_value)


def test_numeric_conversions_reject_boolean() -> None:
    with pytest.raises(ConfigurationError):
        mhz_to_hz(True)
