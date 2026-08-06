"""Configuration utilities for the UCC maritime simulation."""

from __future__ import annotations

import math
from numbers import Real


class ConfigurationError(ValueError):
    """Raised when an experimental parameter is invalid."""


def _as_finite_float(value: Real, field_name: str) -> float:
    """Convert a numeric value to float and ensure that it is finite."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigurationError(
            f"{field_name} must be a real number. Received: {value!r}"
        )

    converted = float(value)

    if not math.isfinite(converted):
        raise ConfigurationError(
            f"{field_name} must be finite. Received: {value!r}"
        )

    return converted


def _as_non_negative_float(value: Real, field_name: str) -> float:
    """Convert a numeric value and reject negative values."""
    converted = _as_finite_float(value, field_name)

    if converted < 0.0:
        raise ConfigurationError(
            f"{field_name} must not be negative. Received: {converted}"
        )

    return converted


def mbit_to_bits(value_mbit: Real) -> float:
    """Convert decimal megabits to bits."""
    value = _as_non_negative_float(value_mbit, "value_mbit")
    return value * 1_000_000.0


def mbps_to_bps(value_mbps: Real) -> float:
    """Convert decimal megabits per second to bits per second."""
    value = _as_non_negative_float(value_mbps, "value_mbps")
    return value * 1_000_000.0


def mhz_to_hz(value_mhz: Real) -> float:
    """Convert megahertz to hertz."""
    value = _as_non_negative_float(value_mhz, "value_mhz")
    return value * 1_000_000.0


def ghz_to_cycles_per_second(value_ghz: Real) -> float:
    """Convert GHz to CPU cycles per second."""
    value = _as_non_negative_float(value_ghz, "value_ghz")
    return value * 1_000_000_000.0


def db_to_linear(value_db: Real) -> float:
    """Convert a power ratio expressed in dB to linear scale."""
    value = _as_finite_float(value_db, "value_db")
    converted = math.pow(10.0, value / 10.0)

    if not math.isfinite(converted) or converted <= 0.0:
        raise ConfigurationError(
            f"dB conversion produced an invalid result: {converted}"
        )

    return converted


def dbm_per_hz_to_w_per_hz(value_dbm_hz: Real) -> float:
    """Convert dBm/Hz to W/Hz."""
    value = _as_finite_float(value_dbm_hz, "value_dbm_hz")
    converted = math.pow(10.0, (value - 30.0) / 10.0)

    if not math.isfinite(converted) or converted <= 0.0:
        raise ConfigurationError(
            f"dBm/Hz conversion produced an invalid result: {converted}"
        )

    return converted
def _as_positive_int(value: int, field_name: str) -> int:
    """Validate and return a strictly positive integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(
            f"{field_name} must be an integer. Received: {value!r}"
        )

    if value <= 0:
        raise ConfigurationError(
            f"{field_name} must be greater than zero. Received: {value}"
        )

    return value


def resolve_factorized_grid(number_of_uavs: int) -> tuple[int, int]:
    """
    Resolve the most balanced integer grid whose product equals N.

    Examples:
        4  -> (2, 2)
        8  -> (2, 4)
        12 -> (3, 4)
        16 -> (4, 4)
    """
    number_of_uavs = _as_positive_int(
        number_of_uavs,
        "number_of_uavs",
    )

    best_rows = 1
    best_columns = number_of_uavs

    for rows in range(1, math.isqrt(number_of_uavs) + 1):
        if number_of_uavs % rows == 0:
            best_rows = rows
            best_columns = number_of_uavs // rows

    return best_rows, best_columns
