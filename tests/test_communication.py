import math

import pytest

from ucc.config import (
    db_to_linear,
    dbm_per_hz_to_w_per_hz,
    mhz_to_hz,
)
from ucc.model import (
    ModelValidationError,
    build_access_link_state,
    compute_access_rate_bps,
    compute_channel_gain,
    compute_snr_linear,
)


REFERENCE_DISTANCE_M = 1.0
REFERENCE_GAIN_LINEAR = db_to_linear(-60.0)
TRANSMIT_POWER_W = 0.8
NOISE_PSD_W_HZ = dbm_per_hz_to_w_per_hz(-174.0)
ALLOCATED_BANDWIDTH_HZ = mhz_to_hz(10.0) / 8.0


def test_channel_gain_at_500_m() -> None:
    gain = compute_channel_gain(
        distance_m=500.0,
        reference_distance_m=REFERENCE_DISTANCE_M,
        reference_gain_linear=REFERENCE_GAIN_LINEAR,
    )

    assert gain == pytest.approx(
        4.0e-12,
        rel=1.0e-12,
    )


def test_snr_at_500_m() -> None:
    gain = compute_channel_gain(
        distance_m=500.0,
        reference_distance_m=REFERENCE_DISTANCE_M,
        reference_gain_linear=REFERENCE_GAIN_LINEAR,
    )

    snr = compute_snr_linear(
        transmit_power_w=TRANSMIT_POWER_W,
        channel_gain_linear=gain,
        noise_psd_w_hz=NOISE_PSD_W_HZ,
        allocated_bandwidth_hz=ALLOCATED_BANDWIDTH_HZ,
    )

    assert snr == pytest.approx(
        643.0429264664504,
        rel=1.0e-12,
    )


def test_access_rate_at_500_m() -> None:
    state = build_access_link_state(
        uav_id=1,
        distance_m=500.0,
        reference_distance_m=REFERENCE_DISTANCE_M,
        reference_gain_linear=REFERENCE_GAIN_LINEAR,
        transmit_power_w=TRANSMIT_POWER_W,
        noise_psd_w_hz=NOISE_PSD_W_HZ,
        allocated_bandwidth_hz=ALLOCATED_BANDWIDTH_HZ,
    )

    assert state.channel_gain_linear == pytest.approx(
        4.0e-12,
        rel=1.0e-12,
    )

    assert state.snr_linear == pytest.approx(
        643.0429264664504,
        rel=1.0e-12,
    )

    assert state.rate_bps == pytest.approx(
        11_663_766.298994914,
        rel=1.0e-12,
    )


def test_access_link_at_search_area_corner() -> None:
    distance = math.sqrt(
        500.0**2
        + 500.0**2
        + 500.0**2
    )

    state = build_access_link_state(
        uav_id=1,
        distance_m=distance,
        reference_distance_m=REFERENCE_DISTANCE_M,
        reference_gain_linear=REFERENCE_GAIN_LINEAR,
        transmit_power_w=TRANSMIT_POWER_W,
        noise_psd_w_hz=NOISE_PSD_W_HZ,
        allocated_bandwidth_hz=ALLOCATED_BANDWIDTH_HZ,
    )

    assert state.channel_gain_linear == pytest.approx(
        1.3333333333333334e-12,
        rel=1.0e-12,
    )

    assert state.snr_linear == pytest.approx(
        214.34764215548347,
        rel=1.0e-12,
    )

    assert state.rate_bps == pytest.approx(
        9_688_154.64668956,
        rel=1.0e-12,
    )


def test_access_rate_decreases_with_distance() -> None:
    near_state = build_access_link_state(
        uav_id=1,
        distance_m=500.0,
        reference_distance_m=REFERENCE_DISTANCE_M,
        reference_gain_linear=REFERENCE_GAIN_LINEAR,
        transmit_power_w=TRANSMIT_POWER_W,
        noise_psd_w_hz=NOISE_PSD_W_HZ,
        allocated_bandwidth_hz=ALLOCATED_BANDWIDTH_HZ,
    )

    far_state = build_access_link_state(
        uav_id=1,
        distance_m=800.0,
        reference_distance_m=REFERENCE_DISTANCE_M,
        reference_gain_linear=REFERENCE_GAIN_LINEAR,
        transmit_power_w=TRANSMIT_POWER_W,
        noise_psd_w_hz=NOISE_PSD_W_HZ,
        allocated_bandwidth_hz=ALLOCATED_BANDWIDTH_HZ,
    )

    assert far_state.channel_gain_linear < near_state.channel_gain_linear
    assert far_state.snr_linear < near_state.snr_linear
    assert far_state.rate_bps < near_state.rate_bps


def test_access_rate_increases_with_transmit_power() -> None:
    gain = compute_channel_gain(
        distance_m=700.0,
        reference_distance_m=REFERENCE_DISTANCE_M,
        reference_gain_linear=REFERENCE_GAIN_LINEAR,
    )

    low_power_snr = compute_snr_linear(
        transmit_power_w=0.2,
        channel_gain_linear=gain,
        noise_psd_w_hz=NOISE_PSD_W_HZ,
        allocated_bandwidth_hz=ALLOCATED_BANDWIDTH_HZ,
    )

    high_power_snr = compute_snr_linear(
        transmit_power_w=1.0,
        channel_gain_linear=gain,
        noise_psd_w_hz=NOISE_PSD_W_HZ,
        allocated_bandwidth_hz=ALLOCATED_BANDWIDTH_HZ,
    )

    low_power_rate = compute_access_rate_bps(
        allocated_bandwidth_hz=ALLOCATED_BANDWIDTH_HZ,
        snr_linear=low_power_snr,
    )

    high_power_rate = compute_access_rate_bps(
        allocated_bandwidth_hz=ALLOCATED_BANDWIDTH_HZ,
        snr_linear=high_power_snr,
    )

    assert high_power_snr > low_power_snr
    assert high_power_rate > low_power_rate


@pytest.mark.parametrize(
    ("distance_m", "reference_distance_m", "reference_gain_linear"),
    [
        (0.0, 1.0, 1.0e-6),
        (-1.0, 1.0, 1.0e-6),
        (500.0, 0.0, 1.0e-6),
        (500.0, 1.0, 0.0),
    ],
)
def test_channel_gain_rejects_invalid_inputs(
    distance_m: float,
    reference_distance_m: float,
    reference_gain_linear: float,
) -> None:
    with pytest.raises(ModelValidationError):
        compute_channel_gain(
            distance_m=distance_m,
            reference_distance_m=reference_distance_m,
            reference_gain_linear=reference_gain_linear,
        )


def test_access_rate_rejects_zero_bandwidth() -> None:
    with pytest.raises(ModelValidationError):
        compute_access_rate_bps(
            allocated_bandwidth_hz=0.0,
            snr_linear=10.0,
        )


def test_access_rate_rejects_zero_snr() -> None:
    with pytest.raises(ModelValidationError):
        compute_access_rate_bps(
            allocated_bandwidth_hz=1_250_000.0,
            snr_linear=0.0,
        )
