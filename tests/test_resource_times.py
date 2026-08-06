import pytest

from ucc.model import (
    ModelValidationError,
    compute_backhaul_rate_per_flow,
    compute_bandwidth_per_uav,
    compute_processing_time,
    compute_remote_capacity_per_job,
    compute_transmission_time,
)


NUMBER_OF_UAVS = 8

INPUT_PAYLOAD_BITS = 6_000_000.0
OUTPUT_PAYLOAD_BITS = 600_000.0
WORKLOAD_CYCLES = 6_000_000_000.0

ACCESS_RATE_AT_500_M_BPS = 11_663_766.298994914


def test_baseline_bandwidth_per_uav() -> None:
    result = compute_bandwidth_per_uav(
        total_bandwidth_hz=10_000_000.0,
        number_of_uavs=NUMBER_OF_UAVS,
    )

    assert result == 1_250_000.0


def test_baseline_backhaul_rate_per_flow() -> None:
    result = compute_backhaul_rate_per_flow(
        aggregate_capacity_bps=5_000_000.0,
        number_of_uavs=NUMBER_OF_UAVS,
    )

    assert result == 625_000.0


def test_baseline_sv_capacity_per_job() -> None:
    result = compute_remote_capacity_per_job(
        total_capacity_cycles_s=2_000_000_000.0,
        number_of_uavs=NUMBER_OF_UAVS,
    )

    assert result == 250_000_000.0


def test_baseline_rcc_capacity_per_job() -> None:
    result = compute_remote_capacity_per_job(
        total_capacity_cycles_s=500_000_000_000.0,
        number_of_uavs=NUMBER_OF_UAVS,
    )

    assert result == 62_500_000_000.0


def test_output_access_transmission_time_at_500_m() -> None:
    result = compute_transmission_time(
        payload_bits=OUTPUT_PAYLOAD_BITS,
        rate_bps=ACCESS_RATE_AT_500_M_BPS,
    )

    assert result == pytest.approx(
        0.05144135990205008,
        rel=1.0e-12,
    )


def test_input_access_transmission_time_at_500_m() -> None:
    result = compute_transmission_time(
        payload_bits=INPUT_PAYLOAD_BITS,
        rate_bps=ACCESS_RATE_AT_500_M_BPS,
    )

    assert result == pytest.approx(
        0.5144135990205008,
        rel=1.0e-12,
    )


def test_output_backhaul_transmission_time() -> None:
    result = compute_transmission_time(
        payload_bits=OUTPUT_PAYLOAD_BITS,
        rate_bps=625_000.0,
    )

    assert result == pytest.approx(0.96)


def test_input_backhaul_transmission_time() -> None:
    result = compute_transmission_time(
        payload_bits=INPUT_PAYLOAD_BITS,
        rate_bps=625_000.0,
    )

    assert result == pytest.approx(9.6)


def test_uav_processing_time() -> None:
    result = compute_processing_time(
        workload_cycles=WORKLOAD_CYCLES,
        effective_capacity_cycles_s=200_000_000.0,
    )

    assert result == pytest.approx(30.0)


def test_sv_processing_time() -> None:
    result = compute_processing_time(
        workload_cycles=WORKLOAD_CYCLES,
        effective_capacity_cycles_s=250_000_000.0,
    )

    assert result == pytest.approx(24.0)


def test_rcc_processing_time() -> None:
    result = compute_processing_time(
        workload_cycles=WORKLOAD_CYCLES,
        effective_capacity_cycles_s=62_500_000_000.0,
    )

    assert result == pytest.approx(0.096)


def test_zero_payload_has_zero_transmission_time() -> None:
    result = compute_transmission_time(
        payload_bits=0.0,
        rate_bps=1_000_000.0,
    )

    assert result == 0.0


def test_higher_rate_reduces_transmission_time() -> None:
    slow = compute_transmission_time(
        payload_bits=INPUT_PAYLOAD_BITS,
        rate_bps=1_000_000.0,
    )

    fast = compute_transmission_time(
        payload_bits=INPUT_PAYLOAD_BITS,
        rate_bps=10_000_000.0,
    )

    assert fast < slow


def test_higher_compute_capacity_reduces_processing_time() -> None:
    slow = compute_processing_time(
        workload_cycles=WORKLOAD_CYCLES,
        effective_capacity_cycles_s=200_000_000.0,
    )

    fast = compute_processing_time(
        workload_cycles=WORKLOAD_CYCLES,
        effective_capacity_cycles_s=2_000_000_000.0,
    )

    assert fast < slow


@pytest.mark.parametrize(
    ("total_resource", "number_of_uavs"),
    [
        (0.0, 8),
        (-1.0, 8),
        (10_000_000.0, 0),
        (10_000_000.0, -1),
    ],
)
def test_equal_bandwidth_rejects_invalid_inputs(
    total_resource: float,
    number_of_uavs: int,
) -> None:
    with pytest.raises(ModelValidationError):
        compute_bandwidth_per_uav(
            total_bandwidth_hz=total_resource,
            number_of_uavs=number_of_uavs,
        )


@pytest.mark.parametrize(
    ("payload_bits", "rate_bps"),
    [
        (-1.0, 1_000_000.0),
        (1_000_000.0, 0.0),
        (1_000_000.0, -1.0),
    ],
)
def test_transmission_time_rejects_invalid_inputs(
    payload_bits: float,
    rate_bps: float,
) -> None:
    with pytest.raises(ModelValidationError):
        compute_transmission_time(
            payload_bits=payload_bits,
            rate_bps=rate_bps,
        )


@pytest.mark.parametrize(
    ("workload_cycles", "capacity_cycles_s"),
    [
        (0.0, 1_000_000.0),
        (-1.0, 1_000_000.0),
        (1_000_000.0, 0.0),
        (1_000_000.0, -1.0),
    ],
)
def test_processing_time_rejects_invalid_inputs(
    workload_cycles: float,
    capacity_cycles_s: float,
) -> None:
    with pytest.raises(ModelValidationError):
        compute_processing_time(
            workload_cycles=workload_cycles,
            effective_capacity_cycles_s=capacity_cycles_s,
        )
