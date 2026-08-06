import pytest

from ucc.config import (
    db_to_linear,
    dbm_per_hz_to_w_per_hz,
    mhz_to_hz,
)
from ucc.geometry import (
    compute_uav_sv_distance,
    generate_spatial_realization,
    UAVPosition,
)
from ucc.model import (
    evaluate_uav_scenario,
)


NUMBER_OF_UAVS = 8

REFERENCE_DISTANCE_M = 1.0
REFERENCE_GAIN_LINEAR = db_to_linear(-60.0)
TRANSMIT_POWER_W = 0.8
NOISE_PSD_W_HZ = dbm_per_hz_to_w_per_hz(-174.0)

ALLOCATED_BANDWIDTH_HZ = (
    mhz_to_hz(10.0) / NUMBER_OF_UAVS
)

BACKHAUL_RATE_BPS = 625_000.0

INPUT_PAYLOAD_BITS = 6_000_000.0
OUTPUT_PAYLOAD_BITS = 600_000.0
WORKLOAD_CYCLES = 6_000_000_000.0

UAV_CAPACITY = 200_000_000.0
SV_CAPACITY = 2_000_000_000.0
RCC_CAPACITY = 500_000_000_000.0

BACKHAUL_FIXED_DELAY_S = 0.10


def build_position_directly_above_sv() -> UAVPosition:
    distance = compute_uav_sv_distance(
        uav_x_m=500.0,
        uav_y_m=500.0,
        altitude_m=500.0,
        sv_x_m=500.0,
        sv_y_m=500.0,
    )

    return UAVPosition(
        uav_id=1,
        subregion_id=1,
        row_index=0,
        column_index=0,
        subregion_x_min_m=0.0,
        subregion_x_max_m=1000.0,
        subregion_y_min_m=0.0,
        subregion_y_max_m=1000.0,
        x_m=500.0,
        y_m=500.0,
        altitude_m=500.0,
        sv_x_m=500.0,
        sv_y_m=500.0,
        distance_to_sv_m=distance,
    )


def evaluate(
    scenario: str,
    position: UAVPosition,
):
    return evaluate_uav_scenario(
        seed=1,
        scenario=scenario,
        position=position,
        reference_distance_m=REFERENCE_DISTANCE_M,
        reference_gain_linear=REFERENCE_GAIN_LINEAR,
        transmit_power_w=TRANSMIT_POWER_W,
        noise_psd_w_hz=NOISE_PSD_W_HZ,
        allocated_bandwidth_hz=ALLOCATED_BANDWIDTH_HZ,
        backhaul_rate_bps=BACKHAUL_RATE_BPS,
        input_payload_bits=INPUT_PAYLOAD_BITS,
        output_payload_bits=OUTPUT_PAYLOAD_BITS,
        workload_cycles=WORKLOAD_CYCLES,
        uav_capacity_cycles_s=UAV_CAPACITY,
        sv_capacity_cycles_s=SV_CAPACITY,
        rcc_capacity_cycles_s=RCC_CAPACITY,
        number_of_uavs=NUMBER_OF_UAVS,
        backhaul_fixed_delay_s=BACKHAUL_FIXED_DELAY_S,
    )


@pytest.mark.parametrize(
    ("scenario", "tier", "expected_latency"),
    [
        ("S1", "UAV", 31.111441359902052),
        ("S2", "SV", 25.574413599020502),
        ("S3", "RCC", 10.3104135990205),
    ],
)
def test_integrated_reference_values(
    scenario: str,
    tier: str,
    expected_latency: float,
) -> None:
    position = build_position_directly_above_sv()

    result = evaluate(
        scenario=scenario,
        position=position,
    )

    assert result.seed == 1
    assert result.scenario == scenario
    assert result.execution_tier == tier
    assert result.uav_id == 1

    assert result.distance_to_sv_m == pytest.approx(500.0)

    assert result.channel_gain_linear == pytest.approx(
        4.0e-12,
        rel=1.0e-12,
    )

    assert result.snr_linear == pytest.approx(
        643.0429264664504,
        rel=1.0e-12,
    )

    assert result.access_rate_bps == pytest.approx(
        11_663_766.298994914,
        rel=1.0e-12,
    )

    assert result.end_to_end_latency_s == pytest.approx(
        expected_latency,
        rel=1.0e-12,
    )


def test_integrated_result_preserves_position() -> None:
    realization = generate_spatial_realization(
        number_of_uavs=NUMBER_OF_UAVS,
        area_side_m=1000.0,
        altitude_m=500.0,
        sv_x_m=500.0,
        sv_y_m=500.0,
        seed=17,
    )

    position = realization.uav_positions[0]

    result = evaluate(
        scenario="S3",
        position=position,
    )

    assert result.uav_id == position.uav_id
    assert result.x_m == position.x_m
    assert result.y_m == position.y_m
    assert result.altitude_m == position.altitude_m
    assert result.distance_to_sv_m == position.distance_to_sv_m


def test_same_position_has_same_access_link_in_all_scenarios() -> None:
    position = build_position_directly_above_sv()

    results = [
        evaluate(scenario, position)
        for scenario in ("S1", "S2", "S3")
    ]

    access_rates = {
        result.access_rate_bps
        for result in results
    }

    channel_gains = {
        result.channel_gain_linear
        for result in results
    }

    snrs = {
        result.snr_linear
        for result in results
    }

    assert len(access_rates) == 1
    assert len(channel_gains) == 1
    assert len(snrs) == 1


def test_scenarios_use_expected_payloads() -> None:
    position = build_position_directly_above_sv()

    s1 = evaluate("S1", position)
    s2 = evaluate("S2", position)
    s3 = evaluate("S3", position)

    assert s1.access_payload_bits == OUTPUT_PAYLOAD_BITS
    assert s1.backhaul_payload_bits == OUTPUT_PAYLOAD_BITS

    assert s2.access_payload_bits == INPUT_PAYLOAD_BITS
    assert s2.backhaul_payload_bits == OUTPUT_PAYLOAD_BITS

    assert s3.access_payload_bits == INPUT_PAYLOAD_BITS
    assert s3.backhaul_payload_bits == INPUT_PAYLOAD_BITS


def test_baseline_ranking_at_same_position() -> None:
    position = build_position_directly_above_sv()

    s1 = evaluate("S1", position)
    s2 = evaluate("S2", position)
    s3 = evaluate("S3", position)

    assert (
        s3.end_to_end_latency_s
        < s2.end_to_end_latency_s
        < s1.end_to_end_latency_s
    )
