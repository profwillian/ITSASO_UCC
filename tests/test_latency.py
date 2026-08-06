import pytest

from ucc.model import (
    ModelValidationError,
    compute_latency_components,
    get_scenario_definition,
)


NUMBER_OF_UAVS = 8

INPUT_PAYLOAD_BITS = 6_000_000.0
OUTPUT_PAYLOAD_BITS = 600_000.0
WORKLOAD_CYCLES = 6_000_000_000.0

UAV_CAPACITY = 200_000_000.0
SV_CAPACITY = 2_000_000_000.0
RCC_CAPACITY = 500_000_000_000.0

ACCESS_RATE_AT_500_M = 11_663_766.298994914
ACCESS_RATE_AT_CORNER = 9_688_154.64668956

BACKHAUL_RATE = 625_000.0
BACKHAUL_FIXED_DELAY = 0.10


def compute_baseline(
    scenario: str,
    access_rate_bps: float = ACCESS_RATE_AT_500_M,
    output_payload_bits: float = OUTPUT_PAYLOAD_BITS,
    backhaul_fixed_delay_s: float = BACKHAUL_FIXED_DELAY,
):
    return compute_latency_components(
        scenario=scenario,
        access_rate_bps=access_rate_bps,
        backhaul_rate_bps=BACKHAUL_RATE,
        input_payload_bits=INPUT_PAYLOAD_BITS,
        output_payload_bits=output_payload_bits,
        workload_cycles=WORKLOAD_CYCLES,
        uav_capacity_cycles_s=UAV_CAPACITY,
        sv_capacity_cycles_s=SV_CAPACITY,
        rcc_capacity_cycles_s=RCC_CAPACITY,
        number_of_uavs=NUMBER_OF_UAVS,
        backhaul_fixed_delay_s=backhaul_fixed_delay_s,
    )


@pytest.mark.parametrize(
    (
        "scenario",
        "execution_tier",
        "access_kind",
        "backhaul_kind",
    ),
    [
        ("S1", "UAV", "output", "output"),
        ("S2", "SV", "input", "output"),
        ("S3", "RCC", "input", "input"),
    ],
)
def test_scenario_definitions(
    scenario: str,
    execution_tier: str,
    access_kind: str,
    backhaul_kind: str,
) -> None:
    definition = get_scenario_definition(scenario)

    assert definition.scenario == scenario
    assert definition.execution_tier == execution_tier
    assert definition.access_payload_kind == access_kind
    assert definition.backhaul_payload_kind == backhaul_kind


def test_s1_baseline_latency_at_500_m() -> None:
    result = compute_baseline("S1")

    assert result.execution_tier == "UAV"
    assert result.access_payload_bits == 600_000.0
    assert result.backhaul_payload_bits == 600_000.0
    assert result.effective_compute_capacity_cycles_s == 200_000_000.0

    assert result.access_time_s == pytest.approx(
        0.05144135990205008,
        rel=1.0e-12,
    )
    assert result.compute_time_s == pytest.approx(30.0)
    assert result.backhaul_transmission_time_s == pytest.approx(0.96)
    assert result.backhaul_fixed_delay_s == pytest.approx(0.10)

    assert result.end_to_end_latency_s == pytest.approx(
        31.111441359902052,
        rel=1.0e-12,
    )


def test_s2_baseline_latency_at_500_m() -> None:
    result = compute_baseline("S2")

    assert result.execution_tier == "SV"
    assert result.access_payload_bits == 6_000_000.0
    assert result.backhaul_payload_bits == 600_000.0
    assert result.effective_compute_capacity_cycles_s == 250_000_000.0

    assert result.access_time_s == pytest.approx(
        0.5144135990205008,
        rel=1.0e-12,
    )
    assert result.compute_time_s == pytest.approx(24.0)
    assert result.backhaul_transmission_time_s == pytest.approx(0.96)

    assert result.end_to_end_latency_s == pytest.approx(
        25.574413599020502,
        rel=1.0e-12,
    )


def test_s3_baseline_latency_at_500_m() -> None:
    result = compute_baseline("S3")

    assert result.execution_tier == "RCC"
    assert result.access_payload_bits == 6_000_000.0
    assert result.backhaul_payload_bits == 6_000_000.0
    assert (
        result.effective_compute_capacity_cycles_s
        == 62_500_000_000.0
    )

    assert result.access_time_s == pytest.approx(
        0.5144135990205008,
        rel=1.0e-12,
    )
    assert result.compute_time_s == pytest.approx(0.096)
    assert result.backhaul_transmission_time_s == pytest.approx(9.6)

    assert result.end_to_end_latency_s == pytest.approx(
        10.3104135990205,
        rel=1.0e-12,
    )


@pytest.mark.parametrize(
    ("scenario", "expected_latency"),
    [
        ("S1", 31.12193129877474),
        ("S2", 25.679312987747384),
        ("S3", 10.415312987747383),
    ],
)
def test_corner_latency_reference_values(
    scenario: str,
    expected_latency: float,
) -> None:
    result = compute_baseline(
        scenario=scenario,
        access_rate_bps=ACCESS_RATE_AT_CORNER,
    )

    assert result.end_to_end_latency_s == pytest.approx(
        expected_latency,
        rel=1.0e-12,
    )


def test_s3_does_not_depend_on_output_payload() -> None:
    low_output = compute_baseline(
        scenario="S3",
        output_payload_bits=60_000.0,
    )

    high_output = compute_baseline(
        scenario="S3",
        output_payload_bits=3_000_000.0,
    )

    assert (
        low_output.end_to_end_latency_s
        == high_output.end_to_end_latency_s
    )


@pytest.mark.parametrize(
    "scenario",
    ["S1", "S2", "S3"],
)
def test_fixed_backhaul_delay_is_added_equally(
    scenario: str,
) -> None:
    low_delay = compute_baseline(
        scenario=scenario,
        backhaul_fixed_delay_s=0.10,
    )

    high_delay = compute_baseline(
        scenario=scenario,
        backhaul_fixed_delay_s=0.25,
    )

    difference = (
        high_delay.end_to_end_latency_s
        - low_delay.end_to_end_latency_s
    )

    assert difference == pytest.approx(
        0.15,
        abs=1.0e-12,
    )


def test_farther_uav_has_higher_latency_in_all_scenarios() -> None:
    for scenario in ("S1", "S2", "S3"):
        near = compute_baseline(
            scenario=scenario,
            access_rate_bps=ACCESS_RATE_AT_500_M,
        )

        far = compute_baseline(
            scenario=scenario,
            access_rate_bps=ACCESS_RATE_AT_CORNER,
        )

        assert far.end_to_end_latency_s > near.end_to_end_latency_s


def test_invalid_scenario_is_rejected() -> None:
    with pytest.raises(ModelValidationError):
        compute_baseline("S4")


def test_output_payload_cannot_exceed_input_payload() -> None:
    with pytest.raises(ModelValidationError):
        compute_baseline(
            scenario="S1",
            output_payload_bits=7_000_000.0,
        )
