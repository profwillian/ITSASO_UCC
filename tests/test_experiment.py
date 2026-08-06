import pytest

from ucc.config import (
    db_to_linear,
    dbm_per_hz_to_w_per_hz,
    ghz_to_cycles_per_second,
    mbit_to_bits,
    mbps_to_bps,
    mhz_to_hz,
)
from ucc.experiment import (
    ExperimentParameters,
    evaluate_seed,
)


def build_baseline_parameters() -> ExperimentParameters:
    number_of_uavs = 8

    input_payload_bits = (
        3 * mbit_to_bits(2.0)
    )

    output_payload_bits = (
        0.10 * input_payload_bits
    )

    workload_cycles = (
        input_payload_bits * 1000.0
    )

    return ExperimentParameters(
        number_of_uavs=number_of_uavs,
        area_side_m=1000.0,
        altitude_m=500.0,
        sv_x_m=500.0,
        sv_y_m=500.0,
        reference_distance_m=1.0,
        reference_gain_linear=db_to_linear(-60.0),
        transmit_power_w=0.8,
        noise_psd_w_hz=dbm_per_hz_to_w_per_hz(-174.0),
        total_bandwidth_hz=mhz_to_hz(10.0),
        aggregate_backhaul_capacity_bps=mbps_to_bps(5.0),
        backhaul_fixed_delay_s=0.10,
        input_payload_bits=input_payload_bits,
        output_payload_bits=output_payload_bits,
        workload_cycles=workload_cycles,
        uav_capacity_cycles_s=(
            ghz_to_cycles_per_second(0.2)
        ),
        sv_capacity_cycles_s=(
            ghz_to_cycles_per_second(2.0)
        ),
        rcc_capacity_cycles_s=(
            ghz_to_cycles_per_second(500.0)
        ),
        scenarios=("S1", "S2", "S3"),
    )


def test_one_seed_produces_expected_result_counts() -> None:
    evaluation = evaluate_seed(
        seed=1,
        parameters=build_baseline_parameters(),
    )

    assert evaluation.seed == 1
    assert len(evaluation.realization.uav_positions) == 8
    assert len(evaluation.scenario_evaluations) == 3

    total_results = sum(
        len(scenario.results)
        for scenario in evaluation.scenario_evaluations
    )

    assert total_results == 24


def test_scenarios_preserve_expected_order() -> None:
    evaluation = evaluate_seed(
        seed=1,
        parameters=build_baseline_parameters(),
    )

    scenarios = tuple(
        result.scenario
        for result in evaluation.scenario_evaluations
    )

    assert scenarios == ("S1", "S2", "S3")


def test_same_geometry_is_used_in_all_scenarios() -> None:
    evaluation = evaluate_seed(
        seed=1,
        parameters=build_baseline_parameters(),
    )

    scenario_maps = []

    for scenario in evaluation.scenario_evaluations:
        scenario_maps.append(
            {
                result.uav_id: (
                    result.x_m,
                    result.y_m,
                    result.altitude_m,
                    result.distance_to_sv_m,
                    result.channel_gain_linear,
                    result.snr_linear,
                    result.access_rate_bps,
                )
                for result in scenario.results
            }
        )

    assert scenario_maps[0] == scenario_maps[1]
    assert scenario_maps[1] == scenario_maps[2]


def test_summary_matches_detailed_results() -> None:
    evaluation = evaluate_seed(
        seed=1,
        parameters=build_baseline_parameters(),
    )

    for scenario in evaluation.scenario_evaluations:
        latencies = [
            result.end_to_end_latency_s
            for result in scenario.results
        ]

        summary = scenario.summary

        assert summary.number_of_uavs == 8
        assert summary.minimum_latency_s == min(latencies)
        assert summary.maximum_latency_s == max(latencies)
        assert summary.latency_range_s == pytest.approx(
            max(latencies) - min(latencies)
        )


def test_worst_uav_is_the_most_distant_uav() -> None:
    evaluation = evaluate_seed(
        seed=1,
        parameters=build_baseline_parameters(),
    )

    maximum_distance = max(
        position.distance_to_sv_m
        for position in evaluation.realization.uav_positions
    )

    for scenario in evaluation.scenario_evaluations:
        assert (
            scenario.summary.worst_uav_distance_m
            == pytest.approx(maximum_distance)
        )


def test_worst_uav_has_lowest_access_rate() -> None:
    evaluation = evaluate_seed(
        seed=1,
        parameters=build_baseline_parameters(),
    )

    for scenario in evaluation.scenario_evaluations:
        minimum_rate = min(
            result.access_rate_bps
            for result in scenario.results
        )

        assert (
            scenario.summary.worst_uav_access_rate_bps
            == pytest.approx(minimum_rate)
        )


def test_baseline_max_latency_ranking() -> None:
    evaluation = evaluate_seed(
        seed=1,
        parameters=build_baseline_parameters(),
    )

    summaries = {
        scenario.scenario: scenario.summary
        for scenario in evaluation.scenario_evaluations
    }

    assert (
        summaries["S3"].maximum_latency_s
        < summaries["S2"].maximum_latency_s
        < summaries["S1"].maximum_latency_s
    )


def test_baseline_max_latency_ranges() -> None:
    evaluation = evaluate_seed(
        seed=1,
        parameters=build_baseline_parameters(),
    )

    summaries = {
        scenario.scenario: scenario.summary
        for scenario in evaluation.scenario_evaluations
    }

    assert 31.111 < summaries["S1"].maximum_latency_s < 31.123
    assert 25.574 < summaries["S2"].maximum_latency_s < 25.681
    assert 10.310 < summaries["S3"].maximum_latency_s < 10.417


def test_repeated_seed_is_deterministic() -> None:
    parameters = build_baseline_parameters()

    first = evaluate_seed(
        seed=17,
        parameters=parameters,
    )

    second = evaluate_seed(
        seed=17,
        parameters=parameters,
    )

    assert first == second


def test_different_seeds_change_the_realization() -> None:
    parameters = build_baseline_parameters()

    first = evaluate_seed(
        seed=1,
        parameters=parameters,
    )

    second = evaluate_seed(
        seed=2,
        parameters=parameters,
    )

    assert first.realization != second.realization
