from dataclasses import replace

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
    evaluate_experiment,
)
from ucc.model import ModelValidationError


SEEDS = tuple(range(1, 51))


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


@pytest.fixture(scope="module")
def baseline_result():
    return evaluate_experiment(
        seeds=SEEDS,
        parameters=build_baseline_parameters(),
    )


def test_complete_experiment_counts(
    baseline_result,
) -> None:
    assert len(baseline_result.seeds) == 50
    assert len(baseline_result.seed_evaluations) == 50
    assert len(baseline_result.paired_comparisons) == 50
    assert len(baseline_result.scenario_summaries) == 3

    detailed_result_count = sum(
        len(scenario.results)
        for seed_evaluation
        in baseline_result.seed_evaluations
        for scenario
        in seed_evaluation.scenario_evaluations
    )

    assert detailed_result_count == 1_200


def test_scenario_summary_order(
    baseline_result,
) -> None:
    scenarios = tuple(
        summary.scenario
        for summary in baseline_result.scenario_summaries
    )

    assert scenarios == ("S1", "S2", "S3")


def test_baseline_mean_maximum_latency_ranking(
    baseline_result,
) -> None:
    summaries = {
        summary.scenario: summary
        for summary in baseline_result.scenario_summaries
    }

    assert (
        summaries["S3"].mean_of_max_latency_s
        < summaries["S2"].mean_of_max_latency_s
        < summaries["S1"].mean_of_max_latency_s
    )


def test_rcc_wins_all_baseline_realizations(
    baseline_result,
) -> None:
    for comparison in baseline_result.paired_comparisons:
        assert comparison.best_scenarios == ("S3",)

        assert comparison.s1_minus_s2_s > 0.0
        assert comparison.s1_minus_s3_s > 0.0
        assert comparison.s2_minus_s3_s > 0.0


def test_win_counts_are_consistent(
    baseline_result,
) -> None:
    summaries = {
        summary.scenario: summary
        for summary in baseline_result.scenario_summaries
    }

    assert summaries["S1"].number_of_wins == 0
    assert summaries["S2"].number_of_wins == 0
    assert summaries["S3"].number_of_wins == 50

    assert summaries["S1"].number_of_ties == 0
    assert summaries["S2"].number_of_ties == 0
    assert summaries["S3"].number_of_ties == 0

    assert summaries["S1"].win_rate == 0.0
    assert summaries["S2"].win_rate == 0.0
    assert summaries["S3"].win_rate == 1.0


def test_confidence_intervals_contain_means(
    baseline_result,
) -> None:
    for summary in baseline_result.scenario_summaries:
        assert (
            summary.confidence_interval_95_lower_s
            <= summary.mean_of_max_latency_s
            <= summary.confidence_interval_95_upper_s
        )

        assert summary.std_of_max_latency_s >= 0.0


def test_all_summaries_use_50_repetitions(
    baseline_result,
) -> None:
    for summary in baseline_result.scenario_summaries:
        assert summary.number_of_repetitions == 50
        assert summary.number_of_uavs == 8


def test_paired_differences_are_exact(
    baseline_result,
) -> None:
    for comparison in baseline_result.paired_comparisons:
        assert comparison.s1_minus_s2_s == pytest.approx(
            comparison.s1_max_latency_s
            - comparison.s2_max_latency_s
        )

        assert comparison.s1_minus_s3_s == pytest.approx(
            comparison.s1_max_latency_s
            - comparison.s3_max_latency_s
        )

        assert comparison.s2_minus_s3_s == pytest.approx(
            comparison.s2_max_latency_s
            - comparison.s3_max_latency_s
        )


def test_single_seed_has_zero_sample_deviation() -> None:
    result = evaluate_experiment(
        seeds=(1,),
        parameters=build_baseline_parameters(),
    )

    for summary in result.scenario_summaries:
        assert summary.std_of_max_latency_s == 0.0

        assert (
            summary.confidence_interval_95_lower_s
            == summary.mean_of_max_latency_s
        )

        assert (
            summary.confidence_interval_95_upper_s
            == summary.mean_of_max_latency_s
        )


def test_duplicate_seeds_are_rejected() -> None:
    with pytest.raises(ModelValidationError):
        evaluate_experiment(
            seeds=(1, 1, 2),
            parameters=build_baseline_parameters(),
        )


def test_complete_experiment_requires_three_scenarios() -> None:
    parameters = replace(
        build_baseline_parameters(),
        scenarios=("S1", "S2"),
    )

    with pytest.raises(ModelValidationError):
        evaluate_experiment(
            seeds=(1, 2),
            parameters=parameters,
        )


def test_repeated_experiment_is_deterministic() -> None:
    parameters = build_baseline_parameters()

    first = evaluate_experiment(
        seeds=(1, 2, 3),
        parameters=parameters,
    )

    second = evaluate_experiment(
        seeds=(1, 2, 3),
        parameters=parameters,
    )

    assert first == second
