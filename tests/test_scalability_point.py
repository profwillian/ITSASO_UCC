from pathlib import Path

import pytest

from ucc.config import load_and_resolve_config
from ucc.experiment import (
    build_scalability_point_parameters,
    evaluate_scalability_point,
)
from ucc.model import ModelValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "cnf"
    / "ucc_config.json"
)


@pytest.fixture(scope="module")
def config():
    return load_and_resolve_config(CONFIG_PATH)


def test_resolve_four_uav_scalability_point(
    config,
) -> None:
    point = build_scalability_point_parameters(
        config=config,
        number_of_uavs=4,
    )

    assert point.number_of_uavs == 4

    assert point.grid_rows == 2
    assert point.grid_columns == 2

    assert point.bandwidth_per_uav_hz == 2_500_000.0

    assert (
        point.backhaul_rate_per_flow_bps
        == 1_250_000.0
    )

    assert (
        point.sv_capacity_per_job_cycles_s
        == 500_000_000.0
    )

    assert (
        point.rcc_capacity_per_job_cycles_s
        == 125_000_000_000.0
    )

    assert (
        point.experiment_parameters.number_of_uavs
        == 4
    )


def test_four_uav_point_preserves_total_resources(
    config,
) -> None:
    point = build_scalability_point_parameters(
        config=config,
        number_of_uavs=4,
    )

    parameters = point.experiment_parameters

    assert parameters.total_bandwidth_hz == 10_000_000.0

    assert (
        parameters.aggregate_backhaul_capacity_bps
        == 5_000_000.0
    )

    assert (
        parameters.sv_capacity_cycles_s
        == 2_000_000_000.0
    )

    assert (
        parameters.rcc_capacity_cycles_s
        == 500_000_000_000.0
    )


def test_four_uav_complete_point_counts(
    config,
) -> None:
    result = evaluate_scalability_point(
        config=config,
        number_of_uavs=4,
    )

    experiment = result.experiment_result

    assert len(experiment.seed_evaluations) == 50
    assert len(experiment.paired_comparisons) == 50
    assert len(experiment.scenario_summaries) == 3

    detailed_count = sum(
        len(scenario.results)
        for seed_evaluation
        in experiment.seed_evaluations
        for scenario
        in seed_evaluation.scenario_evaluations
    )

    assert detailed_count == 600


def test_four_uav_realizations_use_two_by_two_grid(
    config,
) -> None:
    result = evaluate_scalability_point(
        config=config,
        number_of_uavs=4,
    )

    for seed_evaluation in (
        result.experiment_result.seed_evaluations
    ):
        assert seed_evaluation.realization.grid_rows == 2
        assert seed_evaluation.realization.grid_columns == 2

        assert (
            len(seed_evaluation.realization.uav_positions)
            == 4
        )


def test_four_uav_summaries_report_correct_n(
    config,
) -> None:
    result = evaluate_scalability_point(
        config=config,
        number_of_uavs=4,
    )

    for summary in (
        result.experiment_result.scenario_summaries
    ):
        assert summary.number_of_uavs == 4
        assert summary.number_of_repetitions == 50


def test_four_uav_point_is_deterministic(
    config,
) -> None:
    first = evaluate_scalability_point(
        config=config,
        number_of_uavs=4,
    )

    second = evaluate_scalability_point(
        config=config,
        number_of_uavs=4,
    )

    assert first == second


@pytest.mark.parametrize(
    "invalid_uav_count",
    [0, 5, 7, 8_000],
)
def test_invalid_scalability_point_is_rejected(
    config,
    invalid_uav_count: int,
) -> None:
    with pytest.raises(ModelValidationError):
        build_scalability_point_parameters(
            config=config,
            number_of_uavs=invalid_uav_count,
        )
