"""Tests for UAV scalability-point evaluation."""

from pathlib import Path

import pytest

from ucc.config import ResolvedConfig, load_and_resolve_config
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
def config() -> ResolvedConfig:
    """Load the scalability configuration once for this test module."""
    return load_and_resolve_config(CONFIG_PATH)


def test_resolve_four_uav_scalability_point(
    config: ResolvedConfig,
) -> None:
    """Check the derived parameters for N=4."""
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
    config: ResolvedConfig,
) -> None:
    """Ensure total resources are not divided twice."""
    point = build_scalability_point_parameters(
        config=config,
        number_of_uavs=4,
    )

    parameters = point.experiment_parameters

    assert (
        parameters.total_bandwidth_hz
        == 10_000_000.0
    )

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
    config: ResolvedConfig,
) -> None:
    """Check realization and evaluation counts for N=4."""
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
    config: ResolvedConfig,
) -> None:
    """Ensure N=4 uses a 2x2 spatial decomposition."""
    result = evaluate_scalability_point(
        config=config,
        number_of_uavs=4,
    )

    for seed_evaluation in (
        result.experiment_result.seed_evaluations
    ):
        realization = seed_evaluation.realization

        assert realization.grid_rows == 2
        assert realization.grid_columns == 2

        assert (
            len(realization.uav_positions)
            == 4
        )


def test_four_uav_summaries_report_correct_n(
    config: ResolvedConfig,
) -> None:
    """Ensure summaries preserve N and repetition count."""
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
    config: ResolvedConfig,
) -> None:
    """Repeated runs with the same seeds must be identical."""
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
    [
        0,
        5,
        7,
        8_000,
    ],
)
def test_invalid_scalability_point_is_rejected(
    config: ResolvedConfig,
    invalid_uav_count: int,
) -> None:
    """Reject UAV counts outside the configured experiment."""
    with pytest.raises(ModelValidationError):
        build_scalability_point_parameters(
            config=config,
            number_of_uavs=invalid_uav_count,
        )


@pytest.mark.parametrize(
    (
        "number_of_uavs",
        "expected_rows",
        "expected_columns",
        "expected_bandwidth_hz",
        "expected_backhaul_bps",
        "expected_sv_capacity",
        "expected_rcc_capacity",
    ),
    [
        (
            4,
            2,
            2,
            2_500_000.0,
            1_250_000.0,
            500_000_000.0,
            125_000_000_000.0,
        ),
        (
            6,
            2,
            3,
            1_666_666.6666666667,
            833_333.3333333334,
            333_333_333.3333333,
            83_333_333_333.33333,
        ),
        (
            8,
            2,
            4,
            1_250_000.0,
            625_000.0,
            250_000_000.0,
            62_500_000_000.0,
        ),
        (
            10,
            2,
            5,
            1_000_000.0,
            500_000.0,
            200_000_000.0,
            50_000_000_000.0,
        ),
        (
            12,
            3,
            4,
            833_333.3333333334,
            416_666.6666666667,
            166_666_666.66666666,
            41_666_666_666.666664,
        ),
    ],
)
def test_all_scalability_point_resources(
    config: ResolvedConfig,
    number_of_uavs: int,
    expected_rows: int,
    expected_columns: int,
    expected_bandwidth_hz: float,
    expected_backhaul_bps: float,
    expected_sv_capacity: float,
    expected_rcc_capacity: float,
) -> None:
    """Validate N-dependent resource sharing."""
    point = build_scalability_point_parameters(
        config=config,
        number_of_uavs=number_of_uavs,
    )

    assert point.grid_rows == expected_rows
    assert point.grid_columns == expected_columns

    assert point.bandwidth_per_uav_hz == pytest.approx(
        expected_bandwidth_hz
    )

    assert (
        point.backhaul_rate_per_flow_bps
        == pytest.approx(
            expected_backhaul_bps
        )
    )

    assert (
        point.sv_capacity_per_job_cycles_s
        == pytest.approx(
            expected_sv_capacity
        )
    )

    assert (
        point.rcc_capacity_per_job_cycles_s
        == pytest.approx(
            expected_rcc_capacity
        )
    )


@pytest.mark.parametrize(
    (
        "number_of_uavs",
        "expected_evaluations",
    ),
    [
        (4, 600),
        (6, 900),
        (8, 1200),
        (10, 1500),
        (12, 1800),
    ],
)
def test_all_scalability_point_evaluation_counts(
    config: ResolvedConfig,
    number_of_uavs: int,
    expected_evaluations: int,
) -> None:
    """Validate evaluation counts for every scalability point."""
    result = evaluate_scalability_point(
        config=config,
        number_of_uavs=number_of_uavs,
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

    assert detailed_count == expected_evaluations


@pytest.mark.parametrize(
    (
        "number_of_uavs",
        "expected_rows",
        "expected_columns",
    ),
    [
        (4, 2, 2),
        (6, 2, 3),
        (8, 2, 4),
        (10, 2, 5),
        (12, 3, 4),
    ],
)
def test_scalability_realization_grids(
    config: ResolvedConfig,
    number_of_uavs: int,
    expected_rows: int,
    expected_columns: int,
) -> None:
    """Check the spatial grid actually used in every realization."""
    result = evaluate_scalability_point(
        config=config,
        number_of_uavs=number_of_uavs,
    )

    for seed_evaluation in (
        result.experiment_result.seed_evaluations
    ):
        realization = seed_evaluation.realization

        assert realization.grid_rows == expected_rows
        assert realization.grid_columns == expected_columns

        assert (
            len(realization.uav_positions)
            == number_of_uavs
        )


@pytest.mark.parametrize(
    "number_of_uavs",
    [
        4,
        6,
        8,
        10,
        12,
    ],
)
def test_scenarios_share_geometry_within_each_seed(
    config: ResolvedConfig,
    number_of_uavs: int,
) -> None:
    """
    Ensure S1, S2 and S3 use exactly the same geometry
    and access-link conditions within each seed.
    """
    result = evaluate_scalability_point(
        config=config,
        number_of_uavs=number_of_uavs,
    )

    for seed_evaluation in (
        result.experiment_result.seed_evaluations
    ):
        scenario_evaluations = (
            seed_evaluation.scenario_evaluations
        )

        assert len(scenario_evaluations) == 3

        reference = scenario_evaluations[0].results

        for scenario in scenario_evaluations[1:]:
            assert (
                len(scenario.results)
                == len(reference)
            )

            for reference_result, result_item in zip(
                reference,
                scenario.results,
            ):
                assert (
                    result_item.uav_id
                    == reference_result.uav_id
                )

                assert result_item.x_m == pytest.approx(
                    reference_result.x_m
                )

                assert result_item.y_m == pytest.approx(
                    reference_result.y_m
                )

                assert (
                    result_item.altitude_m
                    == pytest.approx(
                        reference_result.altitude_m
                    )
                )

                assert (
                    result_item.distance_to_sv_m
                    == pytest.approx(
                        reference_result.distance_to_sv_m
                    )
                )

                assert (
                    result_item.channel_gain_linear
                    == pytest.approx(
                        reference_result.channel_gain_linear
                    )
                )

                assert (
                    result_item.snr_linear
                    == pytest.approx(
                        reference_result.snr_linear
                    )
                )

                assert (
                    result_item.access_rate_bps
                    == pytest.approx(
                        reference_result.access_rate_bps
                    )
                )
