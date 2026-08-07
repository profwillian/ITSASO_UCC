"""Tests for the complete UAV scalability experiment."""

from pathlib import Path

import pytest

from ucc.config import ResolvedConfig, load_and_resolve_config
from ucc.experiment import evaluate_scalability_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "cnf"
    / "ucc_config.json"
)


@pytest.fixture(scope="module")
def config() -> ResolvedConfig:
    """Load the scalability configuration once."""
    return load_and_resolve_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def scalability_result(
    config: ResolvedConfig,
):
    """Run the complete scalability experiment once."""
    return evaluate_scalability_experiment(config)


def test_scalability_experiment_contains_all_uav_counts(
    scalability_result,
) -> None:
    """The experiment must preserve the configured UAV order."""
    assert scalability_result.uav_counts == (
        4,
        6,
        8,
        10,
        12,
    )

    observed = tuple(
        point.parameters.number_of_uavs
        for point in scalability_result.point_results
    )

    assert observed == (
        4,
        6,
        8,
        10,
        12,
    )


def test_scalability_experiment_has_five_points(
    scalability_result,
) -> None:
    """Exactly five scalability points must be evaluated."""
    assert len(scalability_result.point_results) == 5


def test_scalability_experiment_realization_count(
    scalability_result,
) -> None:
    """Five UAV counts times 50 seeds gives 250 realizations."""
    assert (
        scalability_result.total_spatial_realizations
        == 250
    )


def test_scalability_experiment_per_uav_count(
    scalability_result,
) -> None:
    """The complete experiment must produce 6000 UAV evaluations."""
    assert (
        scalability_result.total_per_uav_evaluations
        == 6000
    )


@pytest.mark.parametrize(
    (
        "index",
        "number_of_uavs",
        "expected_evaluations",
    ),
    [
        (0, 4, 600),
        (1, 6, 900),
        (2, 8, 1200),
        (3, 10, 1500),
        (4, 12, 1800),
    ],
)
def test_each_scalability_point_has_expected_size(
    scalability_result,
    index: int,
    number_of_uavs: int,
    expected_evaluations: int,
) -> None:
    """Check the number of detailed evaluations for each N."""
    point = scalability_result.point_results[index]

    assert (
        point.parameters.number_of_uavs
        == number_of_uavs
    )

    experiment = point.experiment_result

    assert len(experiment.seed_evaluations) == 50
    assert len(experiment.scenario_summaries) == 3
    assert len(experiment.paired_comparisons) == 50

    observed = sum(
        len(scenario.results)
        for seed_evaluation
        in experiment.seed_evaluations
        for scenario
        in seed_evaluation.scenario_evaluations
    )

    assert observed == expected_evaluations


def test_every_point_contains_all_three_scenarios(
    scalability_result,
) -> None:
    """Each N must contain S1, S2 and S3 summaries."""
    for point in scalability_result.point_results:
        scenarios = tuple(
            summary.scenario
            for summary
            in point.experiment_result.scenario_summaries
        )

        assert scenarios == (
            "S1",
            "S2",
            "S3",
        )


def test_all_scenario_latencies_are_positive(
    scalability_result,
) -> None:
    """All consolidated maximum latencies must be positive."""
    for point in scalability_result.point_results:
        for summary in (
            point.experiment_result.scenario_summaries
        ):
            assert summary.mean_of_max_latency_s > 0.0
            assert summary.median_of_max_latency_s > 0.0
            assert summary.minimum_of_max_latency_s > 0.0
            assert summary.maximum_of_max_latency_s > 0.0


def test_maximum_latency_bounds_are_consistent(
    scalability_result,
) -> None:
    """Minimum, mean and maximum statistics must be ordered."""
    for point in scalability_result.point_results:
        for summary in (
            point.experiment_result.scenario_summaries
        ):
            assert (
                summary.minimum_of_max_latency_s
                <= summary.mean_of_max_latency_s
                <= summary.maximum_of_max_latency_s
            )


def test_wins_and_ties_do_not_exceed_repetitions(
    scalability_result,
) -> None:
    """Scenario wins and ties must remain within 50 repetitions."""
    for point in scalability_result.point_results:
        for summary in (
            point.experiment_result.scenario_summaries
        ):
            assert summary.number_of_wins >= 0
            assert summary.number_of_ties >= 0

            assert (
                summary.number_of_wins
                + summary.number_of_ties
                <= 50
            )
