"""Experimental orchestration for the UCC maritime simulation."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from ucc.config import (
    ResolvedConfig,
    resolve_factorized_grid,
)

from ucc.geometry import (
    SpatialRealization,
    generate_spatial_realization,
)

from ucc.model import (
    ModelValidationError,
    UAVScenarioResult,
    compute_backhaul_rate_per_flow,
    compute_bandwidth_per_uav,
    evaluate_uav_scenario,
    get_scenario_definition,
)


@dataclass(frozen=True, slots=True)
class ExperimentParameters:
    """Resolved SI parameters required to evaluate one experiment."""

    number_of_uavs: int

    area_side_m: float
    altitude_m: float
    sv_x_m: float
    sv_y_m: float

    reference_distance_m: float
    reference_gain_linear: float
    transmit_power_w: float
    noise_psd_w_hz: float
    total_bandwidth_hz: float

    aggregate_backhaul_capacity_bps: float
    backhaul_fixed_delay_s: float

    input_payload_bits: float
    output_payload_bits: float
    workload_cycles: float

    uav_capacity_cycles_s: float
    sv_capacity_cycles_s: float
    rcc_capacity_cycles_s: float

    scenarios: tuple[str, ...] = ("S1", "S2", "S3")


@dataclass(frozen=True, slots=True)
class RepetitionSummary:
    """Summary of one scenario in one spatial realization."""

    seed: int
    scenario: str
    execution_tier: str
    number_of_uavs: int

    minimum_latency_s: float
    mean_latency_s: float
    median_latency_s: float
    maximum_latency_s: float
    standard_deviation_s: float
    latency_range_s: float

    best_uav_id: int
    worst_uav_id: int

    best_uav_distance_m: float
    worst_uav_distance_m: float

    best_uav_access_rate_bps: float
    worst_uav_access_rate_bps: float


@dataclass(frozen=True, slots=True)
class ScenarioEvaluation:
    """Detailed and summarized result for one scenario."""

    scenario: str
    results: tuple[UAVScenarioResult, ...]
    summary: RepetitionSummary


@dataclass(frozen=True, slots=True)
class SeedEvaluation:
    """Complete evaluation of one spatial seed."""

    seed: int
    realization: SpatialRealization
    scenario_evaluations: tuple[ScenarioEvaluation, ...]


def _validate_scenarios(
    scenarios: tuple[str, ...],
) -> tuple[str, ...]:
    """Validate scenarios and return normalized identifiers."""
    if not scenarios:
        raise ModelValidationError(
            "At least one scenario must be provided."
        )

    normalized: list[str] = []

    for scenario in scenarios:
        definition = get_scenario_definition(scenario)
        normalized.append(definition.scenario)

    if len(set(normalized)) != len(normalized):
        raise ModelValidationError(
            "Scenario identifiers must not be duplicated."
        )

    return tuple(normalized)

def _validate_seeds(
    seeds: tuple[int, ...],
) -> tuple[int, ...]:
    """Validate a non-empty sequence of unique random seeds."""
    if not seeds:
        raise ModelValidationError(
            "At least one seed must be provided."
        )

    normalized: list[int] = []

    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ModelValidationError(
                f"Each seed must be an integer. Received: {seed!r}"
            )

        if seed < 0:
            raise ModelValidationError(
                f"Each seed must be non-negative. Received: {seed}"
            )

        normalized.append(seed)

    if len(set(normalized)) != len(normalized):
        raise ModelValidationError(
            "Seed identifiers must not be duplicated."
        )

    return tuple(normalized)

def evaluate_realization(
    realization: SpatialRealization,
    scenario: str,
    parameters: ExperimentParameters,
) -> tuple[UAVScenarioResult, ...]:
    """Evaluate all UAVs of one realization under one scenario."""
    definition = get_scenario_definition(scenario)

    if not isinstance(realization, SpatialRealization):
        raise ModelValidationError(
            "realization must be an instance of SpatialRealization."
        )

    if len(realization.uav_positions) != parameters.number_of_uavs:
        raise ModelValidationError(
            "Spatial realization contains an unexpected number of UAVs."
        )

    allocated_bandwidth_hz = compute_bandwidth_per_uav(
        total_bandwidth_hz=parameters.total_bandwidth_hz,
        number_of_uavs=parameters.number_of_uavs,
    )

    backhaul_rate_bps = compute_backhaul_rate_per_flow(
        aggregate_capacity_bps=(
            parameters.aggregate_backhaul_capacity_bps
        ),
        number_of_uavs=parameters.number_of_uavs,
    )

    results = tuple(
        evaluate_uav_scenario(
            seed=realization.seed,
            scenario=definition.scenario,
            position=position,
            reference_distance_m=parameters.reference_distance_m,
            reference_gain_linear=parameters.reference_gain_linear,
            transmit_power_w=parameters.transmit_power_w,
            noise_psd_w_hz=parameters.noise_psd_w_hz,
            allocated_bandwidth_hz=allocated_bandwidth_hz,
            backhaul_rate_bps=backhaul_rate_bps,
            input_payload_bits=parameters.input_payload_bits,
            output_payload_bits=parameters.output_payload_bits,
            workload_cycles=parameters.workload_cycles,
            uav_capacity_cycles_s=parameters.uav_capacity_cycles_s,
            sv_capacity_cycles_s=parameters.sv_capacity_cycles_s,
            rcc_capacity_cycles_s=parameters.rcc_capacity_cycles_s,
            number_of_uavs=parameters.number_of_uavs,
            backhaul_fixed_delay_s=(
                parameters.backhaul_fixed_delay_s
            ),
        )
        for position in realization.uav_positions
    )

    if len(results) != parameters.number_of_uavs:
        raise ModelValidationError(
            "Scenario evaluation produced an unexpected result count."
        )

    uav_ids = tuple(result.uav_id for result in results)

    if len(set(uav_ids)) != len(uav_ids):
        raise ModelValidationError(
            "Scenario evaluation contains duplicated UAV identifiers."
        )

    return tuple(
        sorted(
            results,
            key=lambda result: result.uav_id,
        )
    )


def summarize_repetition(
    results: tuple[UAVScenarioResult, ...],
) -> RepetitionSummary:
    """Summarize all UAV results for one seed and one scenario."""
    if not results:
        raise ModelValidationError(
            "Cannot summarize an empty result collection."
        )

    seeds = {result.seed for result in results}
    scenarios = {result.scenario for result in results}
    tiers = {result.execution_tier for result in results}
    uav_ids = {result.uav_id for result in results}

    if len(seeds) != 1:
        raise ModelValidationError(
            "All results must belong to the same seed."
        )

    if len(scenarios) != 1:
        raise ModelValidationError(
            "All results must belong to the same scenario."
        )

    if len(tiers) != 1:
        raise ModelValidationError(
            "All results must belong to the same execution tier."
        )

    if len(uav_ids) != len(results):
        raise ModelValidationError(
            "Result collection contains duplicated UAV identifiers."
        )

    latencies = tuple(
        result.end_to_end_latency_s
        for result in results
    )

    if not all(
        math.isfinite(latency) and latency > 0.0
        for latency in latencies
    ):
        raise ModelValidationError(
            "Result collection contains invalid latency values."
        )

    best_result = min(
        results,
        key=lambda result: (
            result.end_to_end_latency_s,
            result.uav_id,
        ),
    )

    worst_result = min(
        results,
        key=lambda result: (
            -result.end_to_end_latency_s,
            result.uav_id,
        ),
    )

    minimum_latency = min(latencies)
    maximum_latency = max(latencies)

    return RepetitionSummary(
        seed=next(iter(seeds)),
        scenario=next(iter(scenarios)),
        execution_tier=next(iter(tiers)),
        number_of_uavs=len(results),
        minimum_latency_s=minimum_latency,
        mean_latency_s=statistics.fmean(latencies),
        median_latency_s=statistics.median(latencies),
        maximum_latency_s=maximum_latency,
        standard_deviation_s=statistics.pstdev(latencies),
        latency_range_s=maximum_latency - minimum_latency,
        best_uav_id=best_result.uav_id,
        worst_uav_id=worst_result.uav_id,
        best_uav_distance_m=best_result.distance_to_sv_m,
        worst_uav_distance_m=worst_result.distance_to_sv_m,
        best_uav_access_rate_bps=best_result.access_rate_bps,
        worst_uav_access_rate_bps=worst_result.access_rate_bps,
    )


def evaluate_seed(
    seed: int,
    parameters: ExperimentParameters,
) -> SeedEvaluation:
    """Generate and evaluate one complete spatial realization."""
    scenarios = _validate_scenarios(parameters.scenarios)

    realization = generate_spatial_realization(
        number_of_uavs=parameters.number_of_uavs,
        area_side_m=parameters.area_side_m,
        altitude_m=parameters.altitude_m,
        sv_x_m=parameters.sv_x_m,
        sv_y_m=parameters.sv_y_m,
        seed=seed,
    )

    scenario_evaluations: list[ScenarioEvaluation] = []

    for scenario in scenarios:
        results = evaluate_realization(
            realization=realization,
            scenario=scenario,
            parameters=parameters,
        )

        summary = summarize_repetition(results)

        scenario_evaluations.append(
            ScenarioEvaluation(
                scenario=scenario,
                results=results,
                summary=summary,
            )
        )

    return SeedEvaluation(
        seed=seed,
        realization=realization,
        scenario_evaluations=tuple(scenario_evaluations),
    )

def compare_seed_scenarios(
    evaluation: SeedEvaluation,
    tie_tolerance_s: float = 1.0e-9,
) -> PairedComparison:
    """Compare S1, S2, and S3 using maximum latency for one seed."""
    if not isinstance(evaluation, SeedEvaluation):
        raise ModelValidationError(
            "evaluation must be an instance of SeedEvaluation."
        )

    tolerance = float(tie_tolerance_s)

    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ModelValidationError(
            "tie_tolerance_s must be finite and non-negative."
        )

    maximum_latencies = {
        scenario.scenario: scenario.summary.maximum_latency_s
        for scenario in evaluation.scenario_evaluations
    }

    expected_scenarios = {"S1", "S2", "S3"}

    if set(maximum_latencies) != expected_scenarios:
        raise ModelValidationError(
            "Paired comparison requires exactly S1, S2, and S3."
        )

    best_value = min(maximum_latencies.values())
    worst_value = max(maximum_latencies.values())

    scenario_order = ("S1", "S2", "S3")

    best_scenarios = tuple(
        scenario
        for scenario in scenario_order
        if abs(maximum_latencies[scenario] - best_value)
        <= tolerance
    )

    worst_scenarios = tuple(
        scenario
        for scenario in scenario_order
        if abs(maximum_latencies[scenario] - worst_value)
        <= tolerance
    )

    s1 = maximum_latencies["S1"]
    s2 = maximum_latencies["S2"]
    s3 = maximum_latencies["S3"]

    return PairedComparison(
        seed=evaluation.seed,
        s1_max_latency_s=s1,
        s2_max_latency_s=s2,
        s3_max_latency_s=s3,
        s1_minus_s2_s=s1 - s2,
        s1_minus_s3_s=s1 - s3,
        s2_minus_s3_s=s2 - s3,
        best_scenarios=best_scenarios,
        worst_scenarios=worst_scenarios,
    )


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """Paired comparison of scenario maximum latencies for one seed."""

    seed: int

    s1_max_latency_s: float
    s2_max_latency_s: float
    s3_max_latency_s: float

    s1_minus_s2_s: float
    s1_minus_s3_s: float
    s2_minus_s3_s: float

    best_scenarios: tuple[str, ...]
    worst_scenarios: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScenarioSummary:
    """Aggregate statistics across all spatial realizations."""

    scenario: str
    execution_tier: str

    number_of_repetitions: int
    number_of_uavs: int

    mean_of_max_latency_s: float
    median_of_max_latency_s: float
    minimum_of_max_latency_s: float
    maximum_of_max_latency_s: float
    std_of_max_latency_s: float

    confidence_interval_95_lower_s: float
    confidence_interval_95_upper_s: float

    mean_of_mean_latency_s: float
    mean_of_min_latency_s: float
    mean_latency_range_s: float

    number_of_wins: int
    number_of_ties: int
    win_rate: float


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Complete result across multiple spatial realizations."""

    seeds: tuple[int, ...]
    seed_evaluations: tuple[SeedEvaluation, ...]
    paired_comparisons: tuple[PairedComparison, ...]
    scenario_summaries: tuple[ScenarioSummary, ...]



def summarize_scenario_across_seeds(
    seed_evaluations: tuple[SeedEvaluation, ...],
    paired_comparisons: tuple[PairedComparison, ...],
    scenario: str,
) -> ScenarioSummary:
    """Aggregate one scenario across all spatial realizations."""
    if not seed_evaluations:
        raise ModelValidationError(
            "Cannot summarize an empty experiment."
        )

    definition = get_scenario_definition(scenario)

    summaries: list[RepetitionSummary] = []

    for evaluation in seed_evaluations:
        matching = tuple(
            scenario_evaluation.summary
            for scenario_evaluation
            in evaluation.scenario_evaluations
            if scenario_evaluation.scenario == definition.scenario
        )

        if len(matching) != 1:
            raise ModelValidationError(
                "Each seed must contain exactly one summary "
                f"for scenario {definition.scenario}."
            )

        summaries.append(matching[0])

    repetition_count = len(summaries)

    if len(paired_comparisons) != repetition_count:
        raise ModelValidationError(
            "Paired comparison count is inconsistent "
            "with the number of repetitions."
        )

    number_of_uavs = summaries[0].number_of_uavs

    if any(
        summary.number_of_uavs != number_of_uavs
        for summary in summaries
    ):
        raise ModelValidationError(
            "All repetitions must use the same number of UAVs."
        )

    maximum_latencies = tuple(
        summary.maximum_latency_s
        for summary in summaries
    )

    mean_latencies = tuple(
        summary.mean_latency_s
        for summary in summaries
    )

    minimum_latencies = tuple(
        summary.minimum_latency_s
        for summary in summaries
    )

    latency_ranges = tuple(
        summary.latency_range_s
        for summary in summaries
    )

    mean_of_max = statistics.fmean(maximum_latencies)

    if repetition_count > 1:
        std_of_max = statistics.stdev(maximum_latencies)
    else:
        std_of_max = 0.0

    confidence_half_width = (
        1.96
        * std_of_max
        / math.sqrt(repetition_count)
    )

    number_of_wins = sum(
        comparison.best_scenarios
        == (definition.scenario,)
        for comparison in paired_comparisons
    )

    number_of_ties = sum(
        definition.scenario in comparison.best_scenarios
        and len(comparison.best_scenarios) > 1
        for comparison in paired_comparisons
    )

    return ScenarioSummary(
        scenario=definition.scenario,
        execution_tier=definition.execution_tier,
        number_of_repetitions=repetition_count,
        number_of_uavs=number_of_uavs,
        mean_of_max_latency_s=mean_of_max,
        median_of_max_latency_s=statistics.median(
            maximum_latencies
        ),
        minimum_of_max_latency_s=min(maximum_latencies),
        maximum_of_max_latency_s=max(maximum_latencies),
        std_of_max_latency_s=std_of_max,
        confidence_interval_95_lower_s=(
            mean_of_max - confidence_half_width
        ),
        confidence_interval_95_upper_s=(
            mean_of_max + confidence_half_width
        ),
        mean_of_mean_latency_s=statistics.fmean(
            mean_latencies
        ),
        mean_of_min_latency_s=statistics.fmean(
            minimum_latencies
        ),
        mean_latency_range_s=statistics.fmean(
            latency_ranges
        ),
        number_of_wins=number_of_wins,
        number_of_ties=number_of_ties,
        win_rate=number_of_wins / repetition_count,
    )

def evaluate_experiment(
    seeds: tuple[int, ...],
    parameters: ExperimentParameters,
    tie_tolerance_s: float = 1.0e-9,
) -> ExperimentResult:
    """Evaluate all configured scenarios across multiple seeds."""
    validated_seeds = _validate_seeds(seeds)
    validated_scenarios = _validate_scenarios(
        parameters.scenarios
    )

    if validated_scenarios != ("S1", "S2", "S3"):
        raise ModelValidationError(
            "The complete experiment requires scenarios "
            "in the order S1, S2, S3."
        )

    seed_evaluations = tuple(
        evaluate_seed(
            seed=seed,
            parameters=parameters,
        )
        for seed in validated_seeds
    )

    paired_comparisons = tuple(
        compare_seed_scenarios(
            evaluation=evaluation,
            tie_tolerance_s=tie_tolerance_s,
        )
        for evaluation in seed_evaluations
    )

    scenario_summaries = tuple(
        summarize_scenario_across_seeds(
            seed_evaluations=seed_evaluations,
            paired_comparisons=paired_comparisons,
            scenario=scenario,
        )
        for scenario in validated_scenarios
    )

    expected_evaluation_count = (
        len(validated_seeds)
        * parameters.number_of_uavs
        * len(validated_scenarios)
    )

    observed_evaluation_count = sum(
        len(scenario_evaluation.results)
        for seed_evaluation in seed_evaluations
        for scenario_evaluation
        in seed_evaluation.scenario_evaluations
    )

    if observed_evaluation_count != expected_evaluation_count:
        raise ModelValidationError(
            "The experiment produced an unexpected number "
            "of per-UAV evaluations."
        )

    return ExperimentResult(
        seeds=validated_seeds,
        seed_evaluations=seed_evaluations,
        paired_comparisons=paired_comparisons,
        scenario_summaries=scenario_summaries,
    )
@dataclass(frozen=True, slots=True)
class ScalabilityPointParameters:
    """Resolved parameters for one UAV-count scalability point."""

    number_of_uavs: int

    grid_rows: int
    grid_columns: int

    bandwidth_per_uav_hz: float
    backhaul_rate_per_flow_bps: float

    sv_capacity_per_job_cycles_s: float
    rcc_capacity_per_job_cycles_s: float

    experiment_parameters: ExperimentParameters


@dataclass(frozen=True, slots=True)
class ScalabilityPointResult:
    """Complete result for one UAV-count scalability point."""

    parameters: ScalabilityPointParameters
    experiment_result: ExperimentResult
def build_scalability_point_parameters(
    config: ResolvedConfig,
    number_of_uavs: int,
) -> ScalabilityPointParameters:
    """Resolve all N-dependent parameters for one scalability point."""
    if not isinstance(config, ResolvedConfig):
        raise ModelValidationError(
            "config must be an instance of ResolvedConfig."
        )

    if (
        isinstance(number_of_uavs, bool)
        or not isinstance(number_of_uavs, int)
    ):
        raise ModelValidationError(
            "number_of_uavs must be an integer."
        )

    if number_of_uavs not in config.uav_counts:
        raise ModelValidationError(
            f"number_of_uavs={number_of_uavs} is not part of "
            f"the configured scalability set {config.uav_counts}."
        )

    grid_rows, grid_columns = resolve_factorized_grid(
        number_of_uavs
    )

    bandwidth_per_uav_hz = (
        config.total_bandwidth_hz / number_of_uavs
    )

    backhaul_rate_per_flow_bps = (
        config.backhaul_capacity_bps / number_of_uavs
    )

    sv_capacity_per_job_cycles_s = (
        config.sv_capacity_cycles_s / number_of_uavs
    )

    rcc_capacity_per_job_cycles_s = (
        config.rcc_capacity_cycles_s / number_of_uavs
    )

    derived_values = (
        bandwidth_per_uav_hz,
        backhaul_rate_per_flow_bps,
        sv_capacity_per_job_cycles_s,
        rcc_capacity_per_job_cycles_s,
    )

    if not all(
        math.isfinite(value) and value > 0.0
        for value in derived_values
    ):
        raise ModelValidationError(
            "Scalability-point derivation produced an invalid value."
        )

    experiment_parameters = ExperimentParameters(
        number_of_uavs=number_of_uavs,
        area_side_m=config.search_area_side_m,
        altitude_m=config.uav_altitude_m,
        sv_x_m=config.sv_x_m,
        sv_y_m=config.sv_y_m,
        reference_distance_m=config.reference_distance_m,
        reference_gain_linear=(
            config.reference_channel_gain_linear
        ),
        transmit_power_w=config.uav_transmit_power_w,
        noise_psd_w_hz=config.noise_psd_w_hz,
        total_bandwidth_hz=config.total_bandwidth_hz,
        aggregate_backhaul_capacity_bps=(
            config.backhaul_capacity_bps
        ),
        backhaul_fixed_delay_s=(
            config.backhaul_fixed_delay_s
        ),
        input_payload_bits=config.input_payload_bits,
        output_payload_bits=config.output_payload_bits,
        workload_cycles=config.workload_cycles,
        uav_capacity_cycles_s=(
            config.uav_capacity_cycles_s
        ),
        sv_capacity_cycles_s=(
            config.sv_capacity_cycles_s
        ),
        rcc_capacity_cycles_s=(
            config.rcc_capacity_cycles_s
        ),
        scenarios=config.scenarios,
    )

    return ScalabilityPointParameters(
        number_of_uavs=number_of_uavs,
        grid_rows=grid_rows,
        grid_columns=grid_columns,
        bandwidth_per_uav_hz=bandwidth_per_uav_hz,
        backhaul_rate_per_flow_bps=(
            backhaul_rate_per_flow_bps
        ),
        sv_capacity_per_job_cycles_s=(
            sv_capacity_per_job_cycles_s
        ),
        rcc_capacity_per_job_cycles_s=(
            rcc_capacity_per_job_cycles_s
        ),
        experiment_parameters=experiment_parameters,
    )
def evaluate_scalability_point(
    config: ResolvedConfig,
    number_of_uavs: int,
) -> ScalabilityPointResult:
    """Evaluate all seeds and scenarios for one UAV count."""
    point_parameters = build_scalability_point_parameters(
        config=config,
        number_of_uavs=number_of_uavs,
    )

    experiment_result = evaluate_experiment(
        seeds=config.seeds,
        parameters=point_parameters.experiment_parameters,
    )

    if len(experiment_result.seed_evaluations) != len(
        config.seeds
    ):
        raise ModelValidationError(
            "Scalability point produced an unexpected "
            "number of spatial realizations."
        )

    expected_evaluations = (
        number_of_uavs
        * len(config.seeds)
        * len(config.scenarios)
    )

    observed_evaluations = sum(
        len(scenario.results)
        for seed_evaluation
        in experiment_result.seed_evaluations
        for scenario
        in seed_evaluation.scenario_evaluations
    )

    if observed_evaluations != expected_evaluations:
        raise ModelValidationError(
            "Scalability point produced an unexpected "
            "number of per-UAV evaluations."
        )

    return ScalabilityPointResult(
        parameters=point_parameters,
        experiment_result=experiment_result,
    )

@dataclass(frozen=True, slots=True)
class ScalabilityExperimentResult:
    """Complete result of the UAV scalability experiment."""

    uav_counts: tuple[int, ...]
    point_results: tuple[ScalabilityPointResult, ...]

    total_spatial_realizations: int
    total_per_uav_evaluations: int

def evaluate_scalability_experiment(
    config: ResolvedConfig,
) -> ScalabilityExperimentResult:
    """Evaluate all configured UAV-count scalability points."""
    if not isinstance(config, ResolvedConfig):
        raise ModelValidationError(
            "config must be an instance of ResolvedConfig."
        )

    if config.experiment_type != "scalability":
        raise ModelValidationError(
            "evaluate_scalability_experiment requires "
            "experiment.type='scalability'."
        )

    if not config.uav_counts:
        raise ModelValidationError(
            "The scalability experiment requires at least "
            "one UAV count."
        )

    point_results = tuple(
        evaluate_scalability_point(
            config=config,
            number_of_uavs=number_of_uavs,
        )
        for number_of_uavs in config.uav_counts
    )

    if len(point_results) != len(config.uav_counts):
        raise ModelValidationError(
            "Unexpected number of scalability-point results."
        )

    observed_uav_counts = tuple(
        point.parameters.number_of_uavs
        for point in point_results
    )

    if observed_uav_counts != config.uav_counts:
        raise ModelValidationError(
            "Scalability points do not match the configured "
            "UAV-count order."
        )

    total_spatial_realizations = sum(
        len(
            point.experiment_result.seed_evaluations
        )
        for point in point_results
    )

    expected_spatial_realizations = (
        len(config.uav_counts)
        * len(config.seeds)
    )

    if (
        total_spatial_realizations
        != expected_spatial_realizations
    ):
        raise ModelValidationError(
            "Unexpected number of spatial realizations "
            "in scalability experiment."
        )

    total_per_uav_evaluations = sum(
        len(scenario.results)
        for point in point_results
        for seed_evaluation
        in point.experiment_result.seed_evaluations
        for scenario
        in seed_evaluation.scenario_evaluations
    )

    expected_per_uav_evaluations = sum(
        number_of_uavs
        * len(config.seeds)
        * len(config.scenarios)
        for number_of_uavs in config.uav_counts
    )

    if (
        total_per_uav_evaluations
        != expected_per_uav_evaluations
    ):
        raise ModelValidationError(
            "Unexpected number of per-UAV evaluations "
            "in scalability experiment."
        )

    return ScalabilityExperimentResult(
        uav_counts=config.uav_counts,
        point_results=point_results,
        total_spatial_realizations=(
            total_spatial_realizations
        ),
        total_per_uav_evaluations=(
            total_per_uav_evaluations
        ),
    )
