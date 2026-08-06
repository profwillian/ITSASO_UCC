"""Experimental orchestration for the UCC maritime simulation."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

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
