"""Command-line entry point for the UCC maritime scalability simulation."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from ucc.config import (
    ConfigurationError,
    ResolvedConfig,
    load_and_resolve_config,
)
from ucc.experiment import (
    ScalabilityExperimentResult,
    ScalabilityPointResult,
    build_scalability_point_parameters,
    evaluate_scalability_experiment,
)
from ucc.model import ModelValidationError
from ucc.plots import generate_scalability_latency_plot
from ucc.results import (
    RunArtifacts,
    write_run_artifacts,
)


SCENARIO_LABELS = {
    "S1": "All-UAV",
    "S2": "All-Fog",
    "S3": "All-RCC",
}

SEPARATOR_WIDTH = 96


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the equation-based UCC maritime UAV scalability "
            "experiment."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the UCC JSON configuration file.",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate and resolve the configuration without "
            "running the experiment."
        ),
    )

    return parser


def scenario_label(scenario: str) -> str:
    """Return the human-readable scenario label."""
    return SCENARIO_LABELS.get(
        scenario,
        scenario,
    )


def linear_gain_to_db(value: float) -> float:
    """Convert a positive linear power gain into dB."""
    if not math.isfinite(value) or value <= 0.0:
        raise ModelValidationError(
            "Channel gain must be positive and finite."
        )

    return 10.0 * math.log10(value)


def watts_per_hz_to_dbm_per_hz(
    value: float,
) -> float:
    """Convert noise PSD from W/Hz to dBm/Hz."""
    if not math.isfinite(value) or value <= 0.0:
        raise ModelValidationError(
            "Noise PSD must be positive and finite."
        )

    return 10.0 * math.log10(value) + 30.0


def latency_ms(value_s: float) -> float:
    """Convert latency from seconds to milliseconds."""
    return value_s * 1000.0


def print_section(title: str) -> None:
    """Print a section heading."""
    print()
    print(title)
    print("-" * SEPARATOR_WIDTH)


def print_resolved_configuration(
    config: ResolvedConfig,
) -> None:
    """Print all resolved experimental parameters."""
    print("=" * SEPARATOR_WIDTH)
    print("UCC MARITIME SCALABILITY EXPERIMENT")
    print("=" * SEPARATOR_WIDTH)

    print_section("EXPERIMENT")

    print(
        f"Name:                         "
        f"{config.experiment_name}"
    )
    print(
        f"Description:                  "
        f"{config.experiment_description}"
    )
    print(
        f"Type:                         "
        f"{config.experiment_type}"
    )
    print(
        f"Schema version:               "
        f"{config.schema_version}"
    )
    print(
        f"Execution backend:            "
        f"{config.execution_backend}"
    )
    print(
        f"Time mode:                    "
        f"{config.time_mode}"
    )
    print(
        f"Scenarios:                    "
        f"{', '.join(config.scenarios)}"
    )
    print(
        f"Scenario labels:              "
        f"{', '.join(scenario_label(s) for s in config.scenarios)}"
    )
    print(
        f"UAV counts:                   "
        f"{', '.join(str(n) for n in config.uav_counts)}"
    )
    print(
        f"Repetitions per UAV count:    "
        f"{len(config.seeds)}"
    )
    print(
        f"Seeds:                        "
        f"{config.seeds[0]}..{config.seeds[-1]}"
    )
    print(
        f"Paired scenario geometry:     "
        f"{config.paired_geometry}"
    )

    print_section("GEOMETRY")

    print(
        f"Search area:                  "
        f"{config.search_area_side_m:.3f} x "
        f"{config.search_area_side_m:.3f} m"
    )
    print(
        f"UAV altitude:                 "
        f"{config.uav_altitude_m:.3f} m"
    )
    print(
        f"SV position:                  "
        f"({config.sv_x_m:.3f}, "
        f"{config.sv_y_m:.3f}) m"
    )

    print_section("WORKLOAD")

    print(
        f"Chunks per job (Q):           "
        f"{config.chunks_per_job}"
    )
    print(
        f"Chunk size (D):               "
        f"{config.chunk_size_bits / 1e6:.6f} Mbit"
    )
    print(
        f"Input payload (P):            "
        f"{config.input_payload_bits / 1e6:.6f} Mbit"
    )
    print(
        f"Output/input ratio (mu):      "
        f"{config.output_input_ratio:.6f}"
    )
    print(
        f"Output payload:               "
        f"{config.output_payload_bits / 1e6:.6f} Mbit"
    )
    print(
        f"Compute intensity:            "
        f"{config.compute_intensity_cycles_per_bit:.3f} "
        f"cycles/bit"
    )
    print(
        f"Workload:                     "
        f"{config.workload_cycles / 1e9:.6f} Gcycles"
    )

    print_section("COMPUTATION")

    print(
        f"UAV capacity (f1):            "
        f"{config.uav_capacity_cycles_s / 1e9:.6f} GHz"
    )
    print(
        f"Fog/SV total capacity (f2):   "
        f"{config.sv_capacity_cycles_s / 1e9:.6f} GHz"
    )
    print(
        f"RCC total capacity (f3):      "
        f"{config.rcc_capacity_cycles_s / 1e9:.6f} GHz"
    )
    print(
        "Remote-tier sharing:          "
        "static equal sharing"
    )

    reference_gain_db = linear_gain_to_db(
        config.reference_channel_gain_linear
    )

    noise_psd_dbm_hz = watts_per_hz_to_dbm_per_hz(
        config.noise_psd_w_hz
    )

    print_section("UAV-TO-SV ACCESS LINK")

    print(
        f"Reference distance (d0):      "
        f"{config.reference_distance_m:.6f} m"
    )
    print(
        f"Total bandwidth (B):          "
        f"{config.total_bandwidth_hz / 1e6:.6f} MHz"
    )
    print(
        f"UAV transmit power (p):       "
        f"{config.uav_transmit_power_w:.6f} W"
    )
    print(
        f"Reference channel gain:       "
        f"{reference_gain_db:.3f} dB"
    )
    print(
        f"Noise PSD (N0):               "
        f"{noise_psd_dbm_hz:.3f} dBm/Hz"
    )
    print(
        "Access sharing:               "
        "static equal bandwidth sharing"
    )

    print_section("SV-TO-RCC BACKHAUL")

    print(
        f"Aggregate capacity (C_bh):    "
        f"{config.backhaul_capacity_bps / 1e6:.6f} Mbit/s"
    )
    print(
        f"Fixed backhaul delay:         "
        f"{latency_ms(config.backhaul_fixed_delay_s):.3f} ms"
    )
    print(
        "Backhaul sharing:             "
        "static equal flow sharing"
    )

    print_section("SCALABILITY-DERIVED RESOURCES")

    print(
        f"{'N':>4} "
        f"{'Grid':>8} "
        f"{'B/UAV (MHz)':>15} "
        f"{'BH/flow (Mbit/s)':>19} "
        f"{'Fog/job (GHz)':>16} "
        f"{'RCC/job (GHz)':>16}"
    )

    print("-" * SEPARATOR_WIDTH)

    for number_of_uavs in config.uav_counts:
        point = build_scalability_point_parameters(
            config=config,
            number_of_uavs=number_of_uavs,
        )

        grid = (
            f"{point.grid_rows}x"
            f"{point.grid_columns}"
        )

        print(
            f"{number_of_uavs:>4} "
            f"{grid:>8} "
            f"{point.bandwidth_per_uav_hz / 1e6:>15.6f} "
            f"{point.backhaul_rate_per_flow_bps / 1e6:>19.6f} "
            f"{point.sv_capacity_per_job_cycles_s / 1e9:>16.6f} "
            f"{point.rcc_capacity_per_job_cycles_s / 1e9:>16.6f}"
        )

    print_section("OUTPUT CONFIGURATION")

    print(
        f"Output root directory:        "
        f"{config.output_root_directory}"
    )
    print(
        f"Save resolved config:         "
        f"{config.save_resolved_config}"
    )
    print(
        f"Save UAV positions:           "
        f"{config.save_positions}"
    )
    print(
        f"Save per-UAV results:         "
        f"{config.save_per_uav_results}"
    )
    print(
        f"Save repetition summary:      "
        f"{config.save_repetition_summary}"
    )
    print(
        f"Save scenario summary:        "
        f"{config.save_scenario_summary}"
    )
    print(
        f"Generate plots:               "
        f"{config.generate_plots}"
    )
    print(
        f"Overwrite existing:           "
        f"{config.overwrite_existing}"
    )
    print(
        f"Decimal places:               "
        f"{config.decimal_places}"
    )


def print_point_result(
    point: ScalabilityPointResult,
) -> None:
    """Print detailed results for one UAV-count point."""
    parameters = point.parameters
    experiment = point.experiment_result

    number_of_uavs = parameters.number_of_uavs

    print()
    print("=" * SEPARATOR_WIDTH)
    print(f"N = {number_of_uavs} UAVs")
    print("=" * SEPARATOR_WIDTH)

    print(
        f"Grid:                         "
        f"{parameters.grid_rows}x"
        f"{parameters.grid_columns}"
    )
    print(
        f"Spatial realizations:         "
        f"{len(experiment.seed_evaluations)}"
    )
    print(
        f"Bandwidth per UAV:            "
        f"{parameters.bandwidth_per_uav_hz / 1e6:.6f} MHz"
    )
    print(
        f"Backhaul per flow:            "
        f"{parameters.backhaul_rate_per_flow_bps / 1e6:.6f} "
        f"Mbit/s"
    )
    print(
        f"Fog capacity per job:         "
        f"{parameters.sv_capacity_per_job_cycles_s / 1e9:.6f} "
        f"GHz"
    )
    print(
        f"RCC capacity per job:         "
        f"{parameters.rcc_capacity_per_job_cycles_s / 1e9:.6f} "
        f"GHz"
    )

    for summary in experiment.scenario_summaries:
        label = scenario_label(summary.scenario)

        print()
        print(
            f"{summary.scenario} - {label} "
            f"(execution tier: {summary.execution_tier})"
        )
        print("-" * 72)

        print(
            f"  UAVs:                       "
            f"{summary.number_of_uavs}"
        )
        print(
            f"  Repetitions:                "
            f"{summary.number_of_repetitions}"
        )
        print(
            f"  Mean T_max:                 "
            f"{latency_ms(summary.mean_of_max_latency_s):.3f} ms"
        )
        print(
            f"  Median T_max:               "
            f"{latency_ms(summary.median_of_max_latency_s):.3f} ms"
        )
        print(
            f"  Minimum T_max:              "
            f"{latency_ms(summary.minimum_of_max_latency_s):.3f} ms"
        )
        print(
            f"  Maximum T_max:              "
            f"{latency_ms(summary.maximum_of_max_latency_s):.3f} ms"
        )
        print(
            f"  Std T_max:                  "
            f"{latency_ms(summary.std_of_max_latency_s):.3f} ms"
        )
        print(
            f"  95% CI for mean T_max:      "
            f"["
            f"{latency_ms(summary.confidence_interval_95_lower_s):.3f}, "
            f"{latency_ms(summary.confidence_interval_95_upper_s):.3f}"
            f"] ms"
        )
        print(
            f"  Mean of mean UAV latency:   "
            f"{latency_ms(summary.mean_of_mean_latency_s):.3f} ms"
        )
        print(
            f"  Mean of minimum latency:    "
            f"{latency_ms(summary.mean_of_min_latency_s):.3f} ms"
        )
        print(
            f"  Mean latency range:         "
            f"{latency_ms(summary.mean_latency_range_s):.3f} ms"
        )
        print(
            f"  Wins:                       "
            f"{summary.number_of_wins}/"
            f"{summary.number_of_repetitions}"
        )
        print(
            f"  Ties:                       "
            f"{summary.number_of_ties}/"
            f"{summary.number_of_repetitions}"
        )
        print(
            f"  Win rate:                   "
            f"{summary.win_rate * 100.0:.2f}%"
        )


def best_scenario_for_point(
    point: ScalabilityPointResult,
) -> str:
    """Return scenario label(s) with minimum mean T_max."""
    summaries = point.experiment_result.scenario_summaries

    best_latency_s = min(
        summary.mean_of_max_latency_s
        for summary in summaries
    )

    tolerance_s = 1e-9

    winners = [
        scenario_label(summary.scenario)
        for summary in summaries
        if abs(
            summary.mean_of_max_latency_s
            - best_latency_s
        )
        <= tolerance_s
    ]

    return "/".join(winners)


def print_consolidated_results(
    result: ScalabilityExperimentResult,
) -> None:
    """Print the final scalability comparison table."""
    print()
    print("=" * SEPARATOR_WIDTH)
    print("CONSOLIDATED SCALABILITY RESULTS")
    print("=" * SEPARATOR_WIDTH)

    print()
    print(
        f"Spatial realizations:         "
        f"{result.total_spatial_realizations}"
    )
    print(
        f"Per-UAV evaluations:          "
        f"{result.total_per_uav_evaluations}"
    )

    print()
    print(
        "Primary metric: Mean Maximum End-to-End Latency "
        "(Mean T_max)"
    )
    print("Presentation unit: milliseconds")

    print()
    print(
        f"{'N':>4} "
        f"{'All-UAV (ms)':>18} "
        f"{'All-Fog (ms)':>18} "
        f"{'All-RCC (ms)':>18} "
        f"{'Best':>14}"
    )
    print("-" * 80)

    for point in result.point_results:
        summaries = {
            summary.scenario: summary
            for summary
            in point.experiment_result.scenario_summaries
        }

        number_of_uavs = (
            point.parameters.number_of_uavs
        )

        s1_ms = latency_ms(
            summaries["S1"].mean_of_max_latency_s
        )
        s2_ms = latency_ms(
            summaries["S2"].mean_of_max_latency_s
        )
        s3_ms = latency_ms(
            summaries["S3"].mean_of_max_latency_s
        )

        print(
            f"{number_of_uavs:>4} "
            f"{s1_ms:>18.3f} "
            f"{s2_ms:>18.3f} "
            f"{s3_ms:>18.3f} "
            f"{best_scenario_for_point(point):>14}"
        )

    print()
    print("BEST EXECUTION TIER BY UAV COUNT")
    print("-" * 52)

    for point in result.point_results:
        print(
            f"N={point.parameters.number_of_uavs:<2} "
            f"-> {best_scenario_for_point(point)}"
        )


def print_experiment_result(
    result: ScalabilityExperimentResult,
) -> None:
    """Print detailed and consolidated scalability results."""
    print()
    print("=" * SEPARATOR_WIDTH)
    print("SIMULATION COMPLETED")
    print("=" * SEPARATOR_WIDTH)

    print(
        f"UAV counts evaluated:         "
        f"{', '.join(str(n) for n in result.uav_counts)}"
    )
    print(
        f"Spatial realizations:         "
        f"{result.total_spatial_realizations}"
    )
    print(
        f"Per-UAV evaluations:          "
        f"{result.total_per_uav_evaluations}"
    )

    for point in result.point_results:
        print_point_result(point)

    print_consolidated_results(result)


def print_run_artifacts(
    artifacts: RunArtifacts,
) -> None:
    """Print the paths generated for the completed run."""
    print()
    print("=" * SEPARATOR_WIDTH)
    print("EXPERIMENT ARTIFACTS")
    print("=" * SEPARATOR_WIDTH)

    print(
        f"Run directory:                "
        f"{artifacts.run_directory}"
    )
    print(
        f"Experiment parameters:        "
        f"{artifacts.parameters_path}"
    )
    print(
        f"Scalability summary:          "
        f"{artifacts.summary_path}"
    )


def validate_scalability_configuration(
    config: ResolvedConfig,
) -> None:
    """Ensure the runner received the expected scalability protocol."""
    if config.experiment_type != "scalability":
        raise ModelValidationError(
            "simulate_ucc.py requires "
            "experiment.type='scalability'."
        )

    if config.uav_counts != (
        4,
        6,
        8,
        10,
        12,
    ):
        raise ModelValidationError(
            "The configured UAV scalability set must be "
            "(4, 6, 8, 10, 12)."
        )


def main() -> int:
    """Load the configuration and execute the scalability experiment."""
    parser = build_argument_parser()
    arguments = parser.parse_args()

    try:
        config = load_and_resolve_config(
            arguments.config
        )

        validate_scalability_configuration(
            config
        )

        print_resolved_configuration(
            config
        )

        if arguments.validate_only:
            print()
            print("=" * SEPARATOR_WIDTH)
            print("CONFIGURATION VALIDATION: PASS")
            print("=" * SEPARATOR_WIDTH)
            return 0

        result = evaluate_scalability_experiment(
            config
        )

        print_experiment_result(
            result
        )

        artifacts = write_run_artifacts(
            config=config,
            result=result,
        )

        print_run_artifacts(
            artifacts
        )

        if config.generate_plots:
            pdf_path, png_path = (
                generate_scalability_latency_plot(
                    csv_path=artifacts.summary_path,
                    output_directory=artifacts.run_directory,
                )
            )

            print(
                f"Scalability plot PDF:         "
                f"{pdf_path}"
            )
            print(
                f"Scalability plot PNG:         "
                f"{png_path}"
            )

        return 0

    except ConfigurationError as exc:
        print(
            f"Configuration error: {exc}",
            file=sys.stderr,
        )
        return 1

    except ModelValidationError as exc:
        print(
            f"Model validation error: {exc}",
            file=sys.stderr,
        )
        return 2

    except OSError as exc:
        print(
            f"File-system error: {exc}",
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
