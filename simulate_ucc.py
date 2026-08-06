"""Command-line entry point for the UCC maritime simulation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ucc.config import (
    ConfigurationError,
    ResolvedConfig,
    load_and_resolve_config,
)
from ucc.experiment import (
    ExperimentParameters,
    ExperimentResult,
    evaluate_experiment,
)
from ucc.model import ModelValidationError


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the equation-based UCC maritime inference "
            "offloading experiment."
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
        help="Validate and resolve the configuration without running.",
    )

    return parser


def build_experiment_parameters(
    config: ResolvedConfig,
) -> ExperimentParameters:
    """Convert a resolved configuration into experiment parameters."""
    return ExperimentParameters(
        number_of_uavs=config.number_of_uavs,
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
        backhaul_fixed_delay_s=config.backhaul_fixed_delay_s,
        input_payload_bits=config.input_payload_bits,
        output_payload_bits=config.output_payload_bits,
        workload_cycles=config.workload_cycles,
        uav_capacity_cycles_s=config.uav_capacity_cycles_s,
        sv_capacity_cycles_s=config.sv_capacity_cycles_s,
        rcc_capacity_cycles_s=config.rcc_capacity_cycles_s,
        scenarios=config.scenarios,
    )


def count_per_uav_evaluations(
    result: ExperimentResult,
) -> int:
    """Count detailed UAV-scenario evaluations."""
    return sum(
        len(scenario_evaluation.results)
        for seed_evaluation in result.seed_evaluations
        for scenario_evaluation
        in seed_evaluation.scenario_evaluations
    )


def print_resolved_configuration(
    config: ResolvedConfig,
) -> None:
    """Print the main resolved experimental parameters."""
    print("UCC Maritime Experiment")
    print()
    print(f"Experiment:         {config.experiment_name}")
    print(f"Execution backend:  {config.execution_backend}")
    print(f"Time mode:          {config.time_mode}")
    print(f"Scenarios:          {', '.join(config.scenarios)}")
    print()
    print(f"UAVs:               {config.number_of_uavs}")
    print(
        f"Grid:               "
        f"{config.grid_rows}x{config.grid_columns}"
    )
    print(
        f"Search area:        "
        f"{config.search_area_side_m:.3f} x "
        f"{config.search_area_side_m:.3f} m"
    )
    print(f"UAV altitude:       {config.uav_altitude_m:.3f} m")
    print(
        f"SV position:        "
        f"({config.sv_x_m:.3f}, {config.sv_y_m:.3f}) m"
    )
    print()
    print(f"Repetitions:        {len(config.seeds)}")
    print(
        f"Seeds:              "
        f"{config.seeds[0]}..{config.seeds[-1]}"
    )
    print()
    print(
        f"Input payload:      "
        f"{config.input_payload_bits / 1_000_000:.3f} Mbit"
    )
    print(
        f"Output payload:     "
        f"{config.output_payload_bits / 1_000_000:.3f} Mbit"
    )
    print(
        f"Workload:           "
        f"{config.workload_cycles / 1_000_000_000:.3f} Gcycles"
    )
    print()
    print(
        f"Bandwidth per UAV:  "
        f"{config.bandwidth_per_uav_hz / 1_000_000:.3f} MHz"
    )
    print(
        f"Backhaul per flow:  "
        f"{config.backhaul_rate_per_flow_bps / 1_000_000:.3f} "
        f"Mbit/s"
    )


def print_experiment_result(
    result: ExperimentResult,
) -> None:
    """Print aggregate latency results."""
    print()
    print("Experiment completed")
    print()
    print(f"Spatial realizations:  {len(result.seed_evaluations)}")
    print(
        f"Per-UAV evaluations:   "
        f"{count_per_uav_evaluations(result)}"
    )
    print()

    for summary in result.scenario_summaries:
        print(
            f"{summary.scenario} ({summary.execution_tier})"
        )
        print(
            f"  Mean T_max:     "
            f"{summary.mean_of_max_latency_s:.9f} s"
        )
        print(
            f"  Median T_max:   "
            f"{summary.median_of_max_latency_s:.9f} s"
        )
        print(
            f"  Minimum T_max:  "
            f"{summary.minimum_of_max_latency_s:.9f} s"
        )
        print(
            f"  Maximum T_max:  "
            f"{summary.maximum_of_max_latency_s:.9f} s"
        )
        print(
            f"  Std T_max:      "
            f"{summary.std_of_max_latency_s:.9f} s"
        )
        print(
            f"  CI 95%:         "
            f"[{summary.confidence_interval_95_lower_s:.9f}, "
            f"{summary.confidence_interval_95_upper_s:.9f}] s"
        )
        print(
            f"  Wins:           "
            f"{summary.number_of_wins}/"
            f"{summary.number_of_repetitions}"
        )
        print(
            f"  Ties:           "
            f"{summary.number_of_ties}/"
            f"{summary.number_of_repetitions}"
        )
        print()


def main() -> int:
    """Load the configuration and execute the experiment."""
    parser = build_argument_parser()
    arguments = parser.parse_args()

    try:
        config = load_and_resolve_config(arguments.config)
        print_resolved_configuration(config)

        if arguments.validate_only:
            print()
            print("Configuration validation: PASS")
            return 0

        parameters = build_experiment_parameters(config)

        result = evaluate_experiment(
            seeds=config.seeds,
            parameters=parameters,
        )

        print_experiment_result(result)
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
