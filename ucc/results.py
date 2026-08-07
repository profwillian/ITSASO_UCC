"""Persistence utilities for UCC scalability experiment results."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ucc.config import ResolvedConfig
from ucc.experiment import (
    ScalabilityExperimentResult,
    ScalabilityPointResult,
    build_scalability_point_parameters,
)
from ucc.model import ModelValidationError


SCENARIO_LABELS = {
    "S1": "All-UAV",
    "S2": "All-Fog",
    "S3": "All-RCC",
}

@dataclass(frozen=True, slots=True)
class RunArtifacts:
    """Paths produced by one complete scalability run."""

    run_directory: Path
    parameters_path: Path
    summary_path: Path


def scenario_label(scenario: str) -> str:
    """Return the human-readable scenario label."""
    return SCENARIO_LABELS.get(
        scenario,
        scenario,
    )


def latency_ms(value_s: float) -> float:
    """Convert seconds to milliseconds."""
    return value_s * 1000.0

def configuration_fingerprint(
    config: ResolvedConfig,
) -> str:
    """Return a stable SHA-256 fingerprint of resolved parameters."""
    document = build_parameters_document(config)

    canonical = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
    )

    digest = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()

    return digest

def build_parameters_document(
    config: ResolvedConfig,
) -> dict[str, Any]:
    """
    Build an auditable representation of the resolved experiment
    configuration and all N-dependent derived resources.
    """
    derived_points: list[dict[str, Any]] = []

    for number_of_uavs in config.uav_counts:
        point = build_scalability_point_parameters(
            config=config,
            number_of_uavs=number_of_uavs,
        )

        derived_points.append(
            {
                "number_of_uavs": number_of_uavs,
                "grid_rows": point.grid_rows,
                "grid_columns": point.grid_columns,
                "bandwidth_per_uav_hz": (
                    point.bandwidth_per_uav_hz
                ),
                "backhaul_rate_per_flow_bps": (
                    point.backhaul_rate_per_flow_bps
                ),
                "sv_capacity_per_job_cycles_s": (
                    point.sv_capacity_per_job_cycles_s
                ),
                "rcc_capacity_per_job_cycles_s": (
                    point.rcc_capacity_per_job_cycles_s
                ),
            }
        )

    return {
        "schema_version": config.schema_version,
        "experiment": {
            "name": config.experiment_name,
            "description": config.experiment_description,
            "type": config.experiment_type,
            "scenarios": list(config.scenarios),
            "scenario_labels": {
                scenario: scenario_label(scenario)
                for scenario in config.scenarios
            },
            "uav_counts": list(config.uav_counts),
            "number_of_repetitions": len(config.seeds),
            "seeds": list(config.seeds),
            "paired_geometry": config.paired_geometry,
            "time_mode": config.time_mode,
            "execution_backend": config.execution_backend,
        },
        "topology": {
            "search_area_side_m": config.search_area_side_m,
            "uav_altitude_m": config.uav_altitude_m,
            "sv_x_m": config.sv_x_m,
            "sv_y_m": config.sv_y_m,
        },
        "workload": {
            "chunks_per_job": config.chunks_per_job,
            "chunk_size_bits": config.chunk_size_bits,
            "input_payload_bits": config.input_payload_bits,
            "output_input_ratio": config.output_input_ratio,
            "output_payload_bits": config.output_payload_bits,
            "compute_intensity_cycles_per_bit": (
                config.compute_intensity_cycles_per_bit
            ),
            "workload_cycles": config.workload_cycles,
        },
        "computation": {
            "uav_capacity_cycles_s": (
                config.uav_capacity_cycles_s
            ),
            "sv_capacity_cycles_s": (
                config.sv_capacity_cycles_s
            ),
            "rcc_capacity_cycles_s": (
                config.rcc_capacity_cycles_s
            ),
        },
        "access_link": {
            "reference_distance_m": (
                config.reference_distance_m
            ),
            "total_bandwidth_hz": (
                config.total_bandwidth_hz
            ),
            "uav_transmit_power_w": (
                config.uav_transmit_power_w
            ),
            "reference_channel_gain_linear": (
                config.reference_channel_gain_linear
            ),
            "noise_psd_w_hz": (
                config.noise_psd_w_hz
            ),
        },
        "backhaul": {
            "aggregate_capacity_bps": (
                config.backhaul_capacity_bps
            ),
            "fixed_delay_s": (
                config.backhaul_fixed_delay_s
            ),
        },
        "derived_by_uav_count": derived_points,
    }


def create_run_directory(
    config: ResolvedConfig,
) -> Path:
    """Create a unique non-overwriting directory for one run."""
    root = Path(
        config.output_root_directory
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    fingerprint = configuration_fingerprint(
        config
    )[:8]

    run_name = (
        f"{config.experiment_name}_"
        f"{timestamp}_"
        f"{fingerprint}"
    )

    run_directory = root / run_name

    run_directory.mkdir(
        parents=False,
        exist_ok=False,
    )

    return run_directory

def write_experiment_parameters(
    config: ResolvedConfig,
    output_directory: str | Path,
) -> Path:
    """Write the resolved experiment parameters as JSON."""
    output_path = Path(output_directory)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = (
        output_path
        / "experiment_parameters.json"
    )

    if destination.exists() and not config.overwrite_existing:
        raise FileExistsError(
            f"Result file already exists: {destination}"
        )

    document = build_parameters_document(config)

    destination.write_text(
        json.dumps(
            document,
            indent=2,
            sort_keys=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return destination


def scalability_summary_rows(
    result: ScalabilityExperimentResult,
) -> list[dict[str, Any]]:
    """Build the 15 consolidated scalability result rows."""
    rows: list[dict[str, Any]] = []

    for point in result.point_results:
        number_of_uavs = (
            point.parameters.number_of_uavs
        )

        for summary in (
            point.experiment_result.scenario_summaries
        ):
            rows.append(
                {
                    "number_of_uavs": number_of_uavs,
                    "scenario": summary.scenario,
                    "scenario_label": scenario_label(
                        summary.scenario
                    ),
                    "execution_tier": (
                        summary.execution_tier
                    ),
                    "number_of_repetitions": (
                        summary.number_of_repetitions
                    ),
                    "mean_max_latency_ms": latency_ms(
                        summary.mean_of_max_latency_s
                    ),
                    "median_max_latency_ms": latency_ms(
                        summary.median_of_max_latency_s
                    ),
                    "minimum_max_latency_ms": latency_ms(
                        summary.minimum_of_max_latency_s
                    ),
                    "maximum_max_latency_ms": latency_ms(
                        summary.maximum_of_max_latency_s
                    ),
                    "std_max_latency_ms": latency_ms(
                        summary.std_of_max_latency_s
                    ),
                    "ci95_lower_ms": latency_ms(
                        summary.confidence_interval_95_lower_s
                    ),
                    "ci95_upper_ms": latency_ms(
                        summary.confidence_interval_95_upper_s
                    ),
                    "mean_of_mean_latency_ms": latency_ms(
                        summary.mean_of_mean_latency_s
                    ),
                    "mean_of_min_latency_ms": latency_ms(
                        summary.mean_of_min_latency_s
                    ),
                    "mean_latency_range_ms": latency_ms(
                        summary.mean_latency_range_s
                    ),
                    "number_of_wins": (
                        summary.number_of_wins
                    ),
                    "number_of_ties": (
                        summary.number_of_ties
                    ),
                    "win_rate": summary.win_rate,
                }
            )

    expected_rows = (
        len(result.uav_counts) * 3
    )

    if len(rows) != expected_rows:
        raise ModelValidationError(
            "Unexpected number of scalability summary rows."
        )

    return rows


def write_scalability_summary(
    config: ResolvedConfig,
    result: ScalabilityExperimentResult,
    output_directory: str | Path,
) -> Path:
    """Write the consolidated scalability results as CSV."""
    output_path = Path(output_directory)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = (
        output_path
        / "scalability_summary.csv"
    )

    if destination.exists() and not config.overwrite_existing:
        raise FileExistsError(
            f"Result file already exists: {destination}"
        )

    rows = scalability_summary_rows(result)

    fieldnames = [
        "number_of_uavs",
        "scenario",
        "scenario_label",
        "execution_tier",
        "number_of_repetitions",
        "mean_max_latency_ms",
        "median_max_latency_ms",
        "minimum_max_latency_ms",
        "maximum_max_latency_ms",
        "std_max_latency_ms",
        "ci95_lower_ms",
        "ci95_upper_ms",
        "mean_of_mean_latency_ms",
        "mean_of_min_latency_ms",
        "mean_latency_range_ms",
        "number_of_wins",
        "number_of_ties",
        "win_rate",
    ]

    with destination.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            formatted = dict(row)

            for key, value in formatted.items():
                if isinstance(value, float):
                    formatted[key] = (
                        f"{value:.{config.decimal_places}f}"
                    )

            writer.writerow(formatted)

    return destination


def write_scalability_artifacts(
    config: ResolvedConfig,
    result: ScalabilityExperimentResult,
    output_directory: str | Path,
) -> tuple[Path, Path]:
    """Write the core scientific artifacts for one run."""
    parameters_path = write_experiment_parameters(
        config=config,
        output_directory=output_directory,
    )

    summary_path = write_scalability_summary(
        config=config,
        result=result,
        output_directory=output_directory,
    )

    return (
        parameters_path,
        summary_path,
    )


def write_run_artifacts(
    config: ResolvedConfig,
    result: ScalabilityExperimentResult,
) -> RunArtifacts:
    """Persist the core artifacts of one complete experiment run."""
    run_directory = create_run_directory(
        config
    )

    parameters_path, summary_path = (
        write_scalability_artifacts(
            config=config,
            result=result,
            output_directory=run_directory,
        )
    )

    return RunArtifacts(
        run_directory=run_directory,
        parameters_path=parameters_path,
        summary_path=summary_path,
    )
