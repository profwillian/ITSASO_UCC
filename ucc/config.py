"""Configuration utilities for the UCC maritime simulation."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any


class ConfigurationError(ValueError):
    """Raised when an experimental parameter is invalid."""


def _as_finite_float(value: Real, field_name: str) -> float:
    """Convert a numeric value to float and ensure that it is finite."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigurationError(
            f"{field_name} must be a real number. Received: {value!r}"
        )

    converted = float(value)

    if not math.isfinite(converted):
        raise ConfigurationError(
            f"{field_name} must be finite. Received: {value!r}"
        )

    return converted


def _as_non_negative_float(value: Real, field_name: str) -> float:
    """Convert a numeric value and reject negative values."""
    converted = _as_finite_float(value, field_name)

    if converted < 0.0:
        raise ConfigurationError(
            f"{field_name} must not be negative. Received: {converted}"
        )

    return converted


def mbit_to_bits(value_mbit: Real) -> float:
    """Convert decimal megabits to bits."""
    value = _as_non_negative_float(value_mbit, "value_mbit")
    return value * 1_000_000.0


def mbps_to_bps(value_mbps: Real) -> float:
    """Convert decimal megabits per second to bits per second."""
    value = _as_non_negative_float(value_mbps, "value_mbps")
    return value * 1_000_000.0


def mhz_to_hz(value_mhz: Real) -> float:
    """Convert megahertz to hertz."""
    value = _as_non_negative_float(value_mhz, "value_mhz")
    return value * 1_000_000.0


def ghz_to_cycles_per_second(value_ghz: Real) -> float:
    """Convert GHz to CPU cycles per second."""
    value = _as_non_negative_float(value_ghz, "value_ghz")
    return value * 1_000_000_000.0


def db_to_linear(value_db: Real) -> float:
    """Convert a power ratio expressed in dB to linear scale."""
    value = _as_finite_float(value_db, "value_db")
    converted = math.pow(10.0, value / 10.0)

    if not math.isfinite(converted) or converted <= 0.0:
        raise ConfigurationError(
            f"dB conversion produced an invalid result: {converted}"
        )

    return converted


def dbm_per_hz_to_w_per_hz(value_dbm_hz: Real) -> float:
    """Convert dBm/Hz to W/Hz."""
    value = _as_finite_float(value_dbm_hz, "value_dbm_hz")
    converted = math.pow(10.0, (value - 30.0) / 10.0)

    if not math.isfinite(converted) or converted <= 0.0:
        raise ConfigurationError(
            f"dBm/Hz conversion produced an invalid result: {converted}"
        )

    return converted
def _as_positive_int(value: int, field_name: str) -> int:
    """Validate and return a strictly positive integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(
            f"{field_name} must be an integer. Received: {value!r}"
        )

    if value <= 0:
        raise ConfigurationError(
            f"{field_name} must be greater than zero. Received: {value}"
        )

    return value


def resolve_factorized_grid(
    number_of_uavs: int,
) -> tuple[int, int]:
    """Return the most balanced integer grid for N UAVs."""
    if (
        isinstance(number_of_uavs, bool)
        or not isinstance(number_of_uavs, int)
    ):
        raise ConfigurationError(
            "number_of_uavs must be an integer."
        )

    if number_of_uavs <= 0:
        raise ConfigurationError(
            "number_of_uavs must be greater than zero."
        )

    root = math.isqrt(number_of_uavs)

    for rows in range(root, 0, -1):
        if number_of_uavs % rows == 0:
            columns = number_of_uavs // rows
            return rows, columns

    raise ConfigurationError(
        f"Unable to factorize grid for N={number_of_uavs}."
    )


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    """Validated UCC configuration expressed in SI units."""

    schema_version: str

    experiment_name: str
    experiment_description: str
    experiment_type: str
    scenarios: tuple[str, ...]
    uav_counts: tuple[int, ...]
    seeds: tuple[int, ...]
    paired_geometry: bool
    time_mode: str
    execution_backend: str

    number_of_uavs: int
    search_area_side_m: float
    uav_altitude_m: float
    grid_rows: int
    grid_columns: int
    sv_x_m: float
    sv_y_m: float

    chunks_per_job: int
    chunk_size_bits: float
    input_payload_bits: float
    output_input_ratio: float
    output_payload_bits: float
    compute_intensity_cycles_per_bit: float
    workload_cycles: float

    uav_capacity_cycles_s: float
    sv_capacity_cycles_s: float
    rcc_capacity_cycles_s: float
    sv_capacity_per_job_cycles_s: float
    rcc_capacity_per_job_cycles_s: float

    reference_distance_m: float
    total_bandwidth_hz: float
    bandwidth_per_uav_hz: float
    uav_transmit_power_w: float
    reference_channel_gain_linear: float
    noise_psd_w_hz: float

    backhaul_capacity_bps: float
    backhaul_rate_per_flow_bps: float
    backhaul_fixed_delay_s: float

    output_root_directory: str
    save_resolved_config: bool
    save_positions: bool
    save_per_uav_results: bool
    save_repetition_summary: bool
    save_scenario_summary: bool
    generate_plots: bool
    overwrite_existing: bool
    decimal_places: int


def _as_positive_float(
    value: Real,
    field_name: str,
) -> float:
    """Convert a numeric value and require it to be positive."""
    converted = _as_finite_float(value, field_name)

    if converted <= 0.0:
        raise ConfigurationError(
            f"{field_name} must be greater than zero. "
            f"Received: {converted}"
        )

    return converted


def _require_mapping(
    value: object,
    field_name: str,
) -> Mapping[str, Any]:
    """Require a JSON object."""
    if not isinstance(value, Mapping):
        raise ConfigurationError(
            f"{field_name} must be a JSON object. "
            f"Received: {type(value).__name__}"
        )

    return value


def _require_string(
    value: object,
    field_name: str,
) -> str:
    """Require a non-empty string."""
    if not isinstance(value, str):
        raise ConfigurationError(
            f"{field_name} must be a string. Received: {value!r}"
        )

    normalized = value.strip()

    if not normalized:
        raise ConfigurationError(
            f"{field_name} must not be empty."
        )

    return normalized


def _require_bool(
    value: object,
    field_name: str,
) -> bool:
    """Require a boolean value."""
    if not isinstance(value, bool):
        raise ConfigurationError(
            f"{field_name} must be boolean. Received: {value!r}"
        )

    return value


def _require_exact_keys(
    mapping: Mapping[str, Any],
    expected_keys: set[str],
    field_name: str,
) -> None:
    """Reject missing and unknown configuration fields."""
    observed_keys = set(mapping)

    missing = sorted(expected_keys - observed_keys)
    unknown = sorted(observed_keys - expected_keys)

    if missing:
        raise ConfigurationError(
            f"{field_name} is missing required fields: "
            f"{', '.join(missing)}"
        )

    if unknown:
        raise ConfigurationError(
            f"{field_name} contains unknown fields: "
            f"{', '.join(unknown)}"
        )


def _require_choice(
    value: object,
    allowed: set[str],
    field_name: str,
) -> str:
    """Require a string belonging to a fixed set."""
    normalized = _require_string(value, field_name)

    if normalized not in allowed:
        raise ConfigurationError(
            f"{field_name} must be one of {sorted(allowed)}. "
            f"Received: {normalized!r}"
        )

    return normalized


def _require_supported_uav_count(value: object) -> int:
    """Validate the UAV counts defined in the experimental protocol."""
    number_of_uavs = _as_positive_int(
        value,
        "topology.number_of_uavs",
    )

    supported = {4, 8, 12, 16}

    if number_of_uavs not in supported:
        raise ConfigurationError(
            "topology.number_of_uavs must be one of "
            f"{sorted(supported)}. Received: {number_of_uavs}"
        )

    return number_of_uavs


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    config_path = Path(path)

    if not config_path.exists():
        raise ConfigurationError(
            f"Configuration file does not exist: {config_path}"
        )

    if not config_path.is_file():
        raise ConfigurationError(
            f"Configuration path is not a file: {config_path}"
        )

    try:
        with config_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            raw = json.load(file)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"Could not parse {config_path}: "
            f"line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    except OSError as exc:
        raise ConfigurationError(
            f"Could not read {config_path}: {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise ConfigurationError(
            "The configuration root must be a JSON object."
        )

    return raw



def resolve_seeds(
    seed_start: int,
    number_of_repetitions: int,
) -> tuple[int, ...]:
    """Generate the ordered sequence of experimental seeds."""
    if (
        isinstance(seed_start, bool)
        or not isinstance(seed_start, int)
        or seed_start < 0
    ):
        raise ConfigurationError(
            "experiment.seed_start must be a non-negative integer. "
            f"Received: {seed_start!r}"
        )

    repetitions = _as_positive_int(
        number_of_repetitions,
        "experiment.number_of_repetitions",
    )

    return tuple(
        range(
            seed_start,
            seed_start + repetitions,
        )
    )


def _resolve_scenarios(
    value: object,
) -> tuple[str, ...]:
    """Validate the ordered homogeneous execution scenarios."""
    if not isinstance(value, list):
        raise ConfigurationError(
            "experiment.scenarios must be a JSON array."
        )

    scenarios: list[str] = []

    for index, scenario in enumerate(value):
        normalized = _require_choice(
            scenario,
            {"S1", "S2", "S3"},
            f"experiment.scenarios[{index}]",
        )

        scenarios.append(normalized)

    resolved = tuple(scenarios)

    if resolved != ("S1", "S2", "S3"):
        raise ConfigurationError(
            "experiment.scenarios must contain exactly "
            '["S1", "S2", "S3"] in this order.'
        )

    return resolved


def _resolve_uav_counts(
    value: object,
) -> tuple[int, ...]:
    """Validate the ordered UAV counts used in scalability evaluation."""
    if not isinstance(value, list):
        raise ConfigurationError(
            "experiment.uav_counts must be a JSON array."
        )

    if not value:
        raise ConfigurationError(
            "experiment.uav_counts must not be empty."
        )

    resolved: list[int] = []

    for index, number_of_uavs in enumerate(value):
        if (
            isinstance(number_of_uavs, bool)
            or not isinstance(number_of_uavs, int)
        ):
            raise ConfigurationError(
                f"experiment.uav_counts[{index}] "
                "must be an integer."
            )

        if number_of_uavs <= 0:
            raise ConfigurationError(
                f"experiment.uav_counts[{index}] "
                "must be greater than zero."
            )

        resolved.append(number_of_uavs)

    if len(set(resolved)) != len(resolved):
        raise ConfigurationError(
            "experiment.uav_counts must not contain duplicates."
        )

    if resolved != sorted(resolved):
        raise ConfigurationError(
            "experiment.uav_counts must be in ascending order."
        )

    expected = (4, 6, 8, 10, 12)

    result = tuple(resolved)

    if result != expected:
        raise ConfigurationError(
            "experiment.uav_counts must be exactly "
            "[4, 6, 8, 10, 12]."
        )

    return result


def load_and_resolve_config(
    path: str | Path,
) -> ResolvedConfig:
    """Load, validate, convert, and resolve a UCC configuration."""
    raw = read_json(path)

    _require_exact_keys(
        raw,
        {
            "schema_version",
            "experiment",
            "topology",
            "workload",
            "computation",
            "access_link",
            "backhaul",
            "output",
        },
        "configuration",
    )

    schema_version = _require_choice(
        raw["schema_version"],
        {"1.0"},
        "schema_version",
    )

    experiment = _require_mapping(
        raw["experiment"],
        "experiment",
    )

    topology = _require_mapping(
        raw["topology"],
        "topology",
    )

    workload = _require_mapping(
        raw["workload"],
        "workload",
    )

    computation = _require_mapping(
        raw["computation"],
        "computation",
    )

    access_link = _require_mapping(
        raw["access_link"],
        "access_link",
    )

    backhaul = _require_mapping(
        raw["backhaul"],
        "backhaul",
    )

    output = _require_mapping(
        raw["output"],
        "output",
    )

    _require_exact_keys(
    experiment,
    {
        "name",
        "description",
        "type",
        "scenarios",
        "uav_counts",
        "number_of_repetitions",
        "seed_start",
        "paired_geometry",
        "time_mode",
        "execution_backend",
    },
    "experiment",
    )

    _require_exact_keys(
        topology,
        {
            "number_of_uavs",
            "search_area_side_m",
            "uav_altitude_m",
            "subregion_layout",
            "surface_vessel",
        },
        "topology",
    )

    _require_exact_keys(
        workload,
        {
            "chunks_per_job",
            "chunk_size_mbit",
            "output_input_ratio",
            "compute_intensity_cycles_per_bit",
        },
        "workload",
    )

    _require_exact_keys(
        computation,
        {
            "uav_capacity_ghz",
            "sv_capacity_ghz",
            "rcc_capacity_ghz",
            "remote_capacity_sharing",
        },
        "computation",
    )

    _require_exact_keys(
        access_link,
        {
            "reference_distance_m",
            "total_bandwidth_mhz",
            "uav_transmit_power_w",
            "reference_channel_gain_db",
            "noise_psd_dbm_hz",
            "bandwidth_allocation",
            "channel_model",
        },
        "access_link",
    )

    _require_exact_keys(
        backhaul,
        {
            "aggregate_capacity_mbps",
            "fixed_delay_s",
            "capacity_sharing",
        },
        "backhaul",
    )

    _require_exact_keys(
        output,
        {
            "root_directory",
            "save_resolved_config",
            "save_positions",
            "save_per_uav_results",
            "save_repetition_summary",
            "save_scenario_summary",
            "generate_plots",
            "overwrite_existing",
            "decimal_places",
        },
        "output",
    )

    subregion_layout = _require_mapping(
        topology["subregion_layout"],
        "topology.subregion_layout",
    )

    surface_vessel = _require_mapping(
        topology["surface_vessel"],
        "topology.surface_vessel",
    )

    _require_exact_keys(
        subregion_layout,
        {
            "strategy",
            "uav_assignment",
            "position_distribution",
        },
        "topology.subregion_layout",
    )

    _require_exact_keys(
        surface_vessel,
        {"position_mode"},
        "topology.surface_vessel",
    )

    experiment_name = _require_string(
        experiment["name"],
        "experiment.name",
    )

    if not re.fullmatch(
        r"[A-Za-z0-9_-]+",
        experiment_name,
    ):
        raise ConfigurationError(
            "experiment.name may contain only letters, numbers, "
            "underscores, and hyphens."
        )

    experiment_description = _require_string(
        experiment["description"],
        "experiment.description",
    )

    experiment_type = _require_choice(
    experiment["type"],
    {"baseline", "scalability"},
    "experiment.type",
    )

    scenarios = _resolve_scenarios(
        experiment["scenarios"]
    )

    uav_counts = _resolve_uav_counts(
    experiment["uav_counts"]
    )

    number_of_repetitions = _as_positive_int(
        experiment["number_of_repetitions"],
        "experiment.number_of_repetitions",
    )

    seed_start = experiment["seed_start"]

    seeds = resolve_seeds(
        seed_start=seed_start,
        number_of_repetitions=number_of_repetitions,
    )

    paired_geometry = _require_bool(
        experiment["paired_geometry"],
        "experiment.paired_geometry",
    )

    if not paired_geometry:
        raise ConfigurationError(
            "experiment.paired_geometry must be true."
        )

    time_mode = _require_choice(
        experiment["time_mode"],
        {"virtual"},
        "experiment.time_mode",
    )

    execution_backend = _require_choice(
        experiment["execution_backend"],
        {"equation_based"},
        "experiment.execution_backend",
    )

    number_of_uavs = _require_supported_uav_count(
        topology["number_of_uavs"]
    )

    search_area_side_m = _as_positive_float(
        topology["search_area_side_m"],
        "topology.search_area_side_m",
    )

    uav_altitude_m = _as_positive_float(
        topology["uav_altitude_m"],
        "topology.uav_altitude_m",
    )

    _require_choice(
        subregion_layout["strategy"],
        {"factorized_equal_area"},
        "topology.subregion_layout.strategy",
    )

    _require_choice(
        subregion_layout["uav_assignment"],
        {"row_major"},
        "topology.subregion_layout.uav_assignment",
    )

    _require_choice(
        subregion_layout["position_distribution"],
        {"uniform"},
        "topology.subregion_layout.position_distribution",
    )

    _require_choice(
        surface_vessel["position_mode"],
        {"center"},
        "topology.surface_vessel.position_mode",
    )

    grid_rows, grid_columns = resolve_factorized_grid(
        number_of_uavs
    )

    sv_x_m = search_area_side_m / 2.0
    sv_y_m = search_area_side_m / 2.0

    chunks_per_job = _as_positive_int(
        workload["chunks_per_job"],
        "workload.chunks_per_job",
    )

    chunk_size_mbit = _as_positive_float(
        workload["chunk_size_mbit"],
        "workload.chunk_size_mbit",
    )

    chunk_size_bits = mbit_to_bits(
        chunk_size_mbit
    )

    output_input_ratio = _as_positive_float(
        workload["output_input_ratio"],
        "workload.output_input_ratio",
    )

    if output_input_ratio > 1.0:
        raise ConfigurationError(
            "workload.output_input_ratio must satisfy "
            "0 < value <= 1."
        )

    compute_intensity = _as_positive_float(
        workload["compute_intensity_cycles_per_bit"],
        "workload.compute_intensity_cycles_per_bit",
    )

    input_payload_bits = (
        chunks_per_job * chunk_size_bits
    )

    output_payload_bits = (
        output_input_ratio * input_payload_bits
    )

    workload_cycles = (
        input_payload_bits * compute_intensity
    )

    uav_capacity_cycles_s = ghz_to_cycles_per_second(
        _as_positive_float(
            computation["uav_capacity_ghz"],
            "computation.uav_capacity_ghz",
        )
    )

    sv_capacity_cycles_s = ghz_to_cycles_per_second(
        _as_positive_float(
            computation["sv_capacity_ghz"],
            "computation.sv_capacity_ghz",
        )
    )

    rcc_capacity_cycles_s = ghz_to_cycles_per_second(
        _as_positive_float(
            computation["rcc_capacity_ghz"],
            "computation.rcc_capacity_ghz",
        )
    )

    _require_choice(
        computation["remote_capacity_sharing"],
        {"static_equal"},
        "computation.remote_capacity_sharing",
    )

    sv_capacity_per_job_cycles_s = (
        sv_capacity_cycles_s / number_of_uavs
    )

    rcc_capacity_per_job_cycles_s = (
        rcc_capacity_cycles_s / number_of_uavs
    )

    reference_distance_m = _as_positive_float(
        access_link["reference_distance_m"],
        "access_link.reference_distance_m",
    )

    total_bandwidth_hz = mhz_to_hz(
        _as_positive_float(
            access_link["total_bandwidth_mhz"],
            "access_link.total_bandwidth_mhz",
        )
    )

    bandwidth_per_uav_hz = (
        total_bandwidth_hz / number_of_uavs
    )

    uav_transmit_power_w = _as_positive_float(
        access_link["uav_transmit_power_w"],
        "access_link.uav_transmit_power_w",
    )

    reference_channel_gain_linear = db_to_linear(
        access_link["reference_channel_gain_db"]
    )

    noise_psd_w_hz = dbm_per_hz_to_w_per_hz(
        access_link["noise_psd_dbm_hz"]
    )

    _require_choice(
        access_link["bandwidth_allocation"],
        {"static_equal"},
        "access_link.bandwidth_allocation",
    )

    _require_choice(
        access_link["channel_model"],
        {"free_space_path_loss"},
        "access_link.channel_model",
    )

    backhaul_capacity_bps = mbps_to_bps(
        _as_positive_float(
            backhaul["aggregate_capacity_mbps"],
            "backhaul.aggregate_capacity_mbps",
        )
    )

    backhaul_rate_per_flow_bps = (
        backhaul_capacity_bps / number_of_uavs
    )

    backhaul_fixed_delay_s = _as_non_negative_float(
        backhaul["fixed_delay_s"],
        "backhaul.fixed_delay_s",
    )

    _require_choice(
        backhaul["capacity_sharing"],
        {"static_equal"},
        "backhaul.capacity_sharing",
    )

    output_root_directory = _require_string(
        output["root_directory"],
        "output.root_directory",
    )

    save_resolved_config = _require_bool(
        output["save_resolved_config"],
        "output.save_resolved_config",
    )

    save_positions = _require_bool(
        output["save_positions"],
        "output.save_positions",
    )

    save_per_uav_results = _require_bool(
        output["save_per_uav_results"],
        "output.save_per_uav_results",
    )

    save_repetition_summary = _require_bool(
        output["save_repetition_summary"],
        "output.save_repetition_summary",
    )

    save_scenario_summary = _require_bool(
        output["save_scenario_summary"],
        "output.save_scenario_summary",
    )

    generate_plots = _require_bool(
        output["generate_plots"],
        "output.generate_plots",
    )

    overwrite_existing = _require_bool(
        output["overwrite_existing"],
        "output.overwrite_existing",
    )

    decimal_places = _as_positive_int(
        output["decimal_places"],
        "output.decimal_places",
    )

    derived_values = (
        input_payload_bits,
        output_payload_bits,
        workload_cycles,
        sv_capacity_per_job_cycles_s,
        rcc_capacity_per_job_cycles_s,
        bandwidth_per_uav_hz,
        backhaul_rate_per_flow_bps,
    )

    if not all(
        math.isfinite(value) and value > 0.0
        for value in derived_values
    ):
        raise ConfigurationError(
            "Configuration derivation produced an invalid value."
        )

    return ResolvedConfig(
	schema_version=schema_version,
	experiment_name=experiment_name,
	experiment_description=experiment_description,
	experiment_type=experiment_type,
	scenarios=scenarios,
    	uav_counts=uav_counts,
    	seeds=seeds,
        paired_geometry=paired_geometry,
        time_mode=time_mode,
        execution_backend=execution_backend,
        number_of_uavs=number_of_uavs,
        search_area_side_m=search_area_side_m,
        uav_altitude_m=uav_altitude_m,
        grid_rows=grid_rows,
        grid_columns=grid_columns,
        sv_x_m=sv_x_m,
        sv_y_m=sv_y_m,
        chunks_per_job=chunks_per_job,
        chunk_size_bits=chunk_size_bits,
        input_payload_bits=input_payload_bits,
        output_input_ratio=output_input_ratio,
        output_payload_bits=output_payload_bits,
        compute_intensity_cycles_per_bit=compute_intensity,
        workload_cycles=workload_cycles,
        uav_capacity_cycles_s=uav_capacity_cycles_s,
        sv_capacity_cycles_s=sv_capacity_cycles_s,
        rcc_capacity_cycles_s=rcc_capacity_cycles_s,
        sv_capacity_per_job_cycles_s=(
            sv_capacity_per_job_cycles_s
        ),
        rcc_capacity_per_job_cycles_s=(
            rcc_capacity_per_job_cycles_s
        ),
        reference_distance_m=reference_distance_m,
        total_bandwidth_hz=total_bandwidth_hz,
        bandwidth_per_uav_hz=bandwidth_per_uav_hz,
        uav_transmit_power_w=uav_transmit_power_w,
        reference_channel_gain_linear=(
            reference_channel_gain_linear
        ),
        noise_psd_w_hz=noise_psd_w_hz,
        backhaul_capacity_bps=backhaul_capacity_bps,
        backhaul_rate_per_flow_bps=(
            backhaul_rate_per_flow_bps
        ),
        backhaul_fixed_delay_s=backhaul_fixed_delay_s,
        output_root_directory=output_root_directory,
        save_resolved_config=save_resolved_config,
        save_positions=save_positions,
        save_per_uav_results=save_per_uav_results,
        save_repetition_summary=save_repetition_summary,
        save_scenario_summary=save_scenario_summary,
        generate_plots=generate_plots,
        overwrite_existing=overwrite_existing,
        decimal_places=decimal_places,
    )
