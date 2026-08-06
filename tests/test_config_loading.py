import json
from pathlib import Path

import pytest

from ucc.config import (
    ConfigurationError,
    load_and_resolve_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = PROJECT_ROOT / "cnf" / "ucc_config.json"


def read_baseline_raw() -> dict:
    with BASELINE_CONFIG.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def write_config(
    tmp_path: Path,
    raw: dict,
) -> Path:
    path = tmp_path / "config.json"

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            raw,
            file,
            indent=2,
        )

    return path


def test_load_baseline_configuration() -> None:
    config = load_and_resolve_config(
        BASELINE_CONFIG
    )

    assert config.schema_version == "1.0"
    assert config.experiment_name == "ucc_baseline"
    assert config.scenarios == ("S1", "S2", "S3")
    assert config.seeds == tuple(range(1, 51))

    assert config.number_of_uavs == 8
    assert config.grid_rows == 2
    assert config.grid_columns == 4

    assert config.sv_x_m == 500.0
    assert config.sv_y_m == 500.0

    assert config.chunk_size_bits == 2_000_000.0
    assert config.input_payload_bits == 6_000_000.0
    assert config.output_payload_bits == 600_000.0
    assert config.workload_cycles == 6_000_000_000.0

    assert config.uav_capacity_cycles_s == 200_000_000.0
    assert config.sv_capacity_cycles_s == 2_000_000_000.0
    assert config.rcc_capacity_cycles_s == 500_000_000_000.0

    assert config.sv_capacity_per_job_cycles_s == 250_000_000.0
    assert (
        config.rcc_capacity_per_job_cycles_s
        == 62_500_000_000.0
    )

    assert config.total_bandwidth_hz == 10_000_000.0
    assert config.bandwidth_per_uav_hz == 1_250_000.0

    assert config.backhaul_capacity_bps == 5_000_000.0
    assert config.backhaul_rate_per_flow_bps == 625_000.0

    assert config.backhaul_fixed_delay_s == 0.10


def test_unknown_root_field_is_rejected(
    tmp_path: Path,
) -> None:
    raw = read_baseline_raw()
    raw["unexpected"] = 123

    path = write_config(tmp_path, raw)

    with pytest.raises(
        ConfigurationError,
        match="unknown fields",
    ):
        load_and_resolve_config(path)


def test_unknown_nested_field_is_rejected(
    tmp_path: Path,
) -> None:
    raw = read_baseline_raw()
    raw["workload"]["unknown"] = 123

    path = write_config(tmp_path, raw)

    with pytest.raises(
        ConfigurationError,
        match="unknown fields",
    ):
        load_and_resolve_config(path)


def test_missing_field_is_rejected(
    tmp_path: Path,
) -> None:
    raw = read_baseline_raw()
    del raw["backhaul"]["aggregate_capacity_mbps"]

    path = write_config(tmp_path, raw)

    with pytest.raises(
        ConfigurationError,
        match="missing required fields",
    ):
        load_and_resolve_config(path)


@pytest.mark.parametrize(
    "number_of_uavs",
    [0, 6, 10, 20],
)
def test_unsupported_uav_count_is_rejected(
    tmp_path: Path,
    number_of_uavs: int,
) -> None:
    raw = read_baseline_raw()
    raw["topology"]["number_of_uavs"] = number_of_uavs

    path = write_config(tmp_path, raw)

    with pytest.raises(ConfigurationError):
        load_and_resolve_config(path)


@pytest.mark.parametrize(
    "ratio",
    [0.0, -0.1, 1.1],
)
def test_invalid_output_ratio_is_rejected(
    tmp_path: Path,
    ratio: float,
) -> None:
    raw = read_baseline_raw()
    raw["workload"]["output_input_ratio"] = ratio

    path = write_config(tmp_path, raw)

    with pytest.raises(ConfigurationError):
        load_and_resolve_config(path)


def test_unpaired_geometry_is_rejected(
    tmp_path: Path,
) -> None:
    raw = read_baseline_raw()
    raw["experiment"]["paired_geometry"] = False

    path = write_config(tmp_path, raw)

    with pytest.raises(
        ConfigurationError,
        match="paired_geometry must be true",
    ):
        load_and_resolve_config(path)


def test_invalid_scenario_order_is_rejected(
    tmp_path: Path,
) -> None:
    raw = read_baseline_raw()

    raw["experiment"]["scenarios"] = [
        "S3",
        "S2",
        "S1",
    ]

    path = write_config(tmp_path, raw)

    with pytest.raises(ConfigurationError):
        load_and_resolve_config(path)


def test_invalid_experiment_name_is_rejected(
    tmp_path: Path,
) -> None:
    raw = read_baseline_raw()
    raw["experiment"]["name"] = "invalid name!"

    path = write_config(tmp_path, raw)

    with pytest.raises(ConfigurationError):
        load_and_resolve_config(path)


def test_malformed_json_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "malformed.json"
    path.write_text(
        '{"schema_version": "1.0",',
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigurationError,
        match="Could not parse",
    ):
        load_and_resolve_config(path)


def test_nonexistent_file_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing.json"

    with pytest.raises(
        ConfigurationError,
        match="does not exist",
    ):
        load_and_resolve_config(path)
