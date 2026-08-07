"""Tests for UCC scalability result persistence."""

from __future__ import annotations
from dataclasses import replace


import csv
import json
from pathlib import Path

import pytest

from ucc.config import (
    ResolvedConfig,
    load_and_resolve_config,
)

from ucc.results import (
    build_parameters_document,
    configuration_fingerprint,
    create_run_directory,
    scalability_summary_rows,
    write_experiment_parameters,
    write_run_artifacts,
    write_scalability_summary,
)

from ucc.experiment import (
    ScalabilityExperimentResult,
    evaluate_scalability_experiment,
)
from ucc.results import (
    build_parameters_document,
    scalability_summary_rows,
    write_experiment_parameters,
    write_scalability_summary,
)


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
def result(
    config: ResolvedConfig,
) -> ScalabilityExperimentResult:
    """Run the complete scalability experiment once."""
    return evaluate_scalability_experiment(config)


def test_parameter_document_contains_scalability_protocol(
    config: ResolvedConfig,
) -> None:
    """The parameter document must preserve the protocol."""
    document = build_parameters_document(config)

    assert document["experiment"]["uav_counts"] == [
        4,
        6,
        8,
        10,
        12,
    ]

    assert (
        document["experiment"]["number_of_repetitions"]
        == 50
    )

    assert document["experiment"]["scenarios"] == [
        "S1",
        "S2",
        "S3",
    ]


def test_parameter_document_has_five_derived_points(
    config: ResolvedConfig,
) -> None:
    """Exactly five N-dependent resource sets are required."""
    document = build_parameters_document(config)

    derived = document["derived_by_uav_count"]

    assert len(derived) == 5

    assert [
        item["number_of_uavs"]
        for item in derived
    ] == [
        4,
        6,
        8,
        10,
        12,
    ]


def test_parameter_document_four_uav_resources(
    config: ResolvedConfig,
) -> None:
    """Check reference derived resources for N=4."""
    document = build_parameters_document(config)

    point = document["derived_by_uav_count"][0]

    assert point["grid_rows"] == 2
    assert point["grid_columns"] == 2

    assert (
        point["bandwidth_per_uav_hz"]
        == pytest.approx(2_500_000.0)
    )

    assert (
        point["backhaul_rate_per_flow_bps"]
        == pytest.approx(1_250_000.0)
    )

    assert (
        point["sv_capacity_per_job_cycles_s"]
        == pytest.approx(500_000_000.0)
    )

    assert (
        point["rcc_capacity_per_job_cycles_s"]
        == pytest.approx(
            125_000_000_000.0
        )
    )


def test_scalability_summary_has_fifteen_rows(
    result: ScalabilityExperimentResult,
) -> None:
    """Five UAV counts times three scenarios gives 15 rows."""
    rows = scalability_summary_rows(result)

    assert len(rows) == 15


def test_scalability_summary_order(
    result: ScalabilityExperimentResult,
) -> None:
    """Rows must follow N and scenario order."""
    rows = scalability_summary_rows(result)

    observed = [
        (
            row["number_of_uavs"],
            row["scenario"],
        )
        for row in rows
    ]

    expected = [
        (number_of_uavs, scenario)
        for number_of_uavs in (
            4,
            6,
            8,
            10,
            12,
        )
        for scenario in (
            "S1",
            "S2",
            "S3",
        )
    ]

    assert observed == expected


def test_known_eight_uav_baseline_is_preserved(
    result: ScalabilityExperimentResult,
) -> None:
    """The validated N=8 baseline must remain unchanged."""
    rows = scalability_summary_rows(result)

    lookup = {
        (
            row["number_of_uavs"],
            row["scenario"],
        ): row
        for row in rows
    }

    assert lookup[(8, "S1")][
        "mean_max_latency_ms"
    ] == pytest.approx(
        31_119.064964,
        rel=1e-8,
    )

    assert lookup[(8, "S2")][
        "mean_max_latency_ms"
    ] == pytest.approx(
        25_650.649642,
        rel=1e-8,
    )

    assert lookup[(8, "S3")][
        "mean_max_latency_ms"
    ] == pytest.approx(
        10_386.649642,
        rel=1e-8,
    )


def test_write_experiment_parameters(
    tmp_path: Path,
    config: ResolvedConfig,
) -> None:
    """The parameters JSON must be readable and complete."""
    destination = write_experiment_parameters(
        config=config,
        output_directory=tmp_path,
    )

    assert destination.exists()

    document = json.loads(
        destination.read_text(
            encoding="utf-8"
        )
    )

    assert document["experiment"]["name"] == (
        "ucc_scalability"
    )

    assert document["experiment"]["uav_counts"] == [
        4,
        6,
        8,
        10,
        12,
    ]


def test_write_scalability_summary(
    tmp_path: Path,
    config: ResolvedConfig,
    result: ScalabilityExperimentResult,
) -> None:
    """The CSV must contain one header and 15 data rows."""
    destination = write_scalability_summary(
        config=config,
        result=result,
        output_directory=tmp_path,
    )

    assert destination.exists()

    with destination.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    assert len(rows) == 15

    assert rows[0]["number_of_uavs"] == "4"
    assert rows[0]["scenario"] == "S1"
    assert rows[0]["scenario_label"] == "All-UAV"

    assert rows[-1]["number_of_uavs"] == "12"
    assert rows[-1]["scenario"] == "S3"
    assert rows[-1]["scenario_label"] == "All-RCC"


def test_existing_result_file_is_not_overwritten(
    tmp_path: Path,
    config: ResolvedConfig,
) -> None:
    """Existing artifacts must be protected by default."""
    write_experiment_parameters(
        config=config,
        output_directory=tmp_path,
    )

    with pytest.raises(FileExistsError):
        write_experiment_parameters(
            config=config,
            output_directory=tmp_path,
        )
def test_configuration_fingerprint_is_deterministic(
    config: ResolvedConfig,
) -> None:
    """Equal configurations must produce equal fingerprints."""
    first = configuration_fingerprint(config)
    second = configuration_fingerprint(config)

    assert first == second
    assert len(first) == 64

def test_create_run_directory(
    tmp_path: Path,
    config: ResolvedConfig,
) -> None:
    """A run directory must be unique and newly created."""
    test_config = replace(
        config,
        output_root_directory=str(tmp_path),
    )

    first = create_run_directory(test_config)
    second = create_run_directory(test_config)

    assert first.exists()
    assert first.is_dir()

    assert second.exists()
    assert second.is_dir()

    assert first != second

def test_write_run_artifacts(
    tmp_path: Path,
    config: ResolvedConfig,
    result: ScalabilityExperimentResult,
) -> None:
    """One run must persist both core scientific artifacts."""
    test_config = replace(
        config,
        output_root_directory=str(tmp_path),
    )

    artifacts = write_run_artifacts(
        config=test_config,
        result=result,
    )

    assert artifacts.run_directory.exists()

    assert artifacts.parameters_path.exists()
    assert artifacts.summary_path.exists()

    assert (
        artifacts.parameters_path.parent
        == artifacts.run_directory
    )

    assert (
        artifacts.summary_path.parent
        == artifacts.run_directory
    )
