"""Tests for UCC scalability plotting utilities."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ucc.model import ModelValidationError
from ucc.plots import (
    EXPECTED_SCENARIOS,
    EXPECTED_UAV_COUNTS,
    generate_scalability_latency_plot,
    read_scalability_summary,
)


def write_valid_summary(path: Path) -> Path:
    """Create a minimal valid scalability summary CSV."""
    destination = path / "scalability_summary.csv"

    values = {
        "All-UAV": [
            30613.142,
            30866.301,
            31119.065,
            31371.075,
            31623.167,
        ],
        "All-Fog": [
            12911.416,
            19283.010,
            25650.650,
            32010.746,
            38371.674,
        ],
        "All-RCC": [
            5279.416,
            7835.010,
            10386.650,
            12930.746,
            15475.674,
        ],
    }

    rows = []

    for scenario in EXPECTED_SCENARIOS:
        for number_of_uavs, latency_ms in zip(
            EXPECTED_UAV_COUNTS,
            values[scenario],
        ):
            rows.append(
                {
                    "number_of_uavs": number_of_uavs,
                    "scenario_label": scenario,
                    "mean_max_latency_ms": latency_ms,
                }
            )

    with destination.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "number_of_uavs",
                "scenario_label",
                "mean_max_latency_ms",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    return destination


def test_read_scalability_summary(
    tmp_path: Path,
) -> None:
    """Read all three complete scalability series."""
    csv_path = write_valid_summary(tmp_path)

    series = read_scalability_summary(csv_path)

    assert tuple(series.keys()) == EXPECTED_SCENARIOS

    for scenario in EXPECTED_SCENARIOS:
        assert len(series[scenario]) == 5

        observed_counts = tuple(
            number_of_uavs
            for number_of_uavs, _ in series[scenario]
        )

        assert observed_counts == EXPECTED_UAV_COUNTS


def test_known_scalability_values_are_loaded(
    tmp_path: Path,
) -> None:
    """Preserve representative validated values."""
    csv_path = write_valid_summary(tmp_path)

    series = read_scalability_summary(csv_path)

    assert series["All-UAV"][2] == pytest.approx(
        (8, 31119.065)
    )

    assert series["All-Fog"][3] == pytest.approx(
        (10, 32010.746)
    )

    assert series["All-RCC"][4] == pytest.approx(
        (12, 15475.674)
    )


def test_missing_csv_is_rejected(
    tmp_path: Path,
) -> None:
    """A nonexistent scalability summary must fail."""
    with pytest.raises(FileNotFoundError):
        read_scalability_summary(
            tmp_path / "missing.csv"
        )


def test_missing_required_column_is_rejected(
    tmp_path: Path,
) -> None:
    """Required plotting columns must be present."""
    destination = tmp_path / "invalid.csv"

    destination.write_text(
        "number_of_uavs,scenario_label\n"
        "4,All-UAV\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ModelValidationError,
        match="missing required columns",
    ):
        read_scalability_summary(destination)


def test_incomplete_uav_counts_are_rejected(
    tmp_path: Path,
) -> None:
    """Each scenario must contain all five UAV counts."""
    csv_path = write_valid_summary(tmp_path)

    with csv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    rows = [
        row
        for row in rows
        if not (
            row["scenario_label"] == "All-UAV"
            and row["number_of_uavs"] == "12"
        )
    ]

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "number_of_uavs",
                "scenario_label",
                "mean_max_latency_ms",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ModelValidationError):
        read_scalability_summary(csv_path)


def test_generate_scalability_latency_plot(
    tmp_path: Path,
) -> None:
    """Generate non-empty PDF and PNG plot artifacts."""
    csv_path = write_valid_summary(tmp_path)

    output_directory = tmp_path / "plots"

    pdf_path, png_path = (
        generate_scalability_latency_plot(
            csv_path=csv_path,
            output_directory=output_directory,
        )
    )

    assert pdf_path.exists()
    assert png_path.exists()

    assert pdf_path.name == "scalability_latency.pdf"
    assert png_path.name == "scalability_latency.png"

    assert pdf_path.stat().st_size > 0
    assert png_path.stat().st_size > 0
