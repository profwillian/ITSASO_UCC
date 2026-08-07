"""Plotting utilities for UCC scalability experiment results."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from ucc.model import ModelValidationError


EXPECTED_SCENARIOS = (
    "All-UAV",
    "All-Fog",
    "All-RCC",
)

EXPECTED_UAV_COUNTS = (
    4,
    6,
    8,
    10,
    12,
)

MARKERS = {
    "All-UAV": "o",
    "All-Fog": "s",
    "All-RCC": "^",
}

LINESTYLES = {
    "All-UAV": "-",
    "All-Fog": "--",
    "All-RCC": "-.",
}


def read_scalability_summary(
    csv_path: str | Path,
) -> dict[str, list[tuple[int, float]]]:
    """Read the consolidated scalability CSV for plotting."""
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Scalability summary does not exist: {path}"
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"Scalability summary is not a file: {path}"
        )

    series: dict[str, list[tuple[int, float]]] = {
        scenario: []
        for scenario in EXPECTED_SCENARIOS
    }

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {
            "number_of_uavs",
            "scenario_label",
            "mean_max_latency_ms",
        }

        if reader.fieldnames is None:
            raise ModelValidationError(
                "Scalability summary has no CSV header."
            )

        missing_columns = (
            required_columns
            - set(reader.fieldnames)
        )

        if missing_columns:
            raise ModelValidationError(
                "Scalability summary is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        for row in reader:
            scenario = row["scenario_label"]

            if scenario not in series:
                raise ModelValidationError(
                    f"Unexpected scenario label: {scenario}"
                )

            try:
                number_of_uavs = int(
                    row["number_of_uavs"]
                )

                mean_latency_ms = float(
                    row["mean_max_latency_ms"]
                )

            except (TypeError, ValueError) as exc:
                raise ModelValidationError(
                    "Invalid numeric value in scalability summary."
                ) from exc

            if number_of_uavs <= 0:
                raise ModelValidationError(
                    "number_of_uavs must be positive."
                )

            if mean_latency_ms <= 0.0:
                raise ModelValidationError(
                    "mean_max_latency_ms must be positive."
                )

            series[scenario].append(
                (
                    number_of_uavs,
                    mean_latency_ms,
                )
            )

    for scenario in EXPECTED_SCENARIOS:
        values = sorted(
            series[scenario],
            key=lambda item: item[0],
        )

        observed_uav_counts = tuple(
            number_of_uavs
            for number_of_uavs, _ in values
        )

        if observed_uav_counts != EXPECTED_UAV_COUNTS:
            raise ModelValidationError(
                f"{scenario} must contain UAV counts "
                f"{EXPECTED_UAV_COUNTS}. "
                f"Observed: {observed_uav_counts}"
            )

        series[scenario] = values

    return series


def generate_scalability_latency_plot(
    csv_path: str | Path,
    output_directory: str | Path,
) -> tuple[Path, Path]:
    """
    Generate publication and inspection versions of the
    scalability latency figure.
    """
    series = read_scalability_summary(
        csv_path
    )

    output_path = Path(
        output_directory
    )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_path = (
        output_path
        / "scalability_latency.pdf"
    )

    png_path = (
        output_path
        / "scalability_latency.png"
    )

    figure, axis = plt.subplots(
        figsize=(6.0, 4.0)
    )

    for scenario in EXPECTED_SCENARIOS:
        values = series[scenario]

        x_values = [
            number_of_uavs
            for number_of_uavs, _ in values
        ]

        y_values = [
            latency_ms
            for _, latency_ms in values
        ]

        axis.plot(
            x_values,
            y_values,
            marker=MARKERS[scenario],
            linestyle=LINESTYLES[scenario],
            linewidth=1.5,
            markersize=5,
            label=scenario,
        )

    axis.set_xlabel(
        "Number of UAVs"
    )

    axis.set_ylabel(
        "Mean Maximum End-to-End Latency (ms)"
    )

    axis.set_xticks(
        EXPECTED_UAV_COUNTS
    )

    axis.grid(
        True,
        linestyle=":",
        linewidth=0.6,
        alpha=0.7,
    )

    axis.legend(
        frameon=False
    )

    figure.tight_layout()

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return (
        pdf_path,
        png_path,
    )
