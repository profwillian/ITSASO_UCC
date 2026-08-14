import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


CONFIG_PATH = Path("cnf/ucc_config.json")
RESULTS_DIR = Path("res_ucc")

DEFAULT_CBH_VALUES = [
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    5.0,
]


def write_config(config):
    CONFIG_PATH.write_text(
        json.dumps(
            config,
            indent=2,
        )
        + "\n"
    )


def run_pilot(runs, timeout):
    command = [
        sys.executable,
        "run_ucc_pilot.py",
        "--runs",
        str(runs),
        "--timeout",
        str(timeout),
    ]

    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
    )

    print(result.stdout)

    if result.returncode != 0:
        print(
            result.stderr,
            file=sys.stderr,
        )
        raise RuntimeError(
            "run_ucc_pilot.py failed."
        )

    match = re.search(
        r"\[PILOT\] Results saved to (.+)",
        result.stdout,
    )

    if match:
        return Path(
            match.group(1).strip()
        )

    # Fallback: identify newest pilot directory.
    candidates = sorted(
        RESULTS_DIR.glob("pilot_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise RuntimeError(
            "Could not locate pilot results."
        )

    return candidates[0]


def load_aggregate(pilot_dir):
    path = (
        pilot_dir
        / "aggregate.json"
    )

    if not path.exists():
        raise RuntimeError(
            f"Missing aggregate file: {path}"
        )

    return json.loads(
        path.read_text()
    )


def get_scenario(
    aggregate,
    scenario,
):
    for row in aggregate:
        if row["scenario"] == scenario:
            return row

    raise RuntimeError(
        f"Scenario {scenario} "
        "not found in aggregate."
    )


def plot_results(rows, output_dir):
    scenarios = [
        ("S1", "All-SV"),
        ("S2", "All-RCC"),
        ("S3", "Mixed 50/50"),
    ]

    fig, ax = plt.subplots(
        figsize=(3.5, 2.5)
    )

    for scenario, label in scenarios:
        scenario_rows = [
            row
            for row in rows
            if row["scenario"] == scenario
        ]

        scenario_rows.sort(
            key=lambda row:
                row["cbh_mbps"]
        )

        x = [
            row["cbh_mbps"]
            for row in scenario_rows
        ]

        y = [
            row["tmax_emulated_mean_s"]
            for row in scenario_rows
        ]

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.3,
            markersize=4,
            label=label,
        )

    ax.set_xlabel(
        "Backhaul capacity (Mbit/s)",
        fontsize=9,
    )

    ax.set_ylabel(
        "Maximum latency (s)",
        fontsize=9,
    )

    ax.set_xticks(
        DEFAULT_CBH_VALUES
    )

    ax.set_ylim(
        bottom=0
    )

    ax.tick_params(
        axis="both",
        labelsize=8,
    )

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.4,
        alpha=0.4,
    )

    ax.legend(
        frameon=False,
        fontsize=7,
    )

    fig.tight_layout()

    pdf_path = (
        output_dir
        / "tmax_vs_backhaul_three_scenarios.pdf"
    )

    png_path = (
        output_dir
        / "tmax_vs_backhaul_three_scenarios.png"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"[SWEEP] Plot saved to {pdf_path}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--runs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--values",
        nargs="+",
        type=float,
        default=DEFAULT_CBH_VALUES,
    )

    args = parser.parse_args()

    original_config = json.loads(
        CONFIG_PATH.read_text()
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_dir = (
        RESULTS_DIR
        / f"sweep_mixed_{timestamp}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    try:
        for cbh in args.values:
            print()
            print(
                "=" * 60
            )
            print(
                f"[SWEEP] Cbh = "
                f"{cbh:.3f} Mbit/s"
            )
            print(
                "=" * 60
            )

            config = json.loads(
                json.dumps(
                    original_config
                )
            )

            config[
                "communication"
            ][
                "sv_rcc_backhaul_capacity_mbps"
            ] = cbh

            config[
                "experiment"
            ][
                "scenarios"
            ] = [
                "S1",
                "S2",
                "S3",
            ]

            write_config(config)

            pilot_dir = run_pilot(
                args.runs,
                args.timeout,
            )

            aggregate = load_aggregate(
                pilot_dir
            )

            for scenario in [
                "S1",
                "S2",
                "S3",
            ]:
                result = get_scenario(
                    aggregate,
                    scenario,
                )

                rows.append(
                    {
                        "cbh_mbps":
                            cbh,
                        "scenario":
                            scenario,
                        "runs":
                            result["runs"],
                        "tmax_model_s":
                            result[
                                "tmax_model_s"
                            ],
                        "tmax_emulated_mean_s":
                            result[
                                "tmax_emulated_mean_s"
                            ],
                        "tmax_emulated_std_s":
                            result[
                                "tmax_emulated_std_s"
                            ],
                        "error_mean_percent":
                            100.0
                            * (
                                result[
                                    "tmax_emulated_mean_s"
                                ]
                                - result[
                                    "tmax_model_s"
                                ]
                            )
                            / result[
                                "tmax_model_s"
                            ],
                        "pilot_dir":
                            str(pilot_dir),
                    }
                )

    finally:
        write_config(
            original_config
        )

        print(
            "[SWEEP] Original configuration restored."
        )

    csv_path = (
        output_dir
        / "mixed_sweep_summary.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    json_path = (
        output_dir
        / "mixed_sweep_summary.json"
    )

    json_path.write_text(
        json.dumps(
            rows,
            indent=2,
        )
        + "\n"
    )

    plot_results(
        rows,
        output_dir,
    )

    print()
    print(
        "[SWEEP] ===== SUMMARY ====="
    )

    for cbh in args.values:
        selected = [
            row
            for row in rows
            if row["cbh_mbps"] == cbh
        ]

        selected.sort(
            key=lambda row:
                row[
                    "tmax_emulated_mean_s"
                ]
        )

        winner = selected[0]

        print(
            f"[SWEEP] Cbh={cbh:.3f}: "
            f"best={winner['scenario']}, "
            f"Tmax="
            f"{winner['tmax_emulated_mean_s']:.6f} s"
        )

    print()
    print(
        f"[SWEEP] Results saved to "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
