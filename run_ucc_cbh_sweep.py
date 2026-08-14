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
    2.0,
    5.0,
    10.0,
    20.0,
]


def write_config(config):
    CONFIG_PATH.write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
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

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    campaign_dir = None

    pattern = re.compile(
        r"\[PILOT\] Results saved to (.+)$"
    )

    for line in process.stdout:
        print(line, end="")

        match = pattern.search(
            line.strip()
        )

        if match:
            campaign_dir = Path(
                match.group(1)
            )

    return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            "Pilot execution failed."
        )

    if campaign_dir is None:
        raise RuntimeError(
            "Could not determine pilot campaign directory."
        )

    if not campaign_dir.exists():
        raise RuntimeError(
            f"Campaign directory does not exist: "
            f"{campaign_dir}"
        )

    return campaign_dir


def load_aggregate(campaign_dir):
    aggregate_path = (
        campaign_dir / "aggregate.json"
    )

    with open(
        aggregate_path,
        "r",
    ) as f:
        data = json.load(f)

    return {
        row["scenario"]: row
        for row in data
    }


def load_sweep_rows(sweep_dir):
    json_path = (
        sweep_dir / "cbh_sweep_summary.json"
    )

    if not json_path.exists():
        raise FileNotFoundError(
            f"Summary file not found: {json_path}"
        )

    with open(
        json_path,
        "r",
    ) as f:
        rows = json.load(f)

    if not rows:
        raise RuntimeError(
            f"No sweep results found in {json_path}"
        )

    return rows


def resolve_sweep_dir(value):
    if value == "latest":
        candidates = [
            path
            for path in RESULTS_DIR.glob(
                "sweep_cbh_*"
            )
            if (
                path
                / "cbh_sweep_summary.json"
            ).exists()
        ]

        if not candidates:
            raise RuntimeError(
                "No previous Cbh sweep "
                "with summary JSON was found."
            )

        return max(
            candidates,
            key=lambda path:
                path.stat().st_mtime,
        )

    sweep_dir = Path(value)

    if not sweep_dir.exists():
        raise FileNotFoundError(
            f"Sweep directory not found: "
            f"{sweep_dir}"
        )

    return sweep_dir


def preferred(s1, s2):
    tolerance = 1e-12

    if s1 < s2 - tolerance:
        return "S1"

    if s2 < s1 - tolerance:
        return "S2"

    return "TIE"


def write_csv(path, rows):
    with open(
        path,
        "w",
        newline="",
    ) as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def parse_values(values):
    parsed = []

    for value in values:
        number = float(value)

        if number <= 0:
            raise ValueError(
                "All Cbh values must be > 0."
            )

        parsed.append(number)

    return parsed


def plot_tmax_vs_cbh(rows, sweep_dir):
    rows = sorted(
        rows,
        key=lambda row:
            row["cbh_mbps"],
    )

    cbh_values = [
        row["cbh_mbps"]
        for row in rows
    ]

    s1_values = [
        row[
            "s1_tmax_emulated_mean_s"
        ]
        for row in rows
    ]

    s2_values = [
        row[
            "s2_tmax_emulated_mean_s"
        ]
        for row in rows
    ]

    fig, ax = plt.subplots(
        figsize=(3.5, 2.5)
    )

    ax.plot(
        cbh_values,
        s1_values,
        marker="o",
        linewidth=1.4,
        markersize=4,
        label="SV inference",
    )

    ax.plot(
        cbh_values,
        s2_values,
        marker="s",
        linewidth=1.4,
        markersize=4,
        label="RCC inference",
    )

    ax.set_xlabel(
        "Backhaul capacity (Mbit/s)",
        fontsize=9,
    )

    ax.set_ylabel(
        "Maximum latency (s)",
        fontsize=9,
    )

    ax.set_ylim(
        bottom=0.0,
    )

    ax.set_xticks(
        cbh_values
    )

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.5,
        alpha=0.5,
    )

    ax.legend(
        frameon=False,
        fontsize=8,
        loc="best",
    )

    ax.tick_params(
        axis="both",
        labelsize=8,
    )

    fig.tight_layout()

    pdf_path = (
        sweep_dir
        / "tmax_vs_backhaul_capacity.pdf"
    )

    png_path = (
        sweep_dir
        / "tmax_vs_backhaul_capacity.png"
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
        f"[SWEEP] Plot PDF saved to "
        f"{pdf_path}"
    )

    print(
        f"[SWEEP] Plot PNG saved to "
        f"{png_path}"
    )



def plot_decision_boundary(rows, sweep_dir):
    rows = sorted(
        rows,
        key=lambda row:
            row["cbh_mbps"],
    )

    cbh_values = [
        row["cbh_mbps"]
        for row in rows
    ]

    model_gap = [
        row["model_gap_s"]
        for row in rows
    ]

    emulated_gap = [
        row["emulated_gap_s"]
        for row in rows
    ]

    fig, ax = plt.subplots(
        figsize=(3.5, 2.5)
    )

    ax.plot(
        cbh_values,
        model_gap,
        marker="o",
        linewidth=1.4,
        markersize=4,
        label="Analytical model",
    )

    ax.plot(
        cbh_values,
        emulated_gap,
        marker="s",
        linewidth=1.4,
        markersize=4,
        label="Containerized emulation",
    )

    ax.axhline(
        y=0.0,
        linewidth=0.9,
        linestyle="--",
    )

    ax.set_xlabel(
        "Backhaul capacity (Mbit/s)",
        fontsize=9,
    )

    ax.set_ylabel(
        r"Latency difference $\Delta T$ (s)",
        fontsize=9,
    )

    ax.set_xticks(
        cbh_values
    )

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.5,
        alpha=0.5,
    )

    ax.legend(
        frameon=False,
        fontsize=7.5,
        loc="upper left",
    )

    ax.tick_params(
        axis="both",
        labelsize=8,
    )

    fig.tight_layout()

    pdf_path = (
        sweep_dir
        / "decision_boundary_gap.pdf"
    )

    png_path = (
        sweep_dir
        / "decision_boundary_gap.png"
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
        f"[SWEEP] Decision-boundary PDF saved to "
        f"{pdf_path}"
    )

    print(
        f"[SWEEP] Decision-boundary PNG saved to "
        f"{png_path}"
    )



def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep SV-RCC aggregate "
            "backhaul capacity for "
            "the UCC experiment."
        )
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help=(
            "Independent repetitions "
            "per scenario and Cbh value."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=45,
        help=(
            "Timeout in seconds for "
            "each containerized run."
        ),
    )

    parser.add_argument(
        "--values",
        nargs="+",
        default=[
            str(value)
            for value
            in DEFAULT_CBH_VALUES
        ],
        help=(
            "Aggregate backhaul capacities "
            "in Mbit/s. "
            "Example: --values 2 5 10 20"
        ),
    )

    parser.add_argument(
        "--plot-only",
        nargs="?",
        const="latest",
        default=None,
        help=(
            "Regenerate the plot from an "
            "existing sweep without running "
            "the containers. "
            "Use 'latest' or provide the "
            "sweep directory."
        ),
    )

    parser.add_argument(
        "--boundary-plot",
        nargs="?",
        const="latest",
        default=None,
        help=(
            "Generate the decision-boundary plot "
            "from an existing Cbh sweep without "
            "running the containers. "
            "Use 'latest' or provide the "
            "sweep directory."
        ),
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # DECISION-BOUNDARY PLOT MODE
    # -----------------------------------------------------

    if args.boundary_plot is not None:
        sweep_dir = resolve_sweep_dir(
            args.boundary_plot
        )

        rows = load_sweep_rows(
            sweep_dir
        )

        print(
            "[SWEEP] Decision-boundary plot mode."
        )

        print(
            f"[SWEEP] Loading results from "
            f"{sweep_dir}"
        )

        plot_decision_boundary(
            rows,
            sweep_dir,
        )

        print(
            "[SWEEP] Decision-boundary plot "
            "generated without new experimental runs."
        )

        return

    # -----------------------------------------------------
    # PLOT-ONLY MODE
    # -----------------------------------------------------

    if args.plot_only is not None:
        sweep_dir = resolve_sweep_dir(
            args.plot_only
        )

        rows = load_sweep_rows(
            sweep_dir
        )

        print(
            f"[SWEEP] Plot-only mode."
        )

        print(
            f"[SWEEP] Loading results from "
            f"{sweep_dir}"
        )

        plot_tmax_vs_cbh(
            rows,
            sweep_dir,
        )

        print(
            "[SWEEP] Plot regenerated "
            "without new experimental runs."
        )

        return

    # -----------------------------------------------------
    # EXPERIMENT MODE
    # -----------------------------------------------------

    if args.runs < 1:
        raise ValueError(
            "--runs must be >= 1"
        )

    cbh_values = parse_values(
        args.values
    )

    original_text = (
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    timestamp = (
        datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    sweep_dir = (
        RESULTS_DIR
        / f"sweep_cbh_{timestamp}"
    )

    sweep_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    print(
        f"[SWEEP] Cbh values: "
        f"{cbh_values}"
    )

    print(
        f"[SWEEP] Results directory: "
        f"{sweep_dir}"
    )

    try:
        for cbh in cbh_values:
            print("")
            print(
                "[SWEEP] "
                "=============================="
            )

            print(
                f"[SWEEP] Cbh = "
                f"{cbh:.3f} Mbit/s"
            )

            print(
                "[SWEEP] "
                "=============================="
            )

            config = json.loads(
                original_text
            )

            config[
                "communication"
            ][
                "sv_rcc_backhaul_capacity_mbps"
            ] = cbh

            config[
                "experiment"
            ][
                "name"
            ] = (
                f"ucc_cbh_{cbh:g}"
            )

            write_config(
                config
            )

            campaign_dir = run_pilot(
                runs=args.runs,
                timeout=args.timeout,
            )

            aggregate = load_aggregate(
                campaign_dir
            )

            s1 = aggregate["S1"]
            s2 = aggregate["S2"]

            model_preferred = preferred(
                s1["tmax_model_s"],
                s2["tmax_model_s"],
            )

            emulated_preferred = preferred(
                s1[
                    "tmax_emulated_mean_s"
                ],
                s2[
                    "tmax_emulated_mean_s"
                ],
            )

            model_gap_s = (
                s1["tmax_model_s"]
                - s2["tmax_model_s"]
            )

            emulated_gap_s = (
                s1[
                    "tmax_emulated_mean_s"
                ]
                - s2[
                    "tmax_emulated_mean_s"
                ]
            )

            row = {
                "cbh_mbps":
                    cbh,

                "runs":
                    args.runs,

                "s1_tmax_model_s":
                    s1[
                        "tmax_model_s"
                    ],

                "s1_tmax_emulated_mean_s":
                    s1[
                        "tmax_emulated_mean_s"
                    ],

                "s1_tmax_emulated_std_s":
                    s1[
                        "tmax_emulated_std_s"
                    ],

                "s2_tmax_model_s":
                    s2[
                        "tmax_model_s"
                    ],

                "s2_tmax_emulated_mean_s":
                    s2[
                        "tmax_emulated_mean_s"
                    ],

                "s2_tmax_emulated_std_s":
                    s2[
                        "tmax_emulated_std_s"
                    ],

                "model_gap_s":
                    model_gap_s,

                "emulated_gap_s":
                    emulated_gap_s,

                "preferred_model":
                    model_preferred,

                "preferred_emulated":
                    emulated_preferred,

                "campaign_dir":
                    str(campaign_dir),
            }

            rows.append(row)

            print("")

            print(
                f"[SWEEP] Cbh={cbh:g}: "
                f"S1 model="
                f"{row['s1_tmax_model_s']:.6f} s, "
                f"S2 model="
                f"{row['s2_tmax_model_s']:.6f} s"
            )

            print(
                f"[SWEEP] model gap="
                f"{model_gap_s:+.6f} s, "
                f"emulated gap="
                f"{emulated_gap_s:+.6f} s"
            )

            print(
                f"[SWEEP] preferred model="
                f"{model_preferred}, "
                f"preferred emulated="
                f"{emulated_preferred}"
            )

    finally:
        CONFIG_PATH.write_text(
            original_text,
            encoding="utf-8",
        )

        print("")

        print(
            "[SWEEP] Original "
            "configuration restored."
        )

    csv_path = (
        sweep_dir
        / "cbh_sweep_summary.csv"
    )

    json_path = (
        sweep_dir
        / "cbh_sweep_summary.json"
    )

    write_csv(
        csv_path,
        rows,
    )

    with open(
        json_path,
        "w",
    ) as f:
        json.dump(
            rows,
            f,
            indent=2,
        )

    plot_tmax_vs_cbh(
        rows,
        sweep_dir,
    )

    print("")

    print(
        "[SWEEP] ===== "
        "CBH SWEEP SUMMARY ====="
    )

    for row in rows:
        print(
            f"[SWEEP] Cbh="
            f"{row['cbh_mbps']:>6.2f} | "
            f"S1_model="
            f"{row['s1_tmax_model_s']:.6f} | "
            f"S2_model="
            f"{row['s2_tmax_model_s']:.6f} | "
            f"S1_emulated="
            f"{row['s1_tmax_emulated_mean_s']:.6f} | "
            f"S2_emulated="
            f"{row['s2_tmax_emulated_mean_s']:.6f} | "
            f"model_gap="
            f"{row['model_gap_s']:+.6f} | "
            f"emulated_gap="
            f"{row['emulated_gap_s']:+.6f} | "
            f"model="
            f"{row['preferred_model']} | "
            f"emulated="
            f"{row['preferred_emulated']}"
        )

    print(
        f"[SWEEP] Results saved to "
        f"{sweep_dir}"
    )


if __name__ == "__main__":
    main()
