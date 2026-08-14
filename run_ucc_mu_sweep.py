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

DEFAULT_MU_VALUES = [
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
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

    return campaign_dir


def load_aggregate(campaign_dir):
    aggregate_path = (
        campaign_dir / "aggregate.json"
    )

    with open(aggregate_path, "r") as f:
        data = json.load(f)

    return {
        row["scenario"]: row
        for row in data
    }


def preferred(s1, s2):
    if s1 < s2:
        return "S1"

    if s2 < s1:
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

        if number <= 0 or number > 1:
            raise ValueError(
                "All mu values must satisfy 0 < mu <= 1."
            )

        parsed.append(number)

    return parsed


def plot_tmax_vs_mu(rows, sweep_dir):
    rows = sorted(
        rows,
        key=lambda row:
            row["mu"],
    )

    mu_values = [
        row["mu"]
        for row in rows
    ]

    s1_values = [
        row["s1_tmax_emulated_mean_s"]
        for row in rows
    ]

    s2_values = [
        row["s2_tmax_emulated_mean_s"]
        for row in rows
    ]

    fig, ax = plt.subplots(
        figsize=(3.5, 2.5)
    )

    ax.plot(
        mu_values,
        s1_values,
        marker="o",
        linewidth=1.4,
        markersize=4,
        label="SV inference",
    )

    ax.plot(
        mu_values,
        s2_values,
        marker="s",
        linewidth=1.4,
        markersize=4,
        label="RCC inference",
    )

    ax.set_xlabel(
        r"Output-to-input ratio $\mu$",
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
        mu_values
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
        / "tmax_vs_output_ratio.pdf"
    )

    png_path = (
        sweep_dir
        / "tmax_vs_output_ratio.png"
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
        f"[SWEEP] Plot PDF saved to {pdf_path}"
    )

    print(
        f"[SWEEP] Plot PNG saved to {png_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep output-to-input ratio mu "
            "for the UCC experiment."
        )
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help=(
            "Independent repetitions "
            "per scenario and mu value."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=45,
    )

    parser.add_argument(
        "--values",
        nargs="+",
        default=[
            str(value)
            for value in DEFAULT_MU_VALUES
        ],
        help=(
            "Output-to-input ratios. "
            "Example: --values "
            "0.01 0.05 0.10 0.25 0.50"
        ),
    )

    args = parser.parse_args()

    if args.runs < 1:
        raise ValueError(
            "--runs must be >= 1"
        )

    mu_values = parse_values(
        args.values
    )

    original_text = CONFIG_PATH.read_text(
        encoding="utf-8"
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    sweep_dir = (
        RESULTS_DIR
        / f"sweep_mu_{timestamp}"
    )

    sweep_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    print(
        f"[SWEEP] mu values: {mu_values}"
    )

    print(
        f"[SWEEP] Results directory: "
        f"{sweep_dir}"
    )

    try:
        for mu in mu_values:

            print("")
            print(
                "[SWEEP] =============================="
            )

            print(
                f"[SWEEP] mu = {mu:.3f}"
            )

            print(
                "[SWEEP] =============================="
            )

            config = json.loads(
                original_text
            )

            config[
                "workload"
            ][
                "output_input_ratio"
            ] = mu

            config[
                "experiment"
            ][
                "name"
            ] = (
                f"ucc_mu_{mu:g}"
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
                s1["tmax_emulated_mean_s"],
                s2["tmax_emulated_mean_s"],
            )

            row = {
                "mu":
                    mu,

                "runs":
                    args.runs,

                "s1_tmax_model_s":
                    s1["tmax_model_s"],

                "s1_tmax_emulated_mean_s":
                    s1["tmax_emulated_mean_s"],

                "s1_tmax_emulated_std_s":
                    s1["tmax_emulated_std_s"],

                "s2_tmax_model_s":
                    s2["tmax_model_s"],

                "s2_tmax_emulated_mean_s":
                    s2["tmax_emulated_mean_s"],

                "s2_tmax_emulated_std_s":
                    s2["tmax_emulated_std_s"],

                "preferred_model":
                    model_preferred,

                "preferred_emulated":
                    emulated_preferred,

                "campaign_dir":
                    str(campaign_dir),
            }

            rows.append(row)

            print(
                f"[SWEEP] mu={mu:.3f} | "
                f"S1 model="
                f"{row['s1_tmax_model_s']:.6f} s | "
                f"S2 model="
                f"{row['s2_tmax_model_s']:.6f} s | "
                f"S1 emulated="
                f"{row['s1_tmax_emulated_mean_s']:.6f} s | "
                f"S2 emulated="
                f"{row['s2_tmax_emulated_mean_s']:.6f} s"
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
            "[SWEEP] Original configuration restored."
        )

    csv_path = (
        sweep_dir
        / "mu_sweep_summary.csv"
    )

    json_path = (
        sweep_dir
        / "mu_sweep_summary.json"
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

    plot_tmax_vs_mu(
        rows,
        sweep_dir,
    )

    print("")
    print(
        "[SWEEP] ===== MU SWEEP SUMMARY ====="
    )

    for row in rows:
        print(
            f"[SWEEP] mu="
            f"{row['mu']:.2f} | "
            f"S1_model="
            f"{row['s1_tmax_model_s']:.6f} | "
            f"S2_model="
            f"{row['s2_tmax_model_s']:.6f} | "
            f"S1_emulated="
            f"{row['s1_tmax_emulated_mean_s']:.6f} | "
            f"S2_emulated="
            f"{row['s2_tmax_emulated_mean_s']:.6f} | "
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
