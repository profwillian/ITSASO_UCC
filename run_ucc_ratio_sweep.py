import argparse
import csv
import itertools
import json
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path


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
        json.dumps(
            config,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def run_pilot(
    scenario,
    runs,
    timeout,
):
    command = [
        sys.executable,
        "run_ucc_pilot.py",
        "--runs",
        str(runs),
        "--timeout",
        str(timeout),
        "--scenarios",
        scenario,
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

    if process.stdout is None:
        raise RuntimeError(
            "Could not capture pilot output."
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
            f"Pilot execution failed for "
            f"{scenario}."
        )

    if campaign_dir is None:
        raise RuntimeError(
            "Could not determine pilot "
            "campaign directory."
        )

    if not campaign_dir.exists():
        raise RuntimeError(
            f"Campaign directory does not exist: "
            f"{campaign_dir}"
        )

    return campaign_dir


def load_aggregate(
    campaign_dir,
    scenario,
):
    path = (
        campaign_dir
        / "aggregate.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Aggregate file not found: "
            f"{path}"
        )

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    for row in data:
        if row["scenario"] == scenario:
            return row

    raise RuntimeError(
        f"Scenario {scenario} "
        f"not found in {path}."
    )


def write_csv(path, rows):
    if not rows:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def parse_positive_values(
    values,
    name,
):
    parsed = []

    for value in values:
        number = float(value)

        if number <= 0:
            raise ValueError(
                f"All {name} values "
                "must be > 0."
            )

        parsed.append(number)

    return parsed


def assignment_label(
    sv_ids,
    rcc_ids,
):
    sv_text = (
        ",".join(
            str(value)
            for value in sv_ids
        )
        if sv_ids
        else "-"
    )

    rcc_text = (
        ",".join(
            str(value)
            for value in rcc_ids
        )
        if rcc_ids
        else "-"
    )

    return (
        f"SV[{sv_text}]_"
        f"RCC[{rcc_text}]"
    )


def summarize_ratio_rows(rows):
    grouped = {}

    for row in rows:
        key = (
            row["cbh_mbps"],
            row["m_sv"],
            row["rho_sv"],
        )

        grouped.setdefault(
            key,
            [],
        ).append(row)

    summary_rows = []

    tolerance = 1e-12

    for key in sorted(grouped):
        cbh_mbps, m_sv, rho_sv = key
        group = grouped[key]

        model_values = [
            row["tmax_model_s"]
            for row in group
        ]

        emulated_values = [
            row[
                "tmax_emulated_mean_s"
            ]
            for row in group
        ]

        model_min = min(
            model_values
        )

        model_max = max(
            model_values
        )

        model_span = (
            model_max
            - model_min
        )

        emulated_min = min(
            emulated_values
        )

        emulated_max = max(
            emulated_values
        )

        emulated_span = (
            emulated_max
            - emulated_min
        )

        summary_rows.append(
            {
                "cbh_mbps":
                    cbh_mbps,
                "m_sv":
                    m_sv,
                "m_rcc":
                    group[0]["m_rcc"],
                "rho_sv":
                    rho_sv,
                "num_assignments":
                    len(group),
                "tmax_model_mean_s":
                    statistics.mean(
                        model_values
                    ),
                "tmax_model_min_s":
                    model_min,
                "tmax_model_max_s":
                    model_max,
                "tmax_model_std_s":
                    statistics.stdev(
                        model_values
                    )
                    if len(
                        model_values
                    ) > 1
                    else 0.0,
                "tmax_model_span_s":
                    model_span,
                "model_assignments_equivalent":
                    model_span
                    <= tolerance,
                "tmax_emulated_mean_s":
                    statistics.mean(
                        emulated_values
                    ),
                "tmax_emulated_min_s":
                    emulated_min,
                "tmax_emulated_max_s":
                    emulated_max,
                "tmax_emulated_std_s":
                    statistics.stdev(
                        emulated_values
                    )
                    if len(
                        emulated_values
                    ) > 1
                    else 0.0,
                "tmax_emulated_span_s":
                    emulated_span,
            }
        )

    return summary_rows


def print_ratio_summary(
    summary_rows,
):
    print("")
    print(
        "[RATIO] ===== SUMMARY ====="
    )

    current_cbh = None

    for row in summary_rows:
        if row["cbh_mbps"] != current_cbh:
            current_cbh = (
                row["cbh_mbps"]
            )

            print("")
            print(
                f"[RATIO] "
                f"Cbh={current_cbh:.1f} "
                f"Mbit/s"
            )

        equivalent = (
            "YES"
            if row[
                "model_assignments_equivalent"
            ]
            else "NO"
        )

        print(
            f"[RATIO] "
            f"rho_sv="
            f"{row['rho_sv']:.2f} | "
            f"assignments="
            f"{row['num_assignments']} | "
            f"Tmax_model="
            f"{row['tmax_model_mean_s']:.6f} s | "
            f"model_span="
            f"{row['tmax_model_span_s']:.6f} s | "
            f"equivalent="
            f"{equivalent} | "
            f"Tmax_emulated_mean="
            f"{row['tmax_emulated_mean_s']:.6f} s"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep SV/RCC inference "
            "offloading ratios over "
            "backhaul capacities."
        )
    )

    parser.add_argument(
        "--cbh",
        nargs="+",
        type=float,
        default=DEFAULT_CBH_VALUES,
        help=(
            "Backhaul capacities in "
            "Mbit/s."
        ),
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help=(
            "Number of repetitions "
            "per assignment."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=45,
        help=(
            "Maximum seconds allowed "
            "for each pilot execution."
        ),
    )

    args = parser.parse_args()

    if args.runs < 1:
        raise ValueError(
            "--runs must be >= 1."
        )

    cbh_values = (
        parse_positive_values(
            args.cbh,
            "Cbh",
        )
    )

    original_config_text = (
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    config = json.loads(
        original_config_text
    )

    num_uavs = int(
        config[
            "experiment"
        ][
            "num_uavs"
        ]
    )

    if num_uavs != 4:
        raise RuntimeError(
            "This experiment is currently "
            "designed for N=4 so that "
            "rho_sv = "
            "{0, 0.25, 0.50, 0.75, 1.0}."
        )

    uav_ids = tuple(
        range(
            1,
            num_uavs + 1,
        )
    )

    timestamp = (
        datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    output_dir = (
        RESULTS_DIR
        / f"sweep_ratio_{timestamp}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    assignment_rows = []

    print(
        "[RATIO] "
        "SV/RCC ratio sweep starting."
    )

    print(
        f"[RATIO] "
        f"N={num_uavs}"
    )

    print(
        f"[RATIO] "
        f"Cbh values={cbh_values}"
    )

    print(
        "[RATIO] "
        "rho_sv values="
        "[0.00, 0.25, 0.50, 0.75, 1.00]"
    )

    print(
        f"[RATIO] "
        f"Output directory={output_dir}"
    )

    try:
        for cbh_mbps in cbh_values:

            config[
                "communication"
            ][
                "sv_rcc_backhaul_capacity_mbps"
            ] = cbh_mbps

            print("")
            print(
                "=" * 72
            )
            print(
                f"[RATIO] "
                f"Cbh={cbh_mbps:.1f} "
                f"Mbit/s"
            )
            print(
                "=" * 72
            )

            for m_sv in range(
                0,
                num_uavs + 1,
            ):
                rho_sv = (
                    m_sv / num_uavs
                )

                combinations = list(
                    itertools.combinations(
                        uav_ids,
                        m_sv,
                    )
                )

                for assignment_index, sv_ids in enumerate(
                    combinations,
                    start=1,
                ):
                    sv_ids = tuple(
                        sorted(sv_ids)
                    )

                    rcc_ids = tuple(
                        uav_id
                        for uav_id in uav_ids
                        if uav_id not in sv_ids
                    )

                    if m_sv == 0:
                        scenario = "S2"

                    elif m_sv == num_uavs:
                        scenario = "S1"

                    else:
                        scenario = "S3"

                        config.setdefault(
                            "mixed_offloading",
                            {},
                        )[
                            "sv_uav_ids"
                        ] = list(
                            sv_ids
                        )

                    write_config(
                        config
                    )

                    assignment = (
                        assignment_label(
                            sv_ids,
                            rcc_ids,
                        )
                    )

                    print("")
                    print(
                        f"[RATIO] "
                        f"Cbh="
                        f"{cbh_mbps:.1f} | "
                        f"rho_sv="
                        f"{rho_sv:.2f} | "
                        f"assignment "
                        f"{assignment_index}/"
                        f"{len(combinations)} | "
                        f"{assignment}"
                    )

                    pilot_dir = (
                        run_pilot(
                            scenario,
                            args.runs,
                            args.timeout,
                        )
                    )

                    aggregate = (
                        load_aggregate(
                            pilot_dir,
                            scenario,
                        )
                    )

                    row = {
                        "cbh_mbps":
                            cbh_mbps,
                        "m_sv":
                            m_sv,
                        "m_rcc":
                            num_uavs
                            - m_sv,
                        "rho_sv":
                            rho_sv,
                        "scenario":
                            scenario,
                        "assignment_index":
                            assignment_index,
                        "sv_uav_ids":
                            ",".join(
                                str(value)
                                for value
                                in sv_ids
                            ),
                        "rcc_uav_ids":
                            ",".join(
                                str(value)
                                for value
                                in rcc_ids
                            ),
                        "assignment":
                            assignment,
                        "runs":
                            aggregate[
                                "runs"
                            ],
                        "tmax_model_s":
                            aggregate[
                                "tmax_model_s"
                            ],
                        "tmax_emulated_mean_s":
                            aggregate[
                                "tmax_emulated_mean_s"
                            ],
                        "tmax_emulated_std_s":
                            aggregate[
                                "tmax_emulated_std_s"
                            ],
                        "error_pct_mean":
                            aggregate[
                                "error_pct_mean"
                            ],
                        "pilot_dir":
                            str(
                                pilot_dir
                            ),
                    }

                    assignment_rows.append(
                        row
                    )

                    print(
                        f"[RATIO] RESULT | "
                        f"Tmax_model="
                        f"{row['tmax_model_s']:.6f} s | "
                        f"Tmax_emulated="
                        f"{row['tmax_emulated_mean_s']:.6f} s"
                    )

        summary_rows = (
            summarize_ratio_rows(
                assignment_rows
            )
        )

        assignments_csv = (
            output_dir
            / "ratio_sweep_assignments.csv"
        )

        assignments_json = (
            output_dir
            / "ratio_sweep_assignments.json"
        )

        summary_csv = (
            output_dir
            / "ratio_sweep_summary.csv"
        )

        summary_json = (
            output_dir
            / "ratio_sweep_summary.json"
        )

        write_csv(
            assignments_csv,
            assignment_rows,
        )

        assignments_json.write_text(
            json.dumps(
                assignment_rows,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        write_csv(
            summary_csv,
            summary_rows,
        )

        summary_json.write_text(
            json.dumps(
                summary_rows,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        print_ratio_summary(
            summary_rows
        )

        print("")
        print(
            f"[RATIO] Assignment results: "
            f"{assignments_csv}"
        )

        print(
            f"[RATIO] Ratio summary: "
            f"{summary_csv}"
        )

        print(
            f"[RATIO] Results saved to "
            f"{output_dir}"
        )

    finally:
        CONFIG_PATH.write_text(
            original_config_text,
            encoding="utf-8",
        )

        print(
            "[RATIO] Original configuration "
            "restored."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"[RATIO] ERROR: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
