import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


CONFIG_PATH = Path("cnf/ucc_config.json")
RESULTS_DIR = Path("res_ucc")

F1_VALUES = [
    2.0,
    10.0,
    50.0,
    100.0,
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
            "Could not determine pilot "
            "campaign directory."
        )

    if not campaign_dir.exists():
        raise RuntimeError(
            f"Campaign directory does not exist: "
            f"{campaign_dir}"
        )

    return campaign_dir


def load_aggregate(campaign_dir):
    aggregate_path = (
        campaign_dir
        / "aggregate.json"
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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep SV computing capacity "
            "for the UCC experiment."
        )
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help=(
            "Independent repetitions "
            "per scenario and f1 value."
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

    args = parser.parse_args()

    if args.runs < 1:
        raise ValueError(
            "--runs must be >= 1"
        )

    original_text = CONFIG_PATH.read_text(
        encoding="utf-8"
    )

    original_config = json.loads(
        original_text
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    sweep_dir = (
        RESULTS_DIR
        / f"sweep_f1_{timestamp}"
    )

    sweep_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    print(
        f"[SWEEP] Results directory: "
        f"{sweep_dir}"
    )

    try:
        for f1 in F1_VALUES:
            print("")
            print(
                "[SWEEP] =============================="
            )
            print(
                f"[SWEEP] f1 = "
                f"{f1:.3f} Gcycles/s"
            )
            print(
                "[SWEEP] =============================="
            )

            config = json.loads(
                original_text
            )

            config[
                "computation"
            ][
                "sv_capacity_gcycles_per_s"
            ] = f1

            config[
                "experiment"
            ][
                "name"
            ] = (
                f"ucc_f1_{f1:g}"
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

            row = {
                "f1_gcycles_s":
                    f1,

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
                f"[SWEEP] f1={f1:g}: "
                f"S1 model="
                f"{row['s1_tmax_model_s']:.6f} s, "
                f"S2 model="
                f"{row['s2_tmax_model_s']:.6f} s"
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
            "[SWEEP] Original configuration "
            "restored."
        )

    csv_path = (
        sweep_dir
        / "f1_sweep_summary.csv"
    )

    json_path = (
        sweep_dir
        / "f1_sweep_summary.json"
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

    print("")
    print(
        "[SWEEP] ===== F1 SWEEP SUMMARY ====="
    )

    for row in rows:
        print(
            f"[SWEEP] f1="
            f"{row['f1_gcycles_s']:>6.1f} | "
            f"S1_model="
            f"{row['s1_tmax_model_s']:.6f} | "
            f"S2_model="
            f"{row['s2_tmax_model_s']:.6f} | "
            f"S1_emulated="
            f"{row['s1_tmax_emulated_mean_s']:.6f} | "
            f"S2_emulated="
            f"{row['s2_tmax_emulated_mean_s']:.6f} | "
            f"model={row['preferred_model']} | "
            f"emulated={row['preferred_emulated']}"
        )

    print(
        f"[SWEEP] Results saved to "
        f"{sweep_dir}"
    )


if __name__ == "__main__":
    main()
