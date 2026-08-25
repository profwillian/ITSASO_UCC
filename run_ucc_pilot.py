import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path


COMPOSE_FILE = os.environ.get(
    "UCC_COMPOSE_FILE",
    "docker-compose.ucc.yaml",
)
BASE_RESULTS_DIR = Path("res_ucc")


def run_command(command, env=None, timeout=None):
    return subprocess.run(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )


def compose_down():
    run_command(
        [
            "docker",
            "compose",
            "-f",
            COMPOSE_FILE,
            "down",
        ],
        timeout=30,
    )


def run_scenario(scenario, timeout):
    env = os.environ.copy()
    env["UCC_SCENARIO"] = scenario

    compose_down()

    result = run_command(
        [
            "docker",
            "compose",
            "-f",
            COMPOSE_FILE,
            "up",
            "--no-color",
        ],
        env=env,
        timeout=timeout,
    )

    return result


def copy_run_results(
    scenario,
    run_number,
    campaign_dir,
    log_text,
):
    scenario_dir = (
        campaign_dir
        / scenario
        / f"run_{run_number:02d}"
    )

    scenario_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_source = (
        BASE_RESULTS_DIR
        / f"{scenario}_summary.json"
    )

    results_source = (
        BASE_RESULTS_DIR
        / f"{scenario}_results.csv"
    )

    if not summary_source.exists():
        raise RuntimeError(
            f"Missing result file: "
            f"{summary_source}"
        )

    if not results_source.exists():
        raise RuntimeError(
            f"Missing result file: "
            f"{results_source}"
        )

    summary_target = (
        scenario_dir
        / "summary.json"
    )

    results_target = (
        scenario_dir
        / "results.csv"
    )

    log_target = (
        scenario_dir
        / "docker.log"
    )

    shutil.copy2(
        summary_source,
        summary_target,
    )

    shutil.copy2(
        results_source,
        results_target,
    )

    log_target.write_text(
        log_text,
        encoding="utf-8",
    )

    with open(summary_target, "r") as f:
        summary = json.load(f)

    return (
        summary,
        results_target,
    )


def read_per_uav_results(
    results_path,
    scenario,
    run_number,
):
    rows = []

    with open(
        results_path,
        newline="",
    ) as csvfile:

        reader = csv.DictReader(
            csvfile
        )

        for row in reader:
            rows.append(
                {
                    "scenario": scenario,
                    "run": run_number,
                    "uav_id":
                        int(row["uav_id"]),
                    "access_model_s":
                        float(
                            row[
                                "access_model_s"
                            ]
                        ),
                    "access_emulated_s":
                        float(
                            row[
                                "access_emulated_s"
                            ]
                        ),
                    "compute_model_s":
                        float(
                            row[
                                "compute_model_s"
                            ]
                        ),
                    "compute_emulated_s":
                        float(
                            row[
                                "compute_emulated_s"
                            ]
                        ),
                    "backhaul_model_s":
                        float(
                            row[
                                "backhaul_model_s"
                            ]
                        ),
                    "backhaul_emulated_s":
                        float(
                            row[
                                "backhaul_emulated_s"
                            ]
                        ),
                    "t_model_s":
                        float(
                            row[
                                "t_model_s"
                            ]
                        ),
                    "t_emulated_s":
                        float(
                            row[
                                "t_emulated_s"
                            ]
                        ),
                    "error_pct":
                        float(
                            row["error_pct"]
                        ),
                }
            )

    return rows


def write_csv(path, rows):
    if not rows:
        return

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


def aggregate_scenario(
    scenario,
    summary_rows,
):
    scenario_rows = [
        row
        for row in summary_rows
        if row["scenario"] == scenario
    ]

    emulated = [
        row["tmax_emulated_s"]
        for row in scenario_rows
    ]

    errors = [
        row["tmax_error_pct"]
        for row in scenario_rows
    ]

    return {
        "scenario": scenario,
        "runs": len(scenario_rows),
        "tmax_model_s":
            scenario_rows[0][
                "tmax_model_s"
            ],
        "tmax_emulated_mean_s":
            statistics.mean(
                emulated
            ),
        "tmax_emulated_std_s":
            statistics.stdev(
                emulated
            )
            if len(emulated) > 1
            else 0.0,
        "tmax_emulated_min_s":
            min(emulated),
        "tmax_emulated_max_s":
            max(emulated),
        "error_pct_mean":
            statistics.mean(
                errors
            ),
        "error_pct_std":
            statistics.stdev(
                errors
            )
            if len(errors) > 1
            else 0.0,
        "error_pct_min":
            min(errors),
        "error_pct_max":
            max(errors),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated UCC "
            "containerized experiments."
        )
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help=(
            "Number of repetitions "
            "per scenario."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=45,
        help=(
            "Maximum seconds allowed "
            "for each execution."
        ),
    )

    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=[
            "S1",
            "S2",
            "S3",
        ],
        default=[
            "S1",
            "S2",
            "S3",
        ],
        help=(
            "Scenarios to execute. "
            "Default: S1 S2 S3."
        ),
    )

    args = parser.parse_args()

    if args.runs < 1:
        raise ValueError(
            "--runs must be >= 1"
        )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    campaign_dir = (
        BASE_RESULTS_DIR
        / f"pilot_{timestamp}"
    )

    campaign_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows = []
    per_uav_rows = []

    scenarios = args.scenarios

    print(
        f"[PILOT] Campaign directory: "
        f"{campaign_dir}"
    )

    try:
        for scenario in scenarios:
            for run_number in range(
                1,
                args.runs + 1,
            ):
                print(
                    f"[PILOT] Running "
                    f"{scenario} "
                    f"{run_number}/{args.runs}..."
                )

                try:
                    result = run_scenario(
                        scenario,
                        args.timeout,
                    )
                except subprocess.TimeoutExpired:
                    compose_down()

                    raise RuntimeError(
                        f"{scenario} run "
                        f"{run_number} exceeded "
                        f"{args.timeout} seconds."
                    )

                if result.returncode != 0:
                    compose_down()

                    log_path = (
                        campaign_dir
                        / f"FAILED_"
                        f"{scenario}_"
                        f"run_"
                        f"{run_number:02d}.log"
                    )

                    log_path.write_text(
                        result.stdout,
                        encoding="utf-8",
                    )

                    raise RuntimeError(
                        f"{scenario} run "
                        f"{run_number} failed. "
                        f"See {log_path}"
                    )

                summary, results_path = (
                    copy_run_results(
                        scenario,
                        run_number,
                        campaign_dir,
                        result.stdout,
                    )
                )

                summary_rows.append(
                    {
                        "scenario":
                            scenario,
                        "run":
                            run_number,
                        "tmax_model_s":
                            summary[
                                "tmax_model_s"
                            ],
                        "tmax_emulated_s":
                            summary[
                                "tmax_emulated_s"
                            ],
                        "tmax_error_s":
                            summary[
                                "tmax_error_s"
                            ],
                        "tmax_error_pct":
                            summary[
                                "tmax_error_pct"
                            ],
                        "tmax_model_uav":
                            summary[
                                "tmax_model_uav"
                            ],
                        "tmax_emulated_uav":
                            summary[
                                "tmax_emulated_uav"
                            ],
                    }
                )

                per_uav_rows.extend(
                    read_per_uav_results(
                        results_path,
                        scenario,
                        run_number,
                    )
                )

                print(
                    f"[PILOT] {scenario} "
                    f"run {run_number}: "
                    f"Tmax_model="
                    f"{summary['tmax_model_s']:.6f} s, "
                    f"Tmax_emulated="
                    f"{summary['tmax_emulated_s']:.6f} s, "
                    f"error="
                    f"{summary['tmax_error_pct']:+.3f}%"
                )

    finally:
        compose_down()

    write_csv(
        campaign_dir
        / "summary_runs.csv",
        summary_rows,
    )

    write_csv(
        campaign_dir
        / "per_uav_runs.csv",
        per_uav_rows,
    )

    aggregates = [
        aggregate_scenario(
            scenario,
            summary_rows,
        )
        for scenario in scenarios
    ]

    with open(
        campaign_dir
        / "aggregate.json",
        "w",
    ) as f:

        json.dump(
            aggregates,
            f,
            indent=2,
        )

    print("")
    print(
        "[PILOT] ===== AGGREGATE ====="
    )

    for row in aggregates:
        print(
            f"[PILOT] {row['scenario']}: "
            f"runs={row['runs']}, "
            f"Tmax mean="
            f"{row['tmax_emulated_mean_s']:.6f} s, "
            f"std="
            f"{row['tmax_emulated_std_s']:.6f} s, "
            f"error mean="
            f"{row['error_pct_mean']:+.3f}%"
        )

    print(
        f"[PILOT] Results saved to "
        f"{campaign_dir}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"[PILOT] ERROR: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
