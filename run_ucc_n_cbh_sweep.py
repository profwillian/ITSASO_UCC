import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


CONFIG_PATH = Path("cnf/ucc_config.json")
PILOT_SCRIPT = Path("run_ucc_pilot.py")
RESULTS_DIR = Path("res_ucc")

GENERATED_COMPOSE = Path(
    "docker-compose.ucc.generated.yaml"
)

DEFAULT_N_VALUES = [
    4,
    8,
    12,
    16,
]

DEFAULT_CBH_VALUES = [
    2.0,
    5.0,
    10.0,
    20.0,
]

BASE_ACCESS_RATES = [
    80.0,
    90.0,
    100.0,
    90.0,
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


def build_access_rates(num_uavs):
    """
    Repeat the N=4 baseline access-rate profile.

    The experiment uses multiples of four so that
    every increase in N preserves exactly the same
    distribution of UAV-to-SV effective rates.
    """

    if num_uavs % len(BASE_ACCESS_RATES) != 0:
        raise ValueError(
            "N must be a multiple of 4 in this experiment "
            "to preserve the baseline UAV-to-SV rate profile."
        )

    return [
        BASE_ACCESS_RATES[
            index % len(BASE_ACCESS_RATES)
        ]
        for index in range(num_uavs)
    ]


def generate_compose(
    num_uavs,
    output_path,
):
    lines = [
        "services:",
        "",
        "  rcc_node:",
        "    build:",
        "      context: .",
        "      dockerfile: DockerfileRCC",
        "    image: ucc_rcc_node",
        "    container_name: rcc_node",
        "    privileged: true",
        "    environment:",
        "      - CONFIG=cnf/ucc_config.json",
        "      - SCENARIO=${UCC_SCENARIO:-S1}",
        "    volumes:",
        "      - ./:/home/",
        "    networks:",
        "      - ucc_backhaul_net",
        "",
        "  sv_node:",
        "    build:",
        "      context: .",
        "      dockerfile: DockerfileSV",
        "    image: ucc_sv_node",
        "    container_name: sv_node",
        "    privileged: true",
        "    environment:",
        "      - CONFIG=cnf/ucc_config.json",
        "      - SCENARIO=${UCC_SCENARIO:-S1}",
        "    volumes:",
        "      - ./:/home/",
        "    networks:",
        "      - ucc_access_net",
        "      - ucc_backhaul_net",
        "    depends_on:",
        "      - rcc_node",
        "",
    ]

    for uav_id in range(
        1,
        num_uavs + 1,
    ):
        lines.extend(
            [
                f"  uav_node_{uav_id}:",
                "    build:",
                "      context: .",
                "      dockerfile: DockerfileUAV",
                "    image: ucc_uav_node",
                f"    container_name: uav_node_{uav_id}",
                "    privileged: true",
                "    environment:",
                "      - CONFIG=cnf/ucc_config.json",
                f"      - UAV_ID={uav_id}",
                "      - SCENARIO=${UCC_SCENARIO:-S1}",
                "    volumes:",
                "      - ./:/home/",
                "    networks:",
                "      - ucc_access_net",
                "    depends_on:",
                "      - sv_node",
                "",
            ]
        )

    lines.extend(
        [
            "networks:",
            "",
            "  ucc_access_net:",
            "    name: ucc_access_net",
            "",
            "  ucc_backhaul_net:",
            "    name: ucc_backhaul_net",
            "",
        ]
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def compose_down(
    compose_path,
):
    if not compose_path.exists():
        return

    subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(compose_path),
            "down",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def validate_compose(
    compose_path,
):
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(compose_path),
            "config",
            "--quiet",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Generated Docker Compose is invalid:\n"
            + result.stdout
        )


def run_pilot(
    scenario,
    runs,
    timeout,
    compose_path,
):
    env = os.environ.copy()

    env[
        "UCC_COMPOSE_FILE"
    ] = str(
        compose_path
    )

    command = [
        sys.executable,
        str(PILOT_SCRIPT),
        "--runs",
        str(runs),
        "--scenarios",
        scenario,
        "--timeout",
        str(timeout),
    ]

    result = subprocess.run(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    print(
        result.stdout,
        end="",
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Pilot failed for scenario {scenario}."
        )

    matches = re.findall(
        r"\[PILOT\] Results saved to (.+)",
        result.stdout,
    )

    if not matches:
        raise RuntimeError(
            "Could not locate pilot result directory."
        )

    pilot_dir = Path(
        matches[-1].strip()
    )

    aggregate_path = (
        pilot_dir
        / "aggregate.json"
    )

    if not aggregate_path.exists():
        raise RuntimeError(
            f"Missing aggregate file: "
            f"{aggregate_path}"
        )

    aggregate = json.loads(
        aggregate_path.read_text(
            encoding="utf-8"
        )
    )

    if len(aggregate) != 1:
        raise RuntimeError(
            "Expected exactly one scenario "
            "in pilot aggregate."
        )

    return (
        pilot_dir,
        aggregate[0],
    )


def write_csv(
    path,
    rows,
):
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def save_rows(
    output_dir,
    rows,
):
    csv_path = (
        output_dir
        / "n_cbh_sweep_summary.csv"
    )

    json_path = (
        output_dir
        / "n_cbh_sweep_summary.json"
    )

    write_csv(
        csv_path,
        rows,
    )

    json_path.write_text(
        json.dumps(
            rows,
            indent=2,
        ),
        encoding="utf-8",
    )


def summarize_optima(
    rows,
):
    groups = {}

    for row in rows:
        key = (
            row["num_uavs"],
            row["cbh_mbps"],
        )

        groups.setdefault(
            key,
            [],
        ).append(
            row
        )

    optima = []

    for key in sorted(groups):
        num_uavs, cbh_mbps = key
        group = groups[key]

        model_best = min(
            group,
            key=lambda row:
                row["tmax_model_s"],
        )

        emulated_best = min(
            group,
            key=lambda row:
                row[
                    "tmax_emulated_mean_s"
                ],
        )

        model_min = (
            model_best[
                "tmax_model_s"
            ]
        )

        model_ties = [
            row
            for row in group
            if abs(
                row["tmax_model_s"]
                - model_min
            )
            <= 1e-12
        ]

        optima.append(
            {
                "num_uavs":
                    num_uavs,
                "cbh_mbps":
                    cbh_mbps,
                "model_rho_sv_star":
                    model_best[
                        "rho_sv"
                    ],
                "model_m_sv_star":
                    model_best[
                        "m_sv"
                    ],
                "model_tmax_star_s":
                    model_best[
                        "tmax_model_s"
                    ],
                "model_tie_count":
                    len(
                        model_ties
                    ),
                "emulated_rho_sv_star":
                    emulated_best[
                        "rho_sv"
                    ],
                "emulated_m_sv_star":
                    emulated_best[
                        "m_sv"
                    ],
                "emulated_tmax_star_s":
                    emulated_best[
                        "tmax_emulated_mean_s"
                    ],
            }
        )

    return optima


def save_optima(
    output_dir,
    rows,
):
    optima = (
        summarize_optima(
            rows
        )
    )

    csv_path = (
        output_dir
        / "n_cbh_optima.csv"
    )

    json_path = (
        output_dir
        / "n_cbh_optima.json"
    )

    write_csv(
        csv_path,
        optima,
    )

    json_path.write_text(
        json.dumps(
            optima,
            indent=2,
        ),
        encoding="utf-8",
    )

    return optima


def print_optima(
    optima,
):
    print("")
    print(
        "[N-CBH] ===== OPTIMAL RATIOS ====="
    )

    current_n = None

    for row in optima:

        if row["num_uavs"] != current_n:
            current_n = row[
                "num_uavs"
            ]

            print("")
            print(
                f"[N-CBH] N={current_n}"
            )

        print(
            "[N-CBH] "
            f"Cbh={row['cbh_mbps']:.1f} Mbit/s | "
            f"model rho*="
            f"{row['model_rho_sv_star']:.4f} | "
            f"Tmax*="
            f"{row['model_tmax_star_s']:.6f} s | "
            f"emulated rho*="
            f"{row['emulated_rho_sv_star']:.4f} | "
            f"Tmax*="
            f"{row['emulated_tmax_star_s']:.6f} s"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep the number of monitoring UAVs "
            "and SV-to-RCC backhaul capacity while "
            "evaluating every feasible SV execution ratio."
        )
    )

    parser.add_argument(
        "--n",
        nargs="+",
        type=int,
        default=DEFAULT_N_VALUES,
        help=(
            "Numbers of UAVs. "
            "Default: 4 8 12 16."
        ),
    )

    parser.add_argument(
        "--cbh",
        nargs="+",
        type=float,
        default=DEFAULT_CBH_VALUES,
        help=(
            "Backhaul capacities in Mbit/s. "
            "Default: 2 5 10 20."
        ),
    )

    parser.add_argument(
        "--m-sv",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional subset of numbers of workloads "
            "executed at the SV. If omitted, all values "
            "from 0 to N are evaluated."
        ),
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help=(
            "Containerized repetitions per "
            "configuration. Default: 1."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help=(
            "Maximum seconds allowed for each "
            "containerized execution. Default: 120."
        ),
    )

    args = parser.parse_args()

    if args.runs < 1:
        raise ValueError(
            "--runs must be >= 1."
        )

    n_values = sorted(
        set(
            args.n
        )
    )

    cbh_values = sorted(
        set(
            args.cbh
        )
    )

    for num_uavs in n_values:

        if num_uavs <= 0:
            raise ValueError(
                "N must be positive."
            )

        if (
            num_uavs
            % len(
                BASE_ACCESS_RATES
            )
            != 0
        ):
            raise ValueError(
                "All N values must be multiples "
                "of 4 in this experiment."
            )

    for cbh_mbps in cbh_values:

        if cbh_mbps <= 0:
            raise ValueError(
                "Cbh values must be positive."
            )

    m_sv_by_n = {}

    for num_uavs in n_values:

        if args.m_sv is None:
            values = list(
                range(
                    0,
                    num_uavs + 1,
                )
            )
        else:
            values = sorted(
                set(
                    args.m_sv
                )
            )

            invalid = [
                value
                for value in values
                if not (
                    0
                    <= value
                    <= num_uavs
                )
            ]

            if invalid:
                raise ValueError(
                    f"Invalid --m-sv values "
                    f"for N={num_uavs}: "
                    f"{invalid}"
                )

        m_sv_by_n[
            num_uavs
        ] = values

    original_config_text = (
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    config = json.loads(
        original_config_text
    )

    timestamp = (
        datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    output_dir = (
        RESULTS_DIR
        / f"sweep_n_cbh_{timestamp}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    total_configurations = sum(
        len(
            m_sv_by_n[
                num_uavs
            ]
        )
        * len(
            cbh_values
        )
        for num_uavs
        in n_values
    )

    configuration_index = 0

    print(
        "[N-CBH] Variable-N sweep starting."
    )

    print(
        f"[N-CBH] N values={n_values}"
    )

    print(
        f"[N-CBH] Cbh values={cbh_values}"
    )

    print(
        f"[N-CBH] Runs per configuration="
        f"{args.runs}"
    )

    print(
        f"[N-CBH] Total configurations="
        f"{total_configurations}"
    )

    print(
        f"[N-CBH] Output directory="
        f"{output_dir}"
    )

    try:

        for num_uavs in n_values:

            config[
                "experiment"
            ][
                "num_uavs"
            ] = num_uavs

            config[
                "communication"
            ][
                "uav_sv_rates_mbps"
            ] = (
                build_access_rates(
                    num_uavs
                )
            )

            generate_compose(
                num_uavs,
                GENERATED_COMPOSE,
            )

            validate_compose(
                GENERATED_COMPOSE
            )

            print("")
            print(
                "=" * 72
            )
            print(
                f"[N-CBH] N={num_uavs}"
            )
            print(
                f"[N-CBH] rates="
                f"{config['communication']['uav_sv_rates_mbps']}"
            )
            print(
                "=" * 72
            )

            for cbh_mbps in cbh_values:

                config[
                    "communication"
                ][
                    "sv_rcc_backhaul_capacity_mbps"
                ] = cbh_mbps

                for m_sv in m_sv_by_n[
                    num_uavs
                ]:

                    configuration_index += 1

                    rho_sv = (
                        m_sv
                        / num_uavs
                    )

                    if m_sv == 0:

                        scenario = "S2"

                        sv_ids = []

                    elif m_sv == num_uavs:

                        scenario = "S1"

                        sv_ids = list(
                            range(
                                1,
                                num_uavs + 1,
                            )
                        )

                    else:

                        scenario = "S3"

                        # Canonical workload assignment.
                        # Workloads are homogeneous and the
                        # synchronized access barrier makes
                        # UAV identity irrelevant to the
                        # analytical Tmax for fixed m_sv.
                        sv_ids = list(
                            range(
                                1,
                                m_sv + 1,
                            )
                        )

                        config.setdefault(
                            "mixed_offloading",
                            {},
                        )[
                            "sv_uav_ids"
                        ] = sv_ids

                    rcc_ids = [
                        uav_id
                        for uav_id
                        in range(
                            1,
                            num_uavs + 1,
                        )
                        if uav_id
                        not in set(
                            sv_ids
                        )
                    ]

                    write_config(
                        config
                    )

                    print("")
                    print(
                        f"[N-CBH] "
                        f"{configuration_index}/"
                        f"{total_configurations} | "
                        f"N={num_uavs} | "
                        f"Cbh={cbh_mbps:.1f} | "
                        f"m_sv={m_sv} | "
                        f"rho_sv={rho_sv:.4f} | "
                        f"scenario={scenario}"
                    )

                    start = (
                        time.perf_counter()
                    )

                    pilot_dir, aggregate = (
                        run_pilot(
                            scenario,
                            args.runs,
                            args.timeout,
                            GENERATED_COMPOSE,
                        )
                    )

                    elapsed_s = (
                        time.perf_counter()
                        - start
                    )

                    row = {
                        "num_uavs":
                            num_uavs,
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
                        "elapsed_wall_s":
                            elapsed_s,
                        "pilot_dir":
                            str(
                                pilot_dir
                            ),
                    }

                    rows.append(
                        row
                    )

                    # Persist after every configuration so
                    # partial results survive interruptions.
                    save_rows(
                        output_dir,
                        rows,
                    )

                    print(
                        f"[N-CBH] RESULT | "
                        f"Tmax_model="
                        f"{row['tmax_model_s']:.6f} s | "
                        f"Tmax_emulated="
                        f"{row['tmax_emulated_mean_s']:.6f} s | "
                        f"error="
                        f"{row['error_pct_mean']:+.3f}%"
                    )

        optima = save_optima(
            output_dir,
            rows,
        )

        print_optima(
            optima
        )

        print("")
        print(
            "[N-CBH] Sweep completed successfully."
        )

        print(
            f"[N-CBH] Results saved to "
            f"{output_dir}"
        )

    finally:

        print("")
        print(
            "[N-CBH] Cleaning generated containers."
        )

        compose_down(
            GENERATED_COMPOSE
        )

        print(
            "[N-CBH] Restoring original configuration."
        )

        CONFIG_PATH.write_text(
            original_config_text,
            encoding="utf-8",
        )

        if GENERATED_COMPOSE.exists():
            GENERATED_COMPOSE.unlink()


if __name__ == "__main__":
    main()
