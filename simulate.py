"""
UCC 2026 maritime inference offloading experiment.

This module is the entry point for the containerized adaptation of ITSASO
used in the UCC 2026 experiments.

At the current development stage, it performs two tasks:

1. validates the UCC experiment configuration;
2. computes the deterministic analytical reference defined by the
   system model in the paper.

The containerized execution will be added after the analytical reference
has been validated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = "cnf/ucc_config.json"


def load_config(config_path: str) -> dict[str, Any]:
    """Load the UCC experiment configuration."""

    path = Path(config_path)

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_config(config: dict[str, Any]) -> None:
    """Validate the parameters required by the UCC system model."""

    required_sections = {
        "experiment",
        "workload",
        "computing",
        "communication",
        "network",
        "runtime",
    }

    missing_sections = required_sections.difference(config.keys())

    if missing_sections:
        raise ValueError(
            "Missing configuration section(s): "
            + ", ".join(sorted(missing_sections))
        )

    experiment = config["experiment"]
    workload = config["workload"]
    computing = config["computing"]
    communication = config["communication"]

    n_uavs = experiment["n_uavs"]
    scenarios = experiment["scenarios"]

    if not isinstance(n_uavs, int) or n_uavs <= 0:
        raise ValueError("experiment.n_uavs must be a positive integer.")

    if scenarios != ["S1", "S2"]:
        raise ValueError(
            'experiment.scenarios must currently be exactly ["S1", "S2"].'
        )

    rates = communication["uav_sv_rates_mbps"]

    if len(rates) != n_uavs:
        raise ValueError(
            "The number of UAV-to-SV rates must match experiment.n_uavs. "
            f"Received N={n_uavs} and {len(rates)} rate values."
        )

    positive_parameters = {
        "workload.input_size_mbit": workload["input_size_mbit"],
        "workload.compute_intensity_cycles_per_bit":
            workload["compute_intensity_cycles_per_bit"],
        "computing.sv_capacity_gcycles_per_s":
            computing["sv_capacity_gcycles_per_s"],
        "computing.rcc_capacity_gcycles_per_s":
            computing["rcc_capacity_gcycles_per_s"],
        "communication.sv_rcc_capacity_mbps":
            communication["sv_rcc_capacity_mbps"],
    }

    for name, value in positive_parameters.items():
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero.")

    for index, rate in enumerate(rates, start=1):
        if rate <= 0:
            raise ValueError(
                f"communication.uav_sv_rates_mbps[{index - 1}] "
                "must be greater than zero."
            )

    output_ratio = workload["output_ratio"]

    if not 0 < output_ratio <= 1:
        raise ValueError(
            "workload.output_ratio must satisfy 0 < output_ratio <= 1."
        )

    if communication["sv_rcc_delay_ms"] < 0:
        raise ValueError(
            "communication.sv_rcc_delay_ms must be non-negative."
        )

    if computing["sharing_model"] != "equal_static":
        raise ValueError(
            'Only computing.sharing_model="equal_static" is supported '
            "by the current paper model."
        )


def derive_parameters(config: dict[str, Any]) -> dict[str, Any]:
    """
    Convert configuration values to SI units and derive W, D_out, and R^(2).
    """

    experiment = config["experiment"]
    workload = config["workload"]
    computing = config["computing"]
    communication = config["communication"]

    n_uavs = experiment["n_uavs"]

    d_bits = workload["input_size_mbit"] * 1_000_000.0
    mu = workload["output_ratio"]
    c_inf = workload["compute_intensity_cycles_per_bit"]

    d_out_bits = mu * d_bits
    workload_cycles = d_bits * c_inf

    f_sv = computing["sv_capacity_gcycles_per_s"] * 1_000_000_000.0
    f_rcc = computing["rcc_capacity_gcycles_per_s"] * 1_000_000_000.0

    access_rates = [
        value * 1_000_000.0
        for value in communication["uav_sv_rates_mbps"]
    ]

    c_bh = communication["sv_rcc_capacity_mbps"] * 1_000_000.0
    r2 = c_bh / n_uavs

    tau_bh = communication["sv_rcc_delay_ms"] / 1000.0

    return {
        "n_uavs": n_uavs,
        "d_bits": d_bits,
        "d_out_bits": d_out_bits,
        "mu": mu,
        "c_inf": c_inf,
        "workload_cycles": workload_cycles,
        "f_sv": f_sv,
        "f_rcc": f_rcc,
        "access_rates": access_rates,
        "c_bh": c_bh,
        "r2": r2,
        "tau_bh": tau_bh,
    }


def analytical_reference(
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """
    Evaluate Eqs. (6)-(9) of the current UCC paper model.

    S1:
        inference at the Surface Vessel (SV)

    S2:
        inference at the Rescue Coordination Center (RCC)
    """

    n_uavs = parameters["n_uavs"]
    d_bits = parameters["d_bits"]
    d_out_bits = parameters["d_out_bits"]
    workload_cycles = parameters["workload_cycles"]

    f_sv = parameters["f_sv"]
    f_rcc = parameters["f_rcc"]

    access_rates = parameters["access_rates"]
    r2 = parameters["r2"]
    tau_bh = parameters["tau_bh"]

    # Equal static sharing, Eq. (4).
    t_comp_sv = n_uavs * workload_cycles / f_sv
    t_comp_rcc = n_uavs * workload_cycles / f_rcc

    results_s1 = []
    results_s2 = []

    for uav_index, r1 in enumerate(access_rates, start=1):
        t_access = d_bits / r1

        # Eq. (6): inference executed at the SV.
        t_s1 = (
            t_access
            + t_comp_sv
            + d_out_bits / r2
            + tau_bh
        )

        # Eq. (7): inference executed at the RCC.
        t_s2 = (
            t_access
            + d_bits / r2
            + tau_bh
            + t_comp_rcc
        )

        results_s1.append(
            {
                "uav": uav_index,
                "latency_s": t_s1,
            }
        )

        results_s2.append(
            {
                "uav": uav_index,
                "latency_s": t_s2,
            }
        )

    t_max_s1 = max(item["latency_s"] for item in results_s1)
    t_max_s2 = max(item["latency_s"] for item in results_s2)

    preferred_tier = "S1" if t_max_s1 < t_max_s2 else "S2"

    return {
        "S1": {
            "execution_tier": "SV",
            "per_uav": results_s1,
            "t_max_s": t_max_s1,
        },
        "S2": {
            "execution_tier": "RCC",
            "per_uav": results_s2,
            "t_max_s": t_max_s2,
        },
        "preferred_scenario": preferred_tier,
    }


def print_configuration_summary(
    config: dict[str, Any],
    parameters: dict[str, Any],
) -> None:
    """Print the baseline experiment configuration."""

    experiment = config["experiment"]
    workload = config["workload"]
    computing = config["computing"]
    communication = config["communication"]

    print("=" * 72)
    print("UCC 2026 Maritime Inference Offloading")
    print("Analytical reference validation")
    print("=" * 72)

    print(f"Experiment:              {experiment['name']}")
    print(f"Monitoring UAVs (N):     {experiment['n_uavs']}")
    print(
        "UAV-SV rates:           "
        f"{communication['uav_sv_rates_mbps']} Mbit/s"
    )
    print(
        "SV-RCC capacity:        "
        f"{communication['sv_rcc_capacity_mbps']} Mbit/s"
    )
    print(
        "Per-flow R^(2):         "
        f"{parameters['r2'] / 1_000_000.0:.6f} Mbit/s"
    )
    print(
        "SV-RCC fixed delay:     "
        f"{communication['sv_rcc_delay_ms']} ms"
    )
    print(
        "Video chunk size (D):   "
        f"{workload['input_size_mbit']} Mbit"
    )
    print(f"Output ratio (mu):       {workload['output_ratio']}")
    print(
        "Output size (D_out):    "
        f"{parameters['d_out_bits'] / 1_000_000.0:.6f} Mbit"
    )
    print(
        "Compute intensity:      "
        f"{workload['compute_intensity_cycles_per_bit']} cycles/bit"
    )
    print(
        "Workload (W):           "
        f"{parameters['workload_cycles'] / 1_000_000_000.0:.6f} Gcycles"
    )
    print(
        "SV capacity (f1):       "
        f"{computing['sv_capacity_gcycles_per_s']} Gcycles/s"
    )
    print(
        "RCC capacity (f2):      "
        f"{computing['rcc_capacity_gcycles_per_s']} Gcycles/s"
    )
    print()


def print_results(results: dict[str, Any]) -> None:
    """Print analytical latency results."""

    for scenario in ("S1", "S2"):
        scenario_result = results[scenario]

        print(
            f"{scenario} - execution at "
            f"{scenario_result['execution_tier']}"
        )

        for item in scenario_result["per_uav"]:
            print(
                f"  UAV {item['uav']}: "
                f"{item['latency_s']:.6f} s"
            )

        print(
            f"  T_max,{scenario[-1]}: "
            f"{scenario_result['t_max_s']:.6f} s"
        )
        print()

    preferred = results["preferred_scenario"]

    if preferred == "S1":
        tier = "SV"
    else:
        tier = "RCC"

    print(
        f"Preferred scenario: {preferred} "
        f"(execution at {tier})"
    )
    print("=" * 72)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the UCC 2026 maritime inference offloading experiment."
        )
    )

    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Configuration file. Default: {DEFAULT_CONFIG}",
    )

    parser.add_argument(
        "--mode",
        choices=["validate"],
        default="validate",
        help=(
            "Execution mode. Only 'validate' is available at the current "
            "development stage."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    try:
        config = load_config(args.config)
        validate_config(config)

        parameters = derive_parameters(config)
        results = analytical_reference(parameters)

        print_configuration_summary(config, parameters)
        print_results(results)

    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
