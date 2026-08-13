import csv
import json
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from ucc_protocol import (
    mbit_to_bytes,
    read_frame,
    send_json_line,
)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def save_results(
    config,
    scenario,
    workloads,
):
    results_dir = (
        config["experiment"]["results_dir"]
    )

    os.makedirs(
        results_dir,
        exist_ok=True,
    )

    rows = []

    for message in sorted(
        workloads,
        key=lambda x: x["uav_id"],
    ):
        access_model_s = (
            message["access_expected_s"]
        )

        access_emulated_s = (
            message["access_measured_s"]
        )

        compute_model_s = (
            message["compute_expected_s"]
        )

        compute_emulated_s = (
            message["compute_measured_s"]
        )

        backhaul_model_s = (
            message["backhaul_expected_s"]
        )

        backhaul_emulated_s = (
            message["backhaul_measured_s"]
        )

        t_model_s = (
            access_model_s
            + compute_model_s
            + backhaul_model_s
        )

        t_emulated_s = (
            message["completion_epoch_s"]
            - message["common_start_epoch_s"]
        )

        error_s = (
            t_emulated_s
            - t_model_s
        )

        error_pct = (
            100.0
            * error_s
            / t_model_s
            if t_model_s > 0
            else 0.0
        )

        rows.append(
            {
                "scenario": scenario,
                "uav_id":
                    message["uav_id"],
                "execution_tier":
                    message["execution_tier"],
                "access_rate_mbps":
                    message[
                        "configured_access_rate_mbps"
                    ],
                "generation_skew_ms":
                    message["generation_skew_s"]
                    * 1000.0,
                "access_model_s":
                    access_model_s,
                "access_emulated_s":
                    access_emulated_s,
                "compute_model_s":
                    compute_model_s,
                "compute_emulated_s":
                    compute_emulated_s,
                "backhaul_model_s":
                    backhaul_model_s,
                "backhaul_emulated_s":
                    backhaul_emulated_s,
                "t_model_s":
                    t_model_s,
                "t_emulated_s":
                    t_emulated_s,
                "error_s":
                    error_s,
                "error_pct":
                    error_pct,
                "backhaul_tcp_source_port":
                    message[
                        "backhaul_tcp_source_port"
                    ],
            }
        )

    tmax_model_row = max(
        rows,
        key=lambda row:
            row["t_model_s"],
    )

    tmax_emulated_row = max(
        rows,
        key=lambda row:
            row["t_emulated_s"],
    )

    tmax_model_s = (
        tmax_model_row["t_model_s"]
    )

    tmax_emulated_s = (
        tmax_emulated_row["t_emulated_s"]
    )

    tmax_error_s = (
        tmax_emulated_s
        - tmax_model_s
    )

    tmax_error_pct = (
        100.0
        * tmax_error_s
        / tmax_model_s
    )

    csv_path = os.path.join(
        results_dir,
        f"{scenario}_results.csv",
    )

    with open(
        csv_path,
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

    summary = {
        "experiment":
            config["experiment"]["name"],
        "scenario":
            scenario,
        "num_uavs":
            config["experiment"]["num_uavs"],
        "tmax_model_s":
            tmax_model_s,
        "tmax_emulated_s":
            tmax_emulated_s,
        "tmax_error_s":
            tmax_error_s,
        "tmax_error_pct":
            tmax_error_pct,
        "tmax_model_uav":
            tmax_model_row["uav_id"],
        "tmax_emulated_uav":
            tmax_emulated_row["uav_id"],
        "results_csv":
            csv_path,
    }

    json_path = os.path.join(
        results_dir,
        f"{scenario}_summary.json",
    )

    with open(
        json_path,
        "w",
    ) as jsonfile:

        json.dump(
            summary,
            jsonfile,
            indent=2,
        )

    return rows, summary


def receive_backhaul(connection):
    conn = connection["socket"]
    addr = connection["addr"]

    stream = conn.makefile("rb")

    try:
        message, payload = read_frame(
            stream
        )

        received_epoch_s = time.time()

        if message is None:
            raise RuntimeError(
                "[RCC] Backhaul connection closed "
                "before workload was received."
            )

        if message.get("type") != "WORKLOAD":
            raise RuntimeError(
                "[RCC] Expected WORKLOAD "
                "on backhaul flow."
            )

        if message["scenario"] != scenario:
            raise RuntimeError(
                f"[RCC] Scenario mismatch: "
                f"container={scenario}, "
                f"message={message['scenario']}."
            )

        expected_payload_bytes = (
            mbit_to_bytes(
                message["current_payload_mbit"]
            )
        )

        if len(payload) != expected_payload_bytes:
            raise RuntimeError(
                f"[RCC] Payload mismatch for UAV "
                f"{message['uav_id']}: "
                f"expected={expected_payload_bytes}, "
                f"received={len(payload)}."
            )

        message[
            "backhaul_payload_bytes_received"
        ] = len(payload)

        message[
            "rcc_observed_tcp_source_port"
        ] = addr[1]

        message["rcc_received_at"] = (
            timestamp()
        )

        message["rcc_received_epoch_s"] = (
            received_epoch_s
        )

        message["backhaul_measured_s"] = (
            received_epoch_s
            - message[
                "backhaul_send_start_epoch_s"
            ]
        )

        ack = {
            "type": "DELIVERY_ACK",
            "scenario": scenario,
            "uav_id": message["uav_id"],
            "workload_id": message["workload_id"],
            "payload_bytes_received": len(payload),
            "rcc_received_epoch_s": received_epoch_s,
        }

        send_json_line(
            conn,
            ack,
        )

        print(
            f"[RCC] Delivery ACK sent for "
            f"UAV {message['uav_id']}: "
            f"bytes={len(payload)}.",
            flush=True,
        )

        return message

    finally:
        stream.close()
        conn.close()


config_path = os.environ.get(
    "CONFIG",
    "cnf/ucc_config.json",
)

scenario = os.environ.get(
    "SCENARIO",
    "S1",
)


with open(config_path, "r") as f:
    config = json.load(f)


valid_scenarios = (
    config["experiment"]["scenarios"]
)


if scenario not in valid_scenarios:
    raise ValueError(
        f"Scenario {scenario} is not valid. "
        f"Available scenarios: {valid_scenarios}"
    )


host = "0.0.0.0"

port = (
    config["nodes"]["rcc"]["port"]
)

num_uavs = (
    config["experiment"]["num_uavs"]
)

input_size_mbit = (
    config["workload"]["input_size_mbit"]
)

mu = (
    config["workload"]["output_input_ratio"]
)

c_inf = (
    config["workload"]
    ["computational_intensity_cycles_per_bit"]
)

rcc_capacity_gcycles_s = (
    config["computation"]
    ["rcc_capacity_gcycles_per_s"]
)


input_size_bits = (
    input_size_mbit * 1e6
)

workload_cycles = (
    input_size_bits * c_inf
)

rcc_capacity_cycles_s = (
    rcc_capacity_gcycles_s * 1e9
)

expected_compute_time_s = (
    num_uavs
    * workload_cycles
    / rcc_capacity_cycles_s
)


def process_at_rcc(message):
    message["inference_started_at"] = (
        timestamp()
    )

    start = time.perf_counter()

    # Computation remains model-controlled.
    time.sleep(
        expected_compute_time_s
    )

    measured = (
        time.perf_counter()
        - start
    )

    output_size_mbit = (
        message["input_size_mbit"]
        * mu
    )

    message["execution_tier"] = "RCC"

    message["compute_expected_s"] = (
        expected_compute_time_s
    )

    message["compute_measured_s"] = (
        measured
    )

    message["current_payload_mbit"] = (
        output_size_mbit
    )

    message["inference_finished_at"] = (
        timestamp()
    )

    message["completion_epoch_s"] = (
        time.time()
    )

    return message


print(
    f"[RCC] Starting RCC node "
    f"on port {port}. "
    f"Scenario={scenario}.",
    flush=True,
)


server = socket.socket(
    socket.AF_INET,
    socket.SOCK_STREAM,
)

server.setsockopt(
    socket.SOL_SOCKET,
    socket.SO_REUSEADDR,
    1,
)

server.bind(
    (host, port)
)

server.listen(num_uavs)


print(
    f"[RCC] Waiting for {num_uavs} "
    f"independent backhaul TCP flows.",
    flush=True,
)


connections = []


while len(connections) < num_uavs:
    conn, addr = server.accept()

    connections.append(
        {
            "socket": conn,
            "addr": addr,
        }
    )

    print(
        f"[RCC] Accepted backhaul TCP flow "
        f"{len(connections)}/{num_uavs} "
        f"from {addr[0]}:{addr[1]}.",
        flush=True,
    )


with ThreadPoolExecutor(
    max_workers=num_uavs
) as executor:

    workloads = list(
        executor.map(
            receive_backhaul,
            connections,
        )
    )


server.close()


workloads.sort(
    key=lambda message:
        message["uav_id"]
)


for message in workloads:

    print(
        f"[RCC] UAV {message['uav_id']} "
        f"backhaul observed: "
        f"source_port="
        f"{message['rcc_observed_tcp_source_port']}, "
        f"bytes="
        f"{message['backhaul_payload_bytes_received']}, "
        f"model="
        f"{message['backhaul_expected_s']:.6f} s, "
        f"observed="
        f"{message['backhaul_measured_s']:.6f} s.",
        flush=True,
    )

    if (
        message[
            "rcc_observed_tcp_source_port"
        ]
        != message[
            "backhaul_tcp_source_port"
        ]
    ):
        raise RuntimeError(
            f"[RCC] TCP source-port mismatch "
            f"for UAV {message['uav_id']}."
        )


total_backhaul_bytes_received = sum(
    message[
        "backhaul_payload_bytes_received"
    ]
    for message in workloads
)


print(
    f"[RCC] Backhaul payload total received = "
    f"{total_backhaul_bytes_received} bytes.",
    flush=True,
)


if scenario == "S1":

    for message in workloads:

        if message["execution_tier"] != "SV":
            raise RuntimeError(
                f"[RCC] S1 workload from UAV "
                f"{message['uav_id']} "
                f"was not processed at SV."
            )

        message["completion_epoch_s"] = (
            message["rcc_received_epoch_s"]
        )


elif scenario == "S2":

    print(
        f"[RCC] Complete batch received. "
        f"Starting {num_uavs} concurrent "
        f"inference workers.",
        flush=True,
    )

    batch_start = (
        time.perf_counter()
    )

    with ThreadPoolExecutor(
        max_workers=num_uavs
    ) as executor:

        workloads = list(
            executor.map(
                process_at_rcc,
                workloads,
            )
        )

    batch_measured_s = (
        time.perf_counter()
        - batch_start
    )

    print(
        f"[RCC] Batch inference completed. "
        f"expected={expected_compute_time_s:.6f} s, "
        f"measured={batch_measured_s:.6f} s.",
        flush=True,
    )


rows, summary = save_results(
    config,
    scenario,
    workloads,
)


print(
    "[RCC] ===== UCC END-TO-END RESULTS =====",
    flush=True,
)


for row in rows:

    print(
        f"[RCC] UAV {row['uav_id']} | "
        f"T_model={row['t_model_s']:.6f} s | "
        f"T_emulated={row['t_emulated_s']:.6f} s | "
        f"error={row['error_pct']:+.3f}%",
        flush=True,
    )


print(
    f"[RCC] T_max model    = "
    f"{summary['tmax_model_s']:.6f} s "
    f"(UAV {summary['tmax_model_uav']})",
    flush=True,
)


print(
    f"[RCC] T_max emulated = "
    f"{summary['tmax_emulated_s']:.6f} s "
    f"(UAV {summary['tmax_emulated_uav']})",
    flush=True,
)


print(
    f"[RCC] T_max error    = "
    f"{summary['tmax_error_s']:+.6f} s "
    f"({summary['tmax_error_pct']:+.3f}%)",
    flush=True,
)


print(
    f"[RCC] Completed successfully. "
    f"Received {len(workloads)}/{num_uavs} workloads.",
    flush=True,
)
