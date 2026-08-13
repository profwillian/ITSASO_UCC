import json
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone


def timestamp():
    return datetime.now(timezone.utc).isoformat()


config_path = os.environ.get("CONFIG", "cnf/ucc_config.json")
scenario = os.environ.get("SCENARIO", "S1")

with open(config_path, "r") as f:
    config = json.load(f)

valid_scenarios = config["experiment"]["scenarios"]

if scenario not in valid_scenarios:
    raise ValueError(
        f"Scenario {scenario} is not valid. "
        f"Available scenarios: {valid_scenarios}"
    )

host = "0.0.0.0"
port = config["nodes"]["rcc"]["port"]

num_uavs = config["experiment"]["num_uavs"]

input_size_mbit = config["workload"]["input_size_mbit"]
mu = config["workload"]["output_input_ratio"]
c_inf = config["workload"]["computational_intensity_cycles_per_bit"]

rcc_capacity_gcycles_s = config["computation"]["rcc_capacity_gcycles_per_s"]

input_size_bits = input_size_mbit * 1e6
workload_cycles = input_size_bits * c_inf
rcc_capacity_cycles_s = rcc_capacity_gcycles_s * 1e9

expected_compute_time_s = (
    num_uavs * workload_cycles / rcc_capacity_cycles_s
)


def process_at_rcc(message):
    start = time.perf_counter()

    time.sleep(expected_compute_time_s)

    measured = time.perf_counter() - start

    output_size_mbit = (
        message["input_size_mbit"] * mu
    )

    message["execution_tier"] = "RCC"
    message["compute_expected_s"] = expected_compute_time_s
    message["compute_measured_s"] = measured
    message["current_payload_mbit"] = output_size_mbit
    message["inference_finished_at"] = timestamp()

    return message


print(
    f"[RCC] Starting RCC node on port {port}. "
    f"Scenario={scenario}. Waiting for SV connection.",
    flush=True,
)

print(
    f"[RCC] Computation configuration: "
    f"W={workload_cycles:.0f} cycles, "
    f"capacity={rcc_capacity_gcycles_s:.3f} Gcycles/s, "
    f"N={num_uavs}, "
    f"expected_per_workload={expected_compute_time_s:.6f} s.",
    flush=True,
)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((host, port))
server.listen(1)

conn, addr = server.accept()

print(
    f"[RCC] SV connected from {addr[0]}:{addr[1]}",
    flush=True,
)

workloads = []

with conn.makefile("r") as stream:
    for line in stream:
        message = json.loads(line)

        if message.get("type") == "END":
            print(
                "[RCC] End-of-experiment signal received.",
                flush=True,
            )
            break

        if message["scenario"] != scenario:
            raise RuntimeError(
                f"[RCC] Scenario mismatch: container={scenario}, "
                f"message={message['scenario']}."
            )

        message["rcc_received_at"] = timestamp()
        workloads.append(message)

        print(
            f"[RCC] Received {message['workload_id']} "
            f"with {message['current_payload_mbit']:.3f} Mbit. "
            f"Batch={len(workloads)}/{num_uavs}.",
            flush=True,
        )

conn.close()
server.close()

if len(workloads) != num_uavs:
    raise RuntimeError(
        f"[RCC] Expected {num_uavs} workloads, "
        f"but received {len(workloads)}."
    )

if scenario == "S1":
    for message in workloads:
        if message["execution_tier"] != "SV":
            raise RuntimeError(
                f"[RCC] S1 workload from UAV "
                f"{message['uav_id']} was not processed at SV."
            )

        print(
            f"[RCC] UAV {message['uav_id']} result accepted. "
            f"execution_tier=SV, "
            f"compute_expected="
            f"{message['compute_expected_s']:.6f} s, "
            f"compute_measured="
            f"{message['compute_measured_s']:.6f} s.",
            flush=True,
        )

elif scenario == "S2":
    print(
        f"[RCC] Complete batch received. "
        f"Starting {num_uavs} concurrent inference workers.",
        flush=True,
    )

    batch_start = time.perf_counter()

    for message in workloads:
        message["inference_started_at"] = timestamp()

    with ThreadPoolExecutor(max_workers=num_uavs) as executor:
        workloads = list(
            executor.map(process_at_rcc, workloads)
        )

    batch_measured_s = time.perf_counter() - batch_start

    print(
        f"[RCC] Batch inference completed. "
        f"expected={expected_compute_time_s:.6f} s, "
        f"measured={batch_measured_s:.6f} s.",
        flush=True,
    )

    for message in workloads:
        print(
            f"[RCC] UAV {message['uav_id']} inference complete: "
            f"execution_tier=RCC, "
            f"expected={message['compute_expected_s']:.6f} s, "
            f"measured={message['compute_measured_s']:.6f} s, "
            f"result={message['current_payload_mbit']:.3f} Mbit.",
            flush=True,
        )

print(
    f"[RCC] Completed successfully. "
    f"Received {len(workloads)}/{num_uavs} workloads.",
    flush=True,
)
