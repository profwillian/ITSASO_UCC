import json
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from ucc_protocol import (
    mbit_to_bytes,
    read_frame,
    read_json_line,
    send_frame,
    send_json_line,
)


START_LEAD_TIME_S = 0.5


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def connect_with_retry(host, port, timeout=60):
    deadline = time.time() + timeout

    while time.time() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            sock.connect((host, port))
            return sock
        except OSError:
            sock.close()
            time.sleep(0.2)

    raise RuntimeError(
        f"[SV] Could not connect to RCC at {host}:{port} "
        f"within {timeout} seconds."
    )


def receive_workload(connection):
    metadata, payload = read_frame(connection["stream"])

    if metadata is None:
        raise RuntimeError(
            f"[SV] UAV {connection['uav_id']} closed "
            f"the connection before sending its workload."
        )

    if metadata.get("type") != "WORKLOAD":
        raise RuntimeError(
            f"[SV] Expected WORKLOAD from "
            f"UAV {connection['uav_id']}."
        )

    metadata["sv_received_at"] = timestamp()
    metadata["sv_received_epoch_s"] = time.time()
    metadata["access_payload_bytes_received"] = len(payload)

    return {
        "message": metadata,
        "payload": payload,
    }


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

sv_port = config["nodes"]["sv"]["port"]
rcc_host = config["nodes"]["rcc"]["host"]
rcc_port = config["nodes"]["rcc"]["port"]

num_uavs = config["experiment"]["num_uavs"]

input_size_mbit = config["workload"]["input_size_mbit"]
mu = config["workload"]["output_input_ratio"]
c_inf = config["workload"]["computational_intensity_cycles_per_bit"]

sv_capacity_gcycles_s = (
    config["computation"]["sv_capacity_gcycles_per_s"]
)

backhaul_capacity_mbps = (
    config["communication"]["sv_rcc_backhaul_capacity_mbps"]
)

backhaul_fixed_delay_s = (
    config["communication"]["sv_rcc_fixed_delay_ms"] / 1000.0
)

per_flow_backhaul_rate_mbps = (
    backhaul_capacity_mbps / num_uavs
)

input_size_bits = input_size_mbit * 1e6
workload_cycles = input_size_bits * c_inf
sv_capacity_cycles_s = sv_capacity_gcycles_s * 1e9

expected_compute_time_s = (
    num_uavs * workload_cycles / sv_capacity_cycles_s
)


def process_at_sv(workload):
    message = workload["message"]

    start = time.perf_counter()
    time.sleep(expected_compute_time_s)
    measured = time.perf_counter() - start

    output_size_mbit = (
        message["input_size_mbit"] * mu
    )

    output_payload = bytes(
        mbit_to_bytes(output_size_mbit)
    )

    message["execution_tier"] = "SV"
    message["compute_expected_s"] = expected_compute_time_s
    message["compute_measured_s"] = measured
    message["current_payload_mbit"] = output_size_mbit
    message["inference_finished_at"] = timestamp()

    return {
        "message": message,
        "payload": output_payload,
    }


def transmit_backhaul(workload):
    message = workload["message"]
    payload = workload["payload"]

    payload_mbit = message["current_payload_mbit"]

    transmission_time_s = (
        payload_mbit / per_flow_backhaul_rate_mbps
    )

    expected_backhaul_time_s = (
        transmission_time_s + backhaul_fixed_delay_s
    )

    # One independent TCP connection per UAV workload.
    rcc_sock = connect_with_retry(
        rcc_host,
        rcc_port,
    )

    local_ip, local_port = rcc_sock.getsockname()
    remote_ip, remote_port = rcc_sock.getpeername()

    message["backhaul_tcp_source_port"] = local_port
    message["backhaul_tcp_destination_port"] = remote_port

    start = time.perf_counter()

    # Still using the controlled analytical delay.
    # This will be removed only after tc is introduced.
    time.sleep(expected_backhaul_time_s)

    measured = time.perf_counter() - start

    message["backhaul_rate_mbps"] = (
        per_flow_backhaul_rate_mbps
    )
    message["backhaul_transmission_expected_s"] = (
        transmission_time_s
    )
    message["backhaul_fixed_delay_s"] = (
        backhaul_fixed_delay_s
    )
    message["backhaul_expected_s"] = (
        expected_backhaul_time_s
    )
    message["backhaul_measured_s"] = measured
    message["backhaul_payload_bytes_sent"] = len(payload)

    send_frame(
        rcc_sock,
        message,
        payload,
    )

    rcc_sock.close()

    return workload


print(
    f"[SV] Starting SV node. Scenario={scenario}.",
    flush=True,
)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("0.0.0.0", sv_port))
server.listen(num_uavs)

print(
    f"[SV] Waiting for READY from {num_uavs} UAVs.",
    flush=True,
)

connections = []
uav_ids = set()

while len(connections) < num_uavs:
    conn, addr = server.accept()
    stream = conn.makefile("rb")

    ready = read_json_line(stream)

    if ready is None:
        stream.close()
        conn.close()
        continue

    if ready.get("type") != "READY":
        raise RuntimeError(
            "[SV] Expected READY as first UAV message."
        )

    if ready["scenario"] != scenario:
        raise RuntimeError(
            "[SV] Scenario mismatch in READY."
        )

    uav_id = ready["uav_id"]

    if uav_id in uav_ids:
        raise RuntimeError(
            f"[SV] Duplicate READY from UAV {uav_id}."
        )

    uav_ids.add(uav_id)

    connections.append(
        {
            "uav_id": uav_id,
            "socket": conn,
            "stream": stream,
        }
    )

    print(
        f"[SV] READY received from UAV {uav_id}. "
        f"Ready={len(connections)}/{num_uavs}.",
        flush=True,
    )

start_epoch_s = time.time() + START_LEAD_TIME_S

start_message = {
    "type": "START",
    "scenario": scenario,
    "start_epoch_s": start_epoch_s,
}

for connection in connections:
    send_json_line(
        connection["socket"],
        start_message,
    )

print(
    f"[SV] START sent to {num_uavs}/{num_uavs} UAVs.",
    flush=True,
)

with ThreadPoolExecutor(max_workers=num_uavs) as executor:
    workloads = list(
        executor.map(
            receive_workload,
            connections,
        )
    )

for connection in connections:
    connection["stream"].close()
    connection["socket"].close()

workloads.sort(
    key=lambda workload:
        workload["message"]["uav_id"]
)

expected_access_bytes = mbit_to_bytes(
    input_size_mbit
)

total_access_bytes = 0

for workload in workloads:
    message = workload["message"]
    payload = workload["payload"]

    if len(payload) != expected_access_bytes:
        raise RuntimeError(
            f"[SV] UAV {message['uav_id']} payload mismatch: "
            f"expected={expected_access_bytes}, "
            f"received={len(payload)}."
        )

    total_access_bytes += len(payload)

print(
    f"[SV] Access payload total received = "
    f"{total_access_bytes} bytes.",
    flush=True,
)

if scenario == "S1":
    print(
        f"[SV] Starting {num_uavs} concurrent "
        f"inference workers.",
        flush=True,
    )

    batch_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=num_uavs) as executor:
        workloads = list(
            executor.map(
                process_at_sv,
                workloads,
            )
        )

    batch_measured_s = (
        time.perf_counter() - batch_start
    )

    print(
        f"[SV] Batch inference completed. "
        f"expected={expected_compute_time_s:.6f} s, "
        f"measured={batch_measured_s:.6f} s.",
        flush=True,
    )

else:
    print(
        "[SV] S2 selected: preserving full input "
        "payload for RCC.",
        flush=True,
    )

print(
    f"[SV] Starting {num_uavs} independent "
    f"TCP backhaul flows.",
    flush=True,
)

backhaul_batch_start = time.perf_counter()

with ThreadPoolExecutor(max_workers=num_uavs) as executor:
    transmitted_workloads = list(
        executor.map(
            transmit_backhaul,
            workloads,
        )
    )

backhaul_batch_measured_s = (
    time.perf_counter() - backhaul_batch_start
)

total_backhaul_bytes = sum(
    len(workload["payload"])
    for workload in transmitted_workloads
)

for workload in transmitted_workloads:
    message = workload["message"]

    print(
        f"[SV] UAV {message['uav_id']} backhaul completed: "
        f"tcp_source_port="
        f"{message['backhaul_tcp_source_port']}, "
        f"bytes={message['backhaul_payload_bytes_sent']}, "
        f"expected={message['backhaul_expected_s']:.6f} s, "
        f"measured={message['backhaul_measured_s']:.6f} s.",
        flush=True,
    )

print(
    f"[SV] Backhaul payload total sent = "
    f"{total_backhaul_bytes} bytes.",
    flush=True,
)

print(
    f"[SV] Backhaul batch completed in "
    f"{backhaul_batch_measured_s:.6f} s.",
    flush=True,
)

server.close()

print(
    f"[SV] Completed successfully. "
    f"Forwarded {len(transmitted_workloads)}/{num_uavs} "
    f"workloads.",
    flush=True,
)
