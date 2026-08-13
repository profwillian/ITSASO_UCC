import json
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone


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
            time.sleep(1)

    raise RuntimeError(
        f"[SV] Could not connect to RCC at {host}:{port} "
        f"within {timeout} seconds."
    )


def receive_workload(connection):
    conn = connection["socket"]
    stream = connection["stream"]

    line = stream.readline()

    if not line:
        raise RuntimeError(
            f"[SV] UAV {connection['uav_id']} closed "
            f"the connection before sending its workload."
        )

    message = json.loads(line)

    if message.get("type") != "WORKLOAD":
        raise RuntimeError(
            f"[SV] Expected WORKLOAD from UAV "
            f"{connection['uav_id']}."
        )

    message["sv_received_at"] = timestamp()
    message["sv_received_epoch_s"] = time.time()

    return message


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

send_lock = threading.Lock()


def process_at_sv(message):
    start = time.perf_counter()

    time.sleep(expected_compute_time_s)

    measured = time.perf_counter() - start

    output_size_mbit = (
        message["input_size_mbit"] * mu
    )

    message["execution_tier"] = "SV"
    message["compute_expected_s"] = expected_compute_time_s
    message["compute_measured_s"] = measured
    message["current_payload_mbit"] = output_size_mbit
    message["inference_finished_at"] = timestamp()

    return message


def transmit_backhaul(message, rcc_sock):
    payload_mbit = message["current_payload_mbit"]

    transmission_time_s = (
        payload_mbit / per_flow_backhaul_rate_mbps
    )

    expected_backhaul_time_s = (
        transmission_time_s + backhaul_fixed_delay_s
    )

    start = time.perf_counter()

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
    message["rcc_delivery_epoch_s"] = time.time()

    encoded = (json.dumps(message) + "\n").encode("utf-8")

    with send_lock:
        rcc_sock.sendall(encoded)

    return message


print(
    f"[SV] Starting SV node. Scenario={scenario}. "
    f"Connecting to RCC at {rcc_host}:{rcc_port}.",
    flush=True,
)

print(
    f"[SV] Computation configuration: "
    f"W={workload_cycles:.0f} cycles, "
    f"capacity={sv_capacity_gcycles_s:.3f} Gcycles/s, "
    f"N={num_uavs}, "
    f"expected_per_workload={expected_compute_time_s:.6f} s.",
    flush=True,
)

print(
    f"[SV] Backhaul configuration: "
    f"Cbh={backhaul_capacity_mbps:.3f} Mbit/s, "
    f"per_flow={per_flow_backhaul_rate_mbps:.6f} Mbit/s, "
    f"fixed_delay={backhaul_fixed_delay_s:.6f} s.",
    flush=True,
)

rcc_sock = connect_with_retry(rcc_host, rcc_port)

print("[SV] Connected to RCC.", flush=True)

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
    stream = conn.makefile("r")

    line = stream.readline()

    if not line:
        stream.close()
        conn.close()
        continue

    ready = json.loads(line)

    if ready.get("type") != "READY":
        raise RuntimeError(
            "[SV] Expected READY as first UAV message."
        )

    if ready["scenario"] != scenario:
        raise RuntimeError(
            f"[SV] Scenario mismatch in READY from "
            f"UAV {ready['uav_id']}."
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

encoded_start = (
    json.dumps(start_message) + "\n"
).encode("utf-8")

print(
    f"[SV] All UAVs READY. "
    f"Common START scheduled at {start_epoch_s:.6f} "
    f"(lead={START_LEAD_TIME_S:.3f} s).",
    flush=True,
)

for connection in connections:
    connection["socket"].sendall(encoded_start)

print(
    f"[SV] START sent to {num_uavs}/{num_uavs} UAVs.",
    flush=True,
)

with ThreadPoolExecutor(max_workers=num_uavs) as executor:
    workloads = list(
        executor.map(receive_workload, connections)
    )

for connection in connections:
    connection["stream"].close()
    connection["socket"].close()

workloads.sort(key=lambda message: message["uav_id"])

for message in workloads:
    if message["scenario"] != scenario:
        raise RuntimeError(
            f"[SV] Scenario mismatch in workload from "
            f"UAV {message['uav_id']}."
        )

    print(
        f"[SV] Received {message['workload_id']} from UAV "
        f"{message['uav_id']}. "
        f"generation_skew="
        f"{message['generation_skew_s'] * 1000:.3f} ms, "
        f"access_expected={message['access_expected_s']:.6f} s, "
        f"access_measured={message['access_measured_s']:.6f} s.",
        flush=True,
    )

if scenario == "S1":
    print(
        f"[SV] Complete synchronized batch received. "
        f"Starting {num_uavs} concurrent inference workers.",
        flush=True,
    )

    batch_start = time.perf_counter()

    for message in workloads:
        message["inference_started_at"] = timestamp()

    with ThreadPoolExecutor(max_workers=num_uavs) as executor:
        workloads = list(
            executor.map(process_at_sv, workloads)
        )

    batch_measured_s = time.perf_counter() - batch_start

    print(
        f"[SV] Batch inference completed. "
        f"expected={expected_compute_time_s:.6f} s, "
        f"measured={batch_measured_s:.6f} s.",
        flush=True,
    )

else:
    print(
        "[SV] Complete synchronized batch received. "
        "S2 selected: no inference at SV.",
        flush=True,
    )

print(
    f"[SV] Starting {num_uavs} concurrent backhaul flows.",
    flush=True,
)

backhaul_batch_start = time.perf_counter()

with ThreadPoolExecutor(max_workers=num_uavs) as executor:
    futures = [
        executor.submit(
            transmit_backhaul,
            message,
            rcc_sock
        )
        for message in workloads
    ]

    transmitted_workloads = [
        future.result() for future in futures
    ]

backhaul_batch_measured_s = (
    time.perf_counter() - backhaul_batch_start
)

for message in transmitted_workloads:
    print(
        f"[SV] UAV {message['uav_id']} backhaul completed: "
        f"payload={message['current_payload_mbit']:.3f} Mbit, "
        f"expected={message['backhaul_expected_s']:.6f} s, "
        f"measured={message['backhaul_measured_s']:.6f} s.",
        flush=True,
    )

print(
    f"[SV] Backhaul batch completed in "
    f"{backhaul_batch_measured_s:.6f} s.",
    flush=True,
)

end_message = {
    "type": "END",
    "scenario": scenario,
    "source": "sv_node",
    "workloads_forwarded": len(transmitted_workloads),
}

rcc_sock.sendall(
    (json.dumps(end_message) + "\n").encode("utf-8")
)

rcc_sock.close()
server.close()

print(
    f"[SV] Completed successfully. "
    f"Forwarded {len(transmitted_workloads)}/{num_uavs} workloads.",
    flush=True,
)
