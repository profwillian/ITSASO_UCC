import json
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone


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

sv_capacity_gcycles_s = config["computation"]["sv_capacity_gcycles_per_s"]

input_size_bits = input_size_mbit * 1e6
workload_cycles = input_size_bits * c_inf
sv_capacity_cycles_s = sv_capacity_gcycles_s * 1e9

expected_compute_time_s = (
    num_uavs * workload_cycles / sv_capacity_cycles_s
)


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

rcc_sock = connect_with_retry(rcc_host, rcc_port)

print("[SV] Connected to RCC.", flush=True)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("0.0.0.0", sv_port))
server.listen(num_uavs)

print(
    f"[SV] Listening for {num_uavs} UAVs on port {sv_port}.",
    flush=True,
)

workloads = []

while len(workloads) < num_uavs:
    conn, addr = server.accept()

    with conn.makefile("r") as stream:
        line = stream.readline()

    conn.close()

    if not line:
        continue

    message = json.loads(line)

    if message["scenario"] != scenario:
        raise RuntimeError(
            f"[SV] Scenario mismatch: container={scenario}, "
            f"message={message['scenario']}."
        )

    message["sv_received_at"] = timestamp()

    workloads.append(message)

    print(
        f"[SV] Received {message['workload_id']} from UAV "
        f"{message['uav_id']} with "
        f"{message['current_payload_mbit']:.3f} Mbit. "
        f"Batch={len(workloads)}/{num_uavs}.",
        flush=True,
    )

if scenario == "S1":
    print(
        f"[SV] Complete batch received. "
        f"Starting {num_uavs} concurrent inference workers.",
        flush=True,
    )

    batch_start = time.perf_counter()

    for message in workloads:
        message["inference_started_at"] = timestamp()

    with ThreadPoolExecutor(max_workers=num_uavs) as executor:
        processed_workloads = list(
            executor.map(process_at_sv, workloads)
        )

    batch_measured_s = time.perf_counter() - batch_start

    print(
        f"[SV] Batch inference completed. "
        f"expected={expected_compute_time_s:.6f} s, "
        f"measured={batch_measured_s:.6f} s.",
        flush=True,
    )

    for message in processed_workloads:
        message["sv_forwarded_at"] = timestamp()

        rcc_sock.sendall(
            (json.dumps(message) + "\n").encode("utf-8")
        )

        print(
            f"[SV] UAV {message['uav_id']} inference complete: "
            f"expected={message['compute_expected_s']:.6f} s, "
            f"measured={message['compute_measured_s']:.6f} s, "
            f"payload={message['current_payload_mbit']:.3f} Mbit. "
            f"Forwarded to RCC.",
            flush=True,
        )

elif scenario == "S2":
    print(
        "[SV] Complete batch received. "
        "S2 selected: forwarding full workloads to RCC.",
        flush=True,
    )

    for message in workloads:
        message["sv_forwarded_at"] = timestamp()

        rcc_sock.sendall(
            (json.dumps(message) + "\n").encode("utf-8")
        )

        print(
            f"[SV] UAV {message['uav_id']} workload forwarded "
            f"without inference: "
            f"{message['current_payload_mbit']:.3f} Mbit.",
            flush=True,
        )

end_message = {
    "type": "END",
    "scenario": scenario,
    "source": "sv_node",
    "workloads_forwarded": len(workloads),
}

rcc_sock.sendall(
    (json.dumps(end_message) + "\n").encode("utf-8")
)

rcc_sock.close()
server.close()

print(
    f"[SV] Completed successfully. "
    f"Forwarded {len(workloads)}/{num_uavs} workloads.",
    flush=True,
)
