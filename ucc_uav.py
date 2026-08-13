import json
import os
import socket
import time
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
        f"[UAV] Could not connect to SV at {host}:{port} "
        f"within {timeout} seconds."
    )


config_path = os.environ.get("CONFIG", "cnf/ucc_config.json")
scenario = os.environ.get("SCENARIO", "S1")
uav_id = int(os.environ["UAV_ID"])

with open(config_path, "r") as f:
    config = json.load(f)

valid_scenarios = config["experiment"]["scenarios"]

if scenario not in valid_scenarios:
    raise ValueError(
        f"Scenario {scenario} is not valid. "
        f"Available scenarios: {valid_scenarios}"
    )

num_uavs = config["experiment"]["num_uavs"]

if not 1 <= uav_id <= num_uavs:
    raise ValueError(
        f"UAV_ID={uav_id} is incompatible with "
        f"num_uavs={num_uavs}."
    )

rates = config["communication"]["uav_sv_rates_mbps"]

if len(rates) != num_uavs:
    raise ValueError(
        "The number of UAV-to-SV access rates must be equal "
        "to experiment.num_uavs."
    )

sv_host = config["nodes"]["sv"]["host"]
sv_port = config["nodes"]["sv"]["port"]

input_size_mbit = config["workload"]["input_size_mbit"]
access_rate_mbps = rates[uav_id - 1]

expected_access_time_s = (
    input_size_mbit / access_rate_mbps
)

sock = connect_with_retry(sv_host, sv_port)
stream = sock.makefile("r")

ready_message = {
    "type": "READY",
    "scenario": scenario,
    "uav_id": uav_id,
}

sock.sendall(
    (json.dumps(ready_message) + "\n").encode("utf-8")
)

print(
    f"[UAV {uav_id}] READY sent to SV. "
    f"Waiting for common START.",
    flush=True,
)

line = stream.readline()

if not line:
    raise RuntimeError(
        f"[UAV {uav_id}] SV closed the connection before START."
    )

start_message = json.loads(line)

if start_message.get("type") != "START":
    raise RuntimeError(
        f"[UAV {uav_id}] Expected START message, "
        f"received {start_message.get('type')}."
    )

if start_message["scenario"] != scenario:
    raise RuntimeError(
        f"[UAV {uav_id}] Scenario mismatch in START message."
    )

start_epoch_s = start_message["start_epoch_s"]

print(
    f"[UAV {uav_id}] START received. "
    f"Common generation epoch={start_epoch_s:.6f}.",
    flush=True,
)

remaining_s = start_epoch_s - time.time()

if remaining_s > 0:
    time.sleep(remaining_s)

generated_epoch_s = time.time()
generation_skew_s = generated_epoch_s - start_epoch_s

message = {
    "type": "WORKLOAD",
    "scenario": scenario,
    "uav_id": uav_id,
    "workload_id": f"uav{uav_id}-workload1",
    "input_size_mbit": input_size_mbit,
    "current_payload_mbit": input_size_mbit,
    "configured_access_rate_mbps": access_rate_mbps,
    "execution_tier": None,
    "generated_at": timestamp(),
    "generated_epoch_s": generated_epoch_s,
    "common_start_epoch_s": start_epoch_s,
    "generation_skew_s": generation_skew_s,
}

print(
    f"[UAV {uav_id}] Generated {message['workload_id']} "
    f"at synchronized start. "
    f"skew={generation_skew_s * 1000:.3f} ms.",
    flush=True,
)

print(
    f"[UAV {uav_id}] Access transmission: "
    f"rate={access_rate_mbps:.3f} Mbit/s, "
    f"expected={expected_access_time_s:.6f} s.",
    flush=True,
)

access_start = time.perf_counter()

time.sleep(expected_access_time_s)

access_measured_s = time.perf_counter() - access_start

message["access_expected_s"] = expected_access_time_s
message["access_measured_s"] = access_measured_s
message["access_finished_at"] = timestamp()

sock.sendall(
    (json.dumps(message) + "\n").encode("utf-8")
)

print(
    f"[UAV {uav_id}] Access completed: "
    f"expected={expected_access_time_s:.6f} s, "
    f"measured={access_measured_s:.6f} s. "
    f"Workload delivered to SV.",
    flush=True,
)

stream.close()
sock.close()
