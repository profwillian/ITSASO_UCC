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
    "generated_epoch_s": time.time(),
}

print(
    f"[UAV {uav_id}] Scenario={scenario}. "
    f"Generated {message['workload_id']} "
    f"with {input_size_mbit:.3f} Mbit.",
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

sock = connect_with_retry(sv_host, sv_port)

sock.sendall(
    (json.dumps(message) + "\n").encode("utf-8")
)

sock.close()

print(
    f"[UAV {uav_id}] Access completed: "
    f"expected={expected_access_time_s:.6f} s, "
    f"measured={access_measured_s:.6f} s. "
    f"Workload delivered to SV.",
    flush=True,
)
