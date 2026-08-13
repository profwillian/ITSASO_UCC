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

expected_uavs = config["experiment"]["num_uavs"]
mu = config["workload"]["output_input_ratio"]

print(
    f"[SV] Starting SV node. Scenario={scenario}. "
    f"Connecting to RCC at {rcc_host}:{rcc_port}.",
    flush=True,
)

rcc_sock = connect_with_retry(rcc_host, rcc_port)

print("[SV] Connected to RCC.", flush=True)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("0.0.0.0", sv_port))
server.listen(expected_uavs)

print(
    f"[SV] Listening for {expected_uavs} UAVs on port {sv_port}.",
    flush=True,
)

received = 0

while received < expected_uavs:
    conn, addr = server.accept()

    with conn.makefile("r") as stream:
        line = stream.readline()

    conn.close()

    if not line:
        continue

    message = json.loads(line)
    received += 1

    if message["scenario"] != scenario:
        raise RuntimeError(
            f"[SV] Scenario mismatch: container={scenario}, "
            f"message={message['scenario']}."
        )

    message["sv_received_at"] = timestamp()

    print(
        f"[SV] Received {message['workload_id']} from UAV "
        f"{message['uav_id']} with "
        f"{message['current_payload_mbit']:.3f} Mbit.",
        flush=True,
    )

    if scenario == "S1":
        message["inference_started_at"] = timestamp()

        output_size = (
            message["input_size_mbit"] * mu
        )

        message["execution_tier"] = "SV"
        message["current_payload_mbit"] = output_size
        message["inference_finished_at"] = timestamp()

        print(
            f"[SV] INFERENCE executed for UAV "
            f"{message['uav_id']}. "
            f"Payload reduced to {output_size:.3f} Mbit.",
            flush=True,
        )

    elif scenario == "S2":
        print(
            f"[SV] No inference for UAV {message['uav_id']}. "
            f"Forwarding full input payload to RCC.",
            flush=True,
        )

    message["sv_forwarded_at"] = timestamp()

    rcc_sock.sendall(
        (json.dumps(message) + "\n").encode("utf-8")
    )

    print(
        f"[SV] Forwarded {message['current_payload_mbit']:.3f} Mbit "
        f"for UAV {message['uav_id']} to RCC.",
        flush=True,
    )

end_message = {
    "type": "END",
    "scenario": scenario,
    "source": "sv_node",
    "workloads_forwarded": received,
}

rcc_sock.sendall(
    (json.dumps(end_message) + "\n").encode("utf-8")
)

rcc_sock.close()
server.close()

print(
    f"[SV] Completed successfully. "
    f"Forwarded {received}/{expected_uavs} workloads.",
    flush=True,
)
