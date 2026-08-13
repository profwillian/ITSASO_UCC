import json
import os
import socket
from datetime import datetime, timezone


def timestamp():
    return datetime.now(timezone.utc).isoformat()


config_path = os.environ.get("CONFIG", "cnf/ucc_config.json")

with open(config_path, "r") as f:
    config = json.load(f)

host = "0.0.0.0"
port = config["nodes"]["rcc"]["port"]
expected_uavs = config["experiment"]["num_uavs"]

print(
    f"[RCC] Starting RCC node on port {port}. "
    f"Waiting for SV connection.",
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

received = 0

with conn.makefile("r") as stream:
    for line in stream:
        message = json.loads(line)

        if message.get("type") == "END":
            print("[RCC] End-of-experiment signal received.", flush=True)
            break

        received += 1
        message["rcc_received_at"] = timestamp()

        print(
            "[RCC] Workload received: "
            f"uav_id={message['uav_id']} "
            f"workload_id={message['workload_id']} "
            f"payload_size_mbit={message['payload_size_mbit']}",
            flush=True,
        )

conn.close()
server.close()

if received != expected_uavs:
    raise RuntimeError(
        f"[RCC] Expected {expected_uavs} workloads, "
        f"but received {received}."
    )

print(
    f"[RCC] Completed successfully. "
    f"Received {received}/{expected_uavs} workloads.",
    flush=True,
)
