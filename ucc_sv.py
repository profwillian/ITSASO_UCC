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

with open(config_path, "r") as f:
    config = json.load(f)

sv_port = config["nodes"]["sv"]["port"]

rcc_host = config["nodes"]["rcc"]["host"]
rcc_port = config["nodes"]["rcc"]["port"]

expected_uavs = config["experiment"]["num_uavs"]

print(
    f"[SV] Starting SV node. Connecting to RCC "
    f"at {rcc_host}:{rcc_port}.",
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

    message["sv_received_at"] = timestamp()
    message["forwarded_by"] = "sv_node"

    print(
        "[SV] Workload received from UAV: "
        f"uav_id={message['uav_id']} "
        f"workload_id={message['workload_id']}",
        flush=True,
    )

    rcc_sock.sendall(
        (json.dumps(message) + "\n").encode("utf-8")
    )

    print(
        f"[SV] Workload from UAV {message['uav_id']} "
        f"forwarded to RCC.",
        flush=True,
    )

end_message = {
    "type": "END",
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
