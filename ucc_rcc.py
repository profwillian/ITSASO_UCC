import json
import os
import socket
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
expected_uavs = config["experiment"]["num_uavs"]

mu = config["workload"]["output_input_ratio"]

print(
    f"[RCC] Starting RCC node on port {port}. "
    f"Scenario={scenario}. Waiting for SV connection.",
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

        received += 1
        message["rcc_received_at"] = timestamp()

        print(
            f"[RCC] Received {message['workload_id']} "
            f"with {message['current_payload_mbit']:.3f} Mbit.",
            flush=True,
        )

        if scenario == "S1":
            if message["execution_tier"] != "SV":
                raise RuntimeError(
                    "[RCC] S1 workload did not execute at the SV."
                )

            print(
                f"[RCC] Inference result from SV accepted for UAV "
                f"{message['uav_id']}.",
                flush=True,
            )

        elif scenario == "S2":
            message["inference_started_at"] = timestamp()

            output_size = (
                message["input_size_mbit"] * mu
            )

            message["execution_tier"] = "RCC"
            message["current_payload_mbit"] = output_size
            message["inference_finished_at"] = timestamp()

            print(
                f"[RCC] INFERENCE executed for UAV "
                f"{message['uav_id']}. "
                f"Result size={output_size:.3f} Mbit.",
                flush=True,
            )

        print(
            f"[RCC] Workload complete: "
            f"uav_id={message['uav_id']} "
            f"execution_tier={message['execution_tier']}.",
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
