"""
RCC node for the UCC 2026 maritime inference offloading experiment.

ITSASO mapping:

    Cloud -> Rescue Coordination Center (RCC)

The RCC is the final endpoint of the UAV-SV-RCC communication chain.

Each UAV workload reaches the RCC through an independent TCP connection
opened by the SV. This preserves the notion of concurrent inference flows
sharing the SV-to-RCC backhaul.

At this development stage, no artificial computation delay or network
shaping is applied.
"""

from __future__ import annotations

import json
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any


HEADER_SIZE = 4


def load_config(config_path: str) -> dict[str, Any]:
    """Load the UCC experiment configuration."""

    path = Path(config_path)

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def recv_exact(sock: socket.socket, size: int) -> bytes:
    """Receive exactly *size* bytes from a TCP socket."""

    data = bytearray()

    while len(data) < size:
        chunk = sock.recv(size - len(data))

        if not chunk:
            raise ConnectionError(
                "Socket closed before the expected number of bytes was received."
            )

        data.extend(chunk)

    return bytes(data)


def send_message(
    sock: socket.socket,
    metadata: dict[str, Any],
    payload: bytes = b"",
) -> None:
    """Send a length-prefixed JSON metadata block followed by a payload."""

    metadata = dict(metadata)
    metadata["payload_bytes"] = len(payload)

    metadata_bytes = json.dumps(
        metadata,
        separators=(",", ":"),
    ).encode("utf-8")

    sock.sendall(struct.pack("!I", len(metadata_bytes)))
    sock.sendall(metadata_bytes)

    if payload:
        sock.sendall(payload)


def receive_message(
    sock: socket.socket,
) -> tuple[dict[str, Any], bytes]:
    """Receive one framed message."""

    raw_metadata_size = recv_exact(sock, HEADER_SIZE)
    metadata_size = struct.unpack("!I", raw_metadata_size)[0]

    metadata_bytes = recv_exact(sock, metadata_size)
    metadata = json.loads(metadata_bytes.decode("utf-8"))

    payload_size = int(metadata.get("payload_bytes", 0))

    if payload_size < 0:
        raise ValueError("payload_bytes cannot be negative.")

    payload = recv_exact(sock, payload_size) if payload_size else b""

    return metadata, payload


def expected_payload_bytes(
    config: dict[str, Any],
    scenario: str,
) -> int:
    """Return the expected RCC payload size for the selected scenario."""

    workload = config["workload"]

    input_bits = workload["input_size_mbit"] * 1_000_000.0
    input_bytes = int(round(input_bits / 8.0))

    if scenario == "S1":
        output_ratio = workload["output_ratio"]
        return int(round(input_bytes * output_ratio))

    if scenario == "S2":
        return input_bytes

    raise ValueError(f"Unsupported scenario: {scenario}")


def validate_message(
    config: dict[str, Any],
    metadata: dict[str, Any],
    payload: bytes,
) -> None:
    """Validate one message received from the SV."""

    n_uavs = int(config["experiment"]["n_uavs"])

    required_fields = {
        "type",
        "scenario",
        "uav_id",
        "workload_id",
        "generated_ns",
        "sv_received_ns",
    }

    missing = required_fields.difference(metadata.keys())

    if missing:
        raise ValueError(
            "Missing metadata field(s): "
            + ", ".join(sorted(missing))
        )

    scenario = metadata["scenario"]
    message_type = metadata["type"]
    uav_id = int(metadata["uav_id"])

    if scenario not in {"S1", "S2"}:
        raise ValueError(f"Unsupported scenario: {scenario}")

    if not 1 <= uav_id <= n_uavs:
        raise ValueError(
            f"Invalid UAV id {uav_id}. Expected 1..{n_uavs}."
        )

    expected_type = "result" if scenario == "S1" else "workload"

    if message_type != expected_type:
        raise ValueError(
            f"Scenario {scenario} expects message type "
            f"'{expected_type}', received '{message_type}'."
        )

    expected_size = expected_payload_bytes(config, scenario)

    if len(payload) != expected_size:
        raise ValueError(
            f"Unexpected payload size for UAV {uav_id}: "
            f"received {len(payload)} bytes, expected "
            f"{expected_size} bytes."
        )


def handle_sv_connection(
    conn: socket.socket,
    address: tuple[str, int],
    config: dict[str, Any],
    received_uavs: set[int],
    lock: threading.Lock,
    errors: list[str],
) -> None:
    """Handle one UAV flow arriving from the SV."""

    socket_timeout = float(config["runtime"]["socket_timeout_s"])
    conn.settimeout(socket_timeout)

    try:
        metadata, payload = receive_message(conn)

        validate_message(config, metadata, payload)

        uav_id = int(metadata["uav_id"])
        scenario = metadata["scenario"]
        workload_id = metadata["workload_id"]

        rcc_received_ns = time.time_ns()

        with lock:
            if uav_id in received_uavs:
                raise ValueError(
                    f"Duplicate workload received for UAV {uav_id}."
                )

            received_uavs.add(uav_id)

        print(
            f"RCC received {metadata['type']} "
            f"{workload_id} from UAV {uav_id} via SV | "
            f"scenario={scenario} | "
            f"payload={len(payload)} bytes | "
            f"peer={address[0]}:{address[1]}",
            flush=True,
        )

        ack = {
            "type": "ack",
            "scenario": scenario,
            "uav_id": uav_id,
            "workload_id": workload_id,
            "generated_ns": int(metadata["generated_ns"]),
            "sv_received_ns": int(metadata["sv_received_ns"]),
            "rcc_received_ns": rcc_received_ns,
            "received_payload_bytes": len(payload),
        }

        send_message(conn, ack)

    except Exception as exc:
        with lock:
            errors.append(str(exc))

        print(
            f"RCC connection error from "
            f"{address[0]}:{address[1]}: {exc}",
            file=sys.stderr,
            flush=True,
        )

    finally:
        conn.close()


def run_rcc(config: dict[str, Any]) -> None:
    """Run the RCC TCP endpoint."""

    experiment = config["experiment"]
    network = config["network"]
    runtime = config["runtime"]

    n_uavs = int(experiment["n_uavs"])
    port = int(network["rcc_port"])
    startup_timeout = float(runtime["startup_timeout_s"])

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server.bind(("0.0.0.0", port))
    server.listen(n_uavs)
    server.settimeout(startup_timeout)

    print("=" * 72, flush=True)
    print("UCC 2026 - RCC node", flush=True)
    print(f"Listening on 0.0.0.0:{port}", flush=True)
    print(f"Expected independent SV-RCC flows: {n_uavs}", flush=True)
    print("=" * 72, flush=True)

    received_uavs: set[int] = set()
    errors: list[str] = []
    lock = threading.Lock()
    threads: list[threading.Thread] = []

    try:
        for _ in range(n_uavs):
            conn, address = server.accept()

            thread = threading.Thread(
                target=handle_sv_connection,
                args=(
                    conn,
                    address,
                    config,
                    received_uavs,
                    lock,
                    errors,
                ),
            )

            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        if errors:
            raise RuntimeError(
                "One or more RCC flows failed: "
                + " | ".join(errors)
            )

        if len(received_uavs) != n_uavs:
            raise RuntimeError(
                f"Expected workloads from {n_uavs} UAVs, "
                f"but received {len(received_uavs)}."
            )

        print(
            f"All {n_uavs} UAV workloads reached the RCC.",
            flush=True,
        )

    finally:
        server.close()
        print("RCC node terminated.", flush=True)


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "Usage: python3 cloudNode.py <config.json> [node_id]",
            file=sys.stderr,
        )
        return 2

    config_path = sys.argv[1]

    try:
        config = load_config(config_path)
        run_rcc(config)

    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        json.JSONDecodeError,
    ) as exc:
        print(f"RCC ERROR: {exc}", file=sys.stderr, flush=True)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
