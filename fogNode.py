"""
Surface Vessel (SV) node for the UCC 2026 maritime inference
offloading experiment.

ITSASO mapping:

    Fog -> Surface Vessel (SV)

The SV is the mandatory communication gateway between the monitoring UAVs
and the Rescue Coordination Center (RCC).

Scenario S1:
    UAV sends the full video workload D to the SV.
    The SV represents inference execution and sends only the compact
    result mu*D to the RCC.

Scenario S2:
    UAV sends the full video workload D to the SV.
    The SV forwards the complete workload D to the RCC, where inference
    will be executed.

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


def input_payload_bytes(config: dict[str, Any]) -> int:
    """Return the configured video chunk size in bytes."""

    input_mbit = float(config["workload"]["input_size_mbit"])
    input_bits = input_mbit * 1_000_000.0

    return int(round(input_bits / 8.0))


def output_payload_bytes(config: dict[str, Any]) -> int:
    """Return the post-inference result size in bytes."""

    input_bytes = input_payload_bytes(config)
    output_ratio = float(config["workload"]["output_ratio"])

    return int(round(input_bytes * output_ratio))


def validate_uav_message(
    config: dict[str, Any],
    metadata: dict[str, Any],
    payload: bytes,
) -> None:
    """Validate one workload received from a UAV."""

    n_uavs = int(config["experiment"]["n_uavs"])

    required_fields = {
        "type",
        "scenario",
        "uav_id",
        "workload_id",
        "generated_ns",
    }

    missing = required_fields.difference(metadata.keys())

    if missing:
        raise ValueError(
            "Missing UAV metadata field(s): "
            + ", ".join(sorted(missing))
        )

    if metadata["type"] != "workload":
        raise ValueError(
            "The SV expects message type 'workload' from UAVs."
        )

    scenario = metadata["scenario"]

    if scenario not in {"S1", "S2"}:
        raise ValueError(f"Unsupported scenario: {scenario}")

    uav_id = int(metadata["uav_id"])

    if not 1 <= uav_id <= n_uavs:
        raise ValueError(
            f"Invalid UAV id {uav_id}. Expected 1..{n_uavs}."
        )

    expected_size = input_payload_bytes(config)

    if len(payload) != expected_size:
        raise ValueError(
            f"Unexpected workload size from UAV {uav_id}: "
            f"received {len(payload)} bytes, "
            f"expected {expected_size} bytes."
        )


def connect_to_rcc(
    config: dict[str, Any],
) -> socket.socket:
    """Open one independent SV-RCC TCP flow."""

    network = config["network"]
    runtime = config["runtime"]

    host = network["rcc_host"]
    port = int(network["rcc_port"])

    startup_timeout = float(runtime["startup_timeout_s"])
    socket_timeout = float(runtime["socket_timeout_s"])

    deadline = time.monotonic() + startup_timeout
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(socket_timeout)

        try:
            sock.connect((host, port))
            return sock

        except OSError as exc:
            last_error = exc
            sock.close()
            time.sleep(0.25)

    raise ConnectionError(
        f"Unable to connect to RCC at {host}:{port} "
        f"within {startup_timeout} seconds. "
        f"Last error: {last_error}"
    )


def build_rcc_message(
    config: dict[str, Any],
    metadata: dict[str, Any],
    payload: bytes,
    sv_received_ns: int,
) -> tuple[dict[str, Any], bytes]:
    """
    Build the message that the SV sends to the RCC.

    S1 sends a compact result of size mu*D.
    S2 forwards the complete workload of size D.
    """

    scenario = metadata["scenario"]

    forwarded_metadata = {
        "scenario": scenario,
        "uav_id": int(metadata["uav_id"]),
        "workload_id": metadata["workload_id"],
        "generated_ns": int(metadata["generated_ns"]),
        "sv_received_ns": sv_received_ns,
    }

    if scenario == "S1":
        result_size = output_payload_bytes(config)

        forwarded_metadata["type"] = "result"

        # At this stage inference content is abstracted.
        # Only the resulting payload size matters.
        forwarded_payload = bytes(result_size)

        return forwarded_metadata, forwarded_payload

    forwarded_metadata["type"] = "workload"

    return forwarded_metadata, payload


def handle_uav_connection(
    conn: socket.socket,
    address: tuple[str, int],
    config: dict[str, Any],
    completed_uavs: set[int],
    lock: threading.Lock,
    errors: list[str],
) -> None:
    """Handle the complete path of one UAV workload through the SV."""

    socket_timeout = float(config["runtime"]["socket_timeout_s"])
    conn.settimeout(socket_timeout)

    rcc_sock = None

    try:
        metadata, payload = receive_message(conn)

        validate_uav_message(config, metadata, payload)

        uav_id = int(metadata["uav_id"])
        scenario = metadata["scenario"]
        workload_id = metadata["workload_id"]

        sv_received_ns = time.time_ns()

        print(
            f"SV received workload {workload_id} "
            f"from UAV {uav_id} | "
            f"scenario={scenario} | "
            f"payload={len(payload)} bytes | "
            f"peer={address[0]}:{address[1]}",
            flush=True,
        )

        rcc_metadata, rcc_payload = build_rcc_message(
            config,
            metadata,
            payload,
            sv_received_ns,
        )

        rcc_sock = connect_to_rcc(config)

        send_message(
            rcc_sock,
            rcc_metadata,
            rcc_payload,
        )

        print(
            f"SV forwarded {rcc_metadata['type']} "
            f"{workload_id} from UAV {uav_id} to RCC | "
            f"payload={len(rcc_payload)} bytes",
            flush=True,
        )

        rcc_ack, ack_payload = receive_message(rcc_sock)

        if ack_payload:
            raise ValueError(
                "RCC ACK must not contain a binary payload."
            )

        if rcc_ack.get("type") != "ack":
            raise ValueError(
                f"Unexpected RCC response type: "
                f"{rcc_ack.get('type')}"
            )

        if int(rcc_ack["uav_id"]) != uav_id:
            raise ValueError(
                "RCC ACK UAV id does not match the originating UAV."
            )

        if rcc_ack["workload_id"] != workload_id:
            raise ValueError(
                "RCC ACK workload id does not match the originating workload."
            )

        sv_ack_received_ns = time.time_ns()

        final_ack = {
            "type": "ack",
            "scenario": scenario,
            "uav_id": uav_id,
            "workload_id": workload_id,
            "generated_ns": int(metadata["generated_ns"]),
            "sv_received_ns": sv_received_ns,
            "rcc_received_ns": int(rcc_ack["rcc_received_ns"]),
            "sv_ack_received_ns": sv_ack_received_ns,
            "rcc_received_payload_bytes":
                int(rcc_ack["received_payload_bytes"]),
        }

        send_message(conn, final_ack)

        with lock:
            if uav_id in completed_uavs:
                raise ValueError(
                    f"Duplicate completed workload for UAV {uav_id}."
                )

            completed_uavs.add(uav_id)

        print(
            f"SV completed workload {workload_id} "
            f"for UAV {uav_id}.",
            flush=True,
        )

    except Exception as exc:
        with lock:
            errors.append(str(exc))

        print(
            f"SV connection error from "
            f"{address[0]}:{address[1]}: {exc}",
            file=sys.stderr,
            flush=True,
        )

    finally:
        if rcc_sock is not None:
            rcc_sock.close()

        conn.close()


def run_sv(config: dict[str, Any]) -> None:
    """Run the Surface Vessel TCP gateway."""

    experiment = config["experiment"]
    network = config["network"]
    runtime = config["runtime"]

    n_uavs = int(experiment["n_uavs"])
    port = int(network["sv_port"])
    startup_timeout = float(runtime["startup_timeout_s"])

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server.bind(("0.0.0.0", port))
    server.listen(n_uavs)
    server.settimeout(startup_timeout)

    print("=" * 72, flush=True)
    print("UCC 2026 - Surface Vessel node", flush=True)
    print(f"Listening for UAVs on 0.0.0.0:{port}", flush=True)
    print(f"Expected UAV flows: {n_uavs}", flush=True)
    print(
        f"RCC destination: "
        f"{network['rcc_host']}:{network['rcc_port']}",
        flush=True,
    )
    print("=" * 72, flush=True)

    completed_uavs: set[int] = set()
    errors: list[str] = []
    lock = threading.Lock()
    threads: list[threading.Thread] = []

    try:
        for _ in range(n_uavs):
            conn, address = server.accept()

            thread = threading.Thread(
                target=handle_uav_connection,
                args=(
                    conn,
                    address,
                    config,
                    completed_uavs,
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
                "One or more SV flows failed: "
                + " | ".join(errors)
            )

        if len(completed_uavs) != n_uavs:
            raise RuntimeError(
                f"Expected {n_uavs} completed UAV workloads, "
                f"but completed {len(completed_uavs)}."
            )

        print(
            f"All {n_uavs} UAV workloads crossed the SV successfully.",
            flush=True,
        )

    finally:
        server.close()
        print("SV node terminated.", flush=True)


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "Usage: python3 fogNode.py <config.json> [node_id]",
            file=sys.stderr,
        )
        return 2

    config_path = sys.argv[1]

    try:
        config = load_config(config_path)
        run_sv(config)

    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        json.JSONDecodeError,
    ) as exc:
        print(f"SV ERROR: {exc}", file=sys.stderr, flush=True)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
