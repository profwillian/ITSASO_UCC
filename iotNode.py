"""
Monitoring UAV node for the UCC 2026 maritime inference offloading experiment.

ITSASO mapping:

    IoT device -> Monitoring UAV

Each UAV generates exactly one video inference workload during the
representative monitoring interval and sends the complete video chunk
to the Surface Vessel (SV).

The UAV never executes inference locally and has no direct connection
to the Rescue Coordination Center (RCC).

At this development stage, no network shaping is applied to the UAV-SV
link. The configured effective transmission rate R_n^(1) is retained as
an experiment parameter and will be enforced in a later step.
"""

from __future__ import annotations

import json
import socket
import struct
import sys
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


def expected_rcc_payload_bytes(
    config: dict[str, Any],
    scenario: str,
) -> int:
    """Return the expected payload size that should reach the RCC."""

    input_bytes = input_payload_bytes(config)

    if scenario == "S1":
        output_ratio = float(config["workload"]["output_ratio"])
        return int(round(input_bytes * output_ratio))

    if scenario == "S2":
        return input_bytes

    raise ValueError(f"Unsupported scenario: {scenario}")


def validate_uav_id(
    config: dict[str, Any],
    uav_id: int,
) -> None:
    """Validate the UAV identifier against N."""

    n_uavs = int(config["experiment"]["n_uavs"])

    if not 1 <= uav_id <= n_uavs:
        raise ValueError(
            f"Invalid UAV id {uav_id}. Expected a value from 1 to {n_uavs}."
        )

    rates = config["communication"]["uav_sv_rates_mbps"]

    if len(rates) != n_uavs:
        raise ValueError(
            "The number of UAV-to-SV transmission rates must match "
            f"N. Received N={n_uavs} and {len(rates)} rates."
        )


def connect_to_sv(
    config: dict[str, Any],
) -> socket.socket:
    """Connect the monitoring UAV to the Surface Vessel."""

    network = config["network"]
    runtime = config["runtime"]

    host = network["sv_host"]
    port = int(network["sv_port"])

    startup_timeout = float(runtime["startup_timeout_s"])
    socket_timeout = float(runtime["socket_timeout_s"])

    deadline = time.monotonic() + startup_timeout
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(socket_timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        try:
            sock.connect((host, port))
            return sock

        except OSError as exc:
            last_error = exc
            sock.close()
            time.sleep(0.25)

    raise ConnectionError(
        f"Unable to connect to SV at {host}:{port} "
        f"within {startup_timeout} seconds. "
        f"Last error: {last_error}"
    )


def validate_ack(
    config: dict[str, Any],
    ack: dict[str, Any],
    payload: bytes,
    uav_id: int,
    workload_id: str,
    scenario: str,
) -> None:
    """Validate the final acknowledgement returned through the SV."""

    if payload:
        raise ValueError(
            "Final acknowledgement must not contain a binary payload."
        )

    required_fields = {
        "type",
        "scenario",
        "uav_id",
        "workload_id",
        "generated_ns",
        "sv_received_ns",
        "rcc_received_ns",
        "sv_ack_received_ns",
        "rcc_received_payload_bytes",
    }

    missing = required_fields.difference(ack.keys())

    if missing:
        raise ValueError(
            "Missing ACK metadata field(s): "
            + ", ".join(sorted(missing))
        )

    if ack["type"] != "ack":
        raise ValueError(
            f"Unexpected final response type: {ack['type']}"
        )

    if ack["scenario"] != scenario:
        raise ValueError(
            "ACK scenario does not match the transmitted scenario."
        )

    if int(ack["uav_id"]) != uav_id:
        raise ValueError(
            "ACK UAV id does not match the originating UAV."
        )

    if ack["workload_id"] != workload_id:
        raise ValueError(
            "ACK workload id does not match the transmitted workload."
        )

    expected_size = expected_rcc_payload_bytes(config, scenario)
    received_size = int(ack["rcc_received_payload_bytes"])

    if received_size != expected_size:
        raise ValueError(
            f"RCC payload size mismatch: received {received_size} bytes, "
            f"expected {expected_size} bytes."
        )


def run_uav(
    config: dict[str, Any],
    uav_id: int,
    scenario: str,
) -> None:
    """Generate and transmit one video inference workload."""

    validate_uav_id(config, uav_id)

    if scenario not in {"S1", "S2"}:
        raise ValueError(
            f"Unsupported scenario '{scenario}'. Expected S1 or S2."
        )

    communication = config["communication"]
    network = config["network"]

    configured_rate = float(
        communication["uav_sv_rates_mbps"][uav_id - 1]
    )

    payload_size = input_payload_bytes(config)
    payload = bytes(payload_size)

    workload_id = f"uav{uav_id}_chunk1"

    print("=" * 72, flush=True)
    print(f"UCC 2026 - Monitoring UAV {uav_id}", flush=True)
    print(f"Scenario:               {scenario}", flush=True)
    print(f"SV destination:         {network['sv_host']}:{network['sv_port']}", flush=True)
    print(f"Video workload:         {payload_size} bytes", flush=True)
    print(
        f"Configured R_{uav_id}^(1):     "
        f"{configured_rate} Mbit/s",
        flush=True,
    )
    print("=" * 72, flush=True)

    sock = connect_to_sv(config)

    try:
        generated_ns = time.time_ns()

        metadata = {
            "type": "workload",
            "scenario": scenario,
            "uav_id": uav_id,
            "workload_id": workload_id,
            "generated_ns": generated_ns,
        }

        send_message(
            sock,
            metadata,
            payload,
        )

        print(
            f"UAV {uav_id} sent workload {workload_id} "
            f"to SV | payload={payload_size} bytes",
            flush=True,
        )

        ack, ack_payload = receive_message(sock)

        validate_ack(
            config,
            ack,
            ack_payload,
            uav_id,
            workload_id,
            scenario,
        )

        ack_received_ns = time.time_ns()

        measured_to_rcc_s = (
            int(ack["rcc_received_ns"]) - generated_ns
        ) / 1_000_000_000.0

        full_round_trip_s = (
            ack_received_ns - generated_ns
        ) / 1_000_000_000.0

        print(
            f"UAV {uav_id} completed workload {workload_id}.",
            flush=True,
        )

        print(
            f"Observed generation-to-RCC time: "
            f"{measured_to_rcc_s:.6f} s",
            flush=True,
        )

        print(
            f"Observed full ACK round trip:    "
            f"{full_round_trip_s:.6f} s",
            flush=True,
        )

    finally:
        sock.close()

        print(
            f"Monitoring UAV {uav_id} terminated.",
            flush=True,
        )


def main() -> int:
    if len(sys.argv) < 4:
        print(
            "Usage: python3 iotNode.py <config.json> <uav_id> <S1|S2>",
            file=sys.stderr,
        )
        return 2

    config_path = sys.argv[1]
    uav_id = int(sys.argv[2])
    scenario = sys.argv[3].upper()

    try:
        config = load_config(config_path)

        run_uav(
            config,
            uav_id,
            scenario,
        )

    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        json.JSONDecodeError,
    ) as exc:
        print(
            f"UAV {uav_id} ERROR: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
