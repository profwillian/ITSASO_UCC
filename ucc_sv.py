import json
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from ucc_protocol import (
    mbit_to_bytes,
    read_frame,
    read_json_line,
    send_frame,
    send_json_line,
)

from ucc_tc import configure_netem


START_LEAD_TIME_S = 0.5
BACKHAUL_ACK_TIMEOUT_S = 15.0
BACKHAUL_BARRIER_TIMEOUT_S = 5.0


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def connect_with_retry(host, port, timeout=60):
    deadline = time.time() + timeout

    while time.time() < deadline:
        sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )

        try:
            sock.connect((host, port))
            return sock
        except OSError:
            sock.close()
            time.sleep(0.2)

    raise RuntimeError(
        f"[SV] Could not connect to RCC at "
        f"{host}:{port} within {timeout} seconds."
    )


def receive_workload(connection):
    metadata, payload = read_frame(
        connection["stream"]
    )

    if metadata is None:
        raise RuntimeError(
            f"[SV] UAV {connection['uav_id']} closed "
            f"the connection before sending its workload."
        )

    if metadata.get("type") != "WORKLOAD":
        raise RuntimeError(
            f"[SV] Expected WORKLOAD from "
            f"UAV {connection['uav_id']}."
        )

    metadata["sv_received_at"] = timestamp()
    metadata["sv_received_epoch_s"] = time.time()

    metadata["access_payload_bytes_received"] = len(
        payload
    )

    metadata["access_measured_s"] = (
        metadata["sv_received_epoch_s"]
        - metadata["generated_epoch_s"]
    )

    return {
        "message": metadata,
        "payload": payload,
    }


config_path = os.environ.get(
    "CONFIG",
    "cnf/ucc_config.json",
)

scenario = os.environ.get(
    "SCENARIO",
    "S1",
)

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

input_size_mbit = (
    config["workload"]["input_size_mbit"]
)

mu = (
    config["workload"]["output_input_ratio"]
)

c_inf = (
    config["workload"]
    ["computational_intensity_cycles_per_bit"]
)

sv_capacity_gcycles_s = (
    config["computation"]
    ["sv_capacity_gcycles_per_s"]
)

backhaul_capacity_mbps = (
    config["communication"]
    ["sv_rcc_backhaul_capacity_mbps"]
)

backhaul_fixed_delay_ms = (
    config["communication"]
    ["sv_rcc_fixed_delay_ms"]
)

backhaul_fixed_delay_s = (
    backhaul_fixed_delay_ms / 1000.0
)

per_flow_backhaul_rate_mbps = (
    backhaul_capacity_mbps / num_uavs
)

input_size_bits = (
    input_size_mbit * 1e6
)

workload_cycles = (
    input_size_bits * c_inf
)

sv_capacity_cycles_s = (
    sv_capacity_gcycles_s * 1e9
)

expected_compute_time_s = (
    num_uavs
    * workload_cycles
    / sv_capacity_cycles_s
)


def process_at_sv(workload):
    message = workload["message"]

    start = time.perf_counter()

    # Computation is intentionally model-controlled.
    time.sleep(expected_compute_time_s)

    measured = (
        time.perf_counter() - start
    )

    output_size_mbit = (
        message["input_size_mbit"] * mu
    )

    output_payload = bytes(
        mbit_to_bytes(output_size_mbit)
    )

    message["execution_tier"] = "SV"

    message["compute_expected_s"] = (
        expected_compute_time_s
    )

    message["compute_measured_s"] = (
        measured
    )

    message["current_payload_mbit"] = (
        output_size_mbit
    )

    message["inference_finished_at"] = (
        timestamp()
    )

    return {
        "message": message,
        "payload": output_payload,
    }


print(
    f"[SV] Starting SV node. "
    f"Scenario={scenario}.",
    flush=True,
)


# ---------------------------------------------------------
# BACKHAUL INFRASTRUCTURE SETUP
#
# TCP connections and tc configuration are completed
# before workload generation. Therefore, infrastructure
# setup remains outside the measured E2E latency.
#
# IMPORTANT:
# tc controls ONLY aggregate link capacity.
# The fixed backhaul delay tau_bh is applied later as a
# model-controlled delay outside TCP congestion dynamics.
# ---------------------------------------------------------

setup_start = time.perf_counter()

print(
    f"[SV] Pre-establishing {num_uavs} "
    f"backhaul TCP channels before START.",
    flush=True,
)

backhaul_channels = []

for channel_id in range(1, num_uavs + 1):
    sock = connect_with_retry(
        rcc_host,
        rcc_port,
    )

    sock.settimeout(
        BACKHAUL_ACK_TIMEOUT_S
    )

    local_ip, local_port = (
        sock.getsockname()
    )

    remote_ip, remote_port = (
        sock.getpeername()
    )

    backhaul_channels.append(
        {
            "channel_id": channel_id,
            "socket": sock,
            "local_ip": local_ip,
            "local_port": local_port,
            "remote_ip": remote_ip,
            "remote_port": remote_port,
        }
    )

    print(
        f"[SV] Backhaul channel "
        f"{channel_id}/{num_uavs} established: "
        f"{local_ip}:{local_port} -> "
        f"{remote_ip}:{remote_port}.",
        flush=True,
    )


# Rate-only emulation.
#
# Do NOT insert tau_bh as netem delay here because doing
# so changes TCP dynamics and makes the fixed analytical
# delay interact with congestion control.
tc_result = configure_netem(
    peer_host=rcc_host,
    rate_mbps=backhaul_capacity_mbps,
    delay_ms=0.0,
)

print(
    f"[SV] TC backhaul configured before START: "
    f"interface={tc_result['interface']}, "
    f"aggregate_rate="
    f"{tc_result['rate_mbps']:.3f} Mbit/s, "
    f"netem_delay="
    f"{tc_result['delay_ms']:.3f} ms.",
    flush=True,
)

print(
    f"[SV] Controlled backhaul fixed delay: "
    f"tau_bh={backhaul_fixed_delay_ms:.3f} ms "
    f"(applied during workload execution, "
    f"outside TCP dynamics).",
    flush=True,
)

setup_elapsed_s = (
    time.perf_counter() - setup_start
)

print(
    f"[SV] Backhaul infrastructure ready. "
    f"setup_time={setup_elapsed_s:.6f} s "
    f"(excluded from E2E latency).",
    flush=True,
)


# Barrier synchronizes the beginning of the backhaul phase.
backhaul_barrier = threading.Barrier(
    num_uavs
)


def transmit_backhaul(workload, channel):
    message = workload["message"]
    payload = workload["payload"]

    rcc_sock = channel["socket"]

    payload_mbit = (
        message["current_payload_mbit"]
    )

    model_transmission_s = (
        payload_mbit
        / per_flow_backhaul_rate_mbps
    )

    model_backhaul_s = (
        model_transmission_s
        + backhaul_fixed_delay_s
    )

    message["backhaul_rate_mbps"] = (
        per_flow_backhaul_rate_mbps
    )

    message[
        "backhaul_transmission_expected_s"
    ] = model_transmission_s

    message["backhaul_fixed_delay_s"] = (
        backhaul_fixed_delay_s
    )

    message["backhaul_expected_s"] = (
        model_backhaul_s
    )

    message["backhaul_payload_bytes_sent"] = (
        len(payload)
    )

    message["backhaul_tcp_source_port"] = (
        channel["local_port"]
    )

    message[
        "backhaul_tcp_destination_port"
    ] = channel["remote_port"]

    print(
        f"[SV] UAV {message['uav_id']} "
        f"assigned to pre-established "
        f"backhaul channel "
        f"{channel['channel_id']} "
        f"(source_port={channel['local_port']}).",
        flush=True,
    )

    try:
        backhaul_barrier.wait(
            timeout=BACKHAUL_BARRIER_TIMEOUT_S
        )
    except threading.BrokenBarrierError as exc:
        raise RuntimeError(
            f"[SV] Backhaul synchronization "
            f"failed for UAV "
            f"{message['uav_id']}."
        ) from exc

    # -----------------------------------------------------
    # Beginning of the measured backhaul phase.
    #
    # Keep this field name for compatibility with the RCC.
    # It now marks the beginning of the full analytical
    # backhaul phase:
    #
    #     tau_bh + TCP transmission
    #
    # Therefore RCC's backhaul_measured_s remains directly
    # comparable with backhaul_expected_s.
    # -----------------------------------------------------

    message["backhaul_send_start_epoch_s"] = (
        time.time()
    )

    message[
        "backhaul_phase_start_epoch_s"
    ] = message[
        "backhaul_send_start_epoch_s"
    ]

    fixed_delay_start = (
        time.perf_counter()
    )

    if backhaul_fixed_delay_s > 0:
        time.sleep(
            backhaul_fixed_delay_s
        )

    fixed_delay_measured_s = (
        time.perf_counter()
        - fixed_delay_start
    )

    message[
        "backhaul_fixed_delay_measured_s"
    ] = fixed_delay_measured_s

    message[
        "backhaul_transport_send_start_epoch_s"
    ] = time.time()

    print(
        f"[SV] UAV {message['uav_id']} "
        f"controlled backhaul delay completed: "
        f"expected="
        f"{backhaul_fixed_delay_s:.6f} s, "
        f"measured="
        f"{fixed_delay_measured_s:.6f} s.",
        flush=True,
    )

    # Only the actual payload transfer is governed
    # by TCP + tc rate shaping.
    local_send_start = (
        time.perf_counter()
    )

    send_frame(
        rcc_sock,
        message,
        payload,
    )

    local_send_call_s = (
        time.perf_counter()
        - local_send_start
    )

    message[
        "backhaul_local_send_call_s"
    ] = local_send_call_s

    print(
        f"[SV] UAV {message['uav_id']} "
        f"TCP send returned: "
        f"bytes={len(payload)}, "
        f"local_send_call="
        f"{local_send_call_s:.6f} s. "
        f"Waiting for RCC delivery ACK.",
        flush=True,
    )

    ack_stream = (
        rcc_sock.makefile("rb")
    )

    ack_wait_start = (
        time.perf_counter()
    )

    try:
        ack = read_json_line(
            ack_stream
        )
    except socket.timeout as exc:
        raise RuntimeError(
            f"[SV] Backhaul delivery ACK timeout "
            f"for UAV {message['uav_id']} "
            f"after "
            f"{BACKHAUL_ACK_TIMEOUT_S:.1f} s."
        ) from exc

    ack_wait_s = (
        time.perf_counter()
        - ack_wait_start
    )

    if ack is None:
        raise RuntimeError(
            f"[SV] RCC closed connection "
            f"without delivery ACK for "
            f"UAV {message['uav_id']}."
        )

    if ack.get("type") != "DELIVERY_ACK":
        raise RuntimeError(
            f"[SV] Invalid RCC response for "
            f"UAV {message['uav_id']}: "
            f"{ack.get('type')}."
        )

    if ack.get("scenario") != scenario:
        raise RuntimeError(
            f"[SV] ACK scenario mismatch for "
            f"UAV {message['uav_id']}."
        )

    if ack.get("uav_id") != message["uav_id"]:
        raise RuntimeError(
            f"[SV] ACK UAV mismatch: "
            f"expected={message['uav_id']}, "
            f"received={ack.get('uav_id')}."
        )

    if (
        ack.get("workload_id")
        != message["workload_id"]
    ):
        raise RuntimeError(
            f"[SV] ACK workload mismatch for "
            f"UAV {message['uav_id']}."
        )

    if (
        ack.get("payload_bytes_received")
        != len(payload)
    ):
        raise RuntimeError(
            f"[SV] ACK payload mismatch for UAV "
            f"{message['uav_id']}: "
            f"sent={len(payload)}, "
            f"RCC confirmed="
            f"{ack.get('payload_bytes_received')}."
        )

    message["backhaul_ack_wait_s"] = (
        ack_wait_s
    )

    message["backhaul_ack_received_at"] = (
        timestamp()
    )

    message[
        "rcc_confirmed_received_epoch_s"
    ] = ack["rcc_received_epoch_s"]

    print(
        f"[SV] UAV {message['uav_id']} "
        f"delivery ACK received: "
        f"bytes="
        f"{ack['payload_bytes_received']}, "
        f"ack_wait={ack_wait_s:.6f} s.",
        flush=True,
    )

    ack_stream.close()
    rcc_sock.close()

    return workload


# ---------------------------------------------------------
# UAV ACCESS SIDE
# ---------------------------------------------------------

server = socket.socket(
    socket.AF_INET,
    socket.SOCK_STREAM,
)

server.setsockopt(
    socket.SOL_SOCKET,
    socket.SO_REUSEADDR,
    1,
)

server.bind(
    ("0.0.0.0", sv_port)
)

server.listen(num_uavs)

print(
    f"[SV] Waiting for READY from "
    f"{num_uavs} UAVs.",
    flush=True,
)


connections = []
uav_ids = set()


while len(connections) < num_uavs:
    conn, addr = server.accept()

    stream = conn.makefile("rb")

    ready = read_json_line(
        stream
    )

    if ready is None:
        stream.close()
        conn.close()
        continue

    if ready.get("type") != "READY":
        raise RuntimeError(
            "[SV] Expected READY as "
            "first UAV message."
        )

    if ready["scenario"] != scenario:
        raise RuntimeError(
            "[SV] Scenario mismatch in READY."
        )

    uav_id = ready["uav_id"]

    if uav_id in uav_ids:
        raise RuntimeError(
            f"[SV] Duplicate READY "
            f"from UAV {uav_id}."
        )

    uav_ids.add(uav_id)

    connections.append(
        {
            "uav_id": uav_id,
            "socket": conn,
            "stream": stream,
        }
    )

    print(
        f"[SV] READY received from "
        f"UAV {uav_id}. "
        f"Ready="
        f"{len(connections)}/{num_uavs}.",
        flush=True,
    )


# Infrastructure is already configured at this point.
# The common experiment clock begins only now.

start_epoch_s = (
    time.time()
    + START_LEAD_TIME_S
)

start_message = {
    "type": "START",
    "scenario": scenario,
    "start_epoch_s": start_epoch_s,
}


print(
    f"[SV] Experimental infrastructure ready. "
    f"Scheduling common START at "
    f"{start_epoch_s:.6f}.",
    flush=True,
)


for connection in connections:
    send_json_line(
        connection["socket"],
        start_message,
    )


print(
    f"[SV] START sent to "
    f"{num_uavs}/{num_uavs} UAVs.",
    flush=True,
)


with ThreadPoolExecutor(
    max_workers=num_uavs
) as executor:

    workloads = list(
        executor.map(
            receive_workload,
            connections,
        )
    )


for connection in connections:
    connection["stream"].close()
    connection["socket"].close()


workloads.sort(
    key=lambda workload:
        workload["message"]["uav_id"]
)


expected_access_bytes = (
    mbit_to_bytes(input_size_mbit)
)

total_access_bytes = 0


for workload in workloads:
    message = workload["message"]
    payload = workload["payload"]

    if len(payload) != expected_access_bytes:
        raise RuntimeError(
            f"[SV] UAV {message['uav_id']} "
            f"payload mismatch: "
            f"expected={expected_access_bytes}, "
            f"received={len(payload)}."
        )

    total_access_bytes += len(payload)

    print(
        f"[SV] UAV {message['uav_id']} "
        f"access observed: "
        f"bytes={len(payload)}, "
        f"model="
        f"{message['access_expected_s']:.6f} s, "
        f"observed="
        f"{message['access_measured_s']:.6f} s.",
        flush=True,
    )


print(
    f"[SV] Access payload total received = "
    f"{total_access_bytes} bytes.",
    flush=True,
)


if scenario == "S1":

    print(
        f"[SV] Starting {num_uavs} concurrent "
        f"inference workers.",
        flush=True,
    )

    batch_start = (
        time.perf_counter()
    )

    with ThreadPoolExecutor(
        max_workers=num_uavs
    ) as executor:

        workloads = list(
            executor.map(
                process_at_sv,
                workloads,
            )
        )

    batch_measured_s = (
        time.perf_counter()
        - batch_start
    )

    print(
        f"[SV] Batch inference completed. "
        f"expected="
        f"{expected_compute_time_s:.6f} s, "
        f"measured="
        f"{batch_measured_s:.6f} s.",
        flush=True,
    )

else:

    print(
        "[SV] S2 selected: preserving "
        "full input payload for RCC.",
        flush=True,
    )


print(
    f"[SV] Starting {num_uavs} synchronized "
    f"backhaul transfers using "
    f"pre-established TCP channels.",
    flush=True,
)


with ThreadPoolExecutor(
    max_workers=num_uavs
) as executor:

    transmitted_workloads = list(
        executor.map(
            transmit_backhaul,
            workloads,
            backhaul_channels,
        )
    )


total_backhaul_bytes = sum(
    len(workload["payload"])
    for workload in transmitted_workloads
)


print(
    f"[SV] Backhaul payload total delivered = "
    f"{total_backhaul_bytes} bytes.",
    flush=True,
)


server.close()


print(
    f"[SV] Completed successfully. "
    f"Delivered "
    f"{len(transmitted_workloads)}/{num_uavs} "
    f"workloads with RCC ACK.",
    flush=True,
)
