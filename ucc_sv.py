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



START_LEAD_TIME_S = 0.5
BACKHAUL_ACK_TIMEOUT_S = 120.0
BACKHAUL_BARRIER_TIMEOUT_S = 5.0


class WorkConservingBackhaulScheduler:
    """
    Real-time implementation of a work-conserving
    processor-sharing backhaul.

    If q(t) flows are active, every active flow receives

        C_bh / q(t)

    of virtual service capacity.

    The scheduler controls service time only. Actual TCP
    delivery occurs after the corresponding virtual service
    completes and is intentionally left unshaped.
    """

    def __init__(
        self,
        capacity_mbps,
    ):
        self.capacity_mbps = float(
            capacity_mbps
        )

        if self.capacity_mbps <= 0:
            raise ValueError(
                "Backhaul capacity must be positive."
            )

        self.condition = threading.Condition()

        self.active = {}

        self.last_update_s = (
            time.perf_counter()
        )

        self.closed = False

        self.worker = threading.Thread(
            target=self._run,
            name="ucc-backhaul-scheduler",
            daemon=True,
        )

        self.worker.start()


    def _advance_locked(
        self,
        target_s,
    ):
        """
        Advance the fluid server exactly from the previous
        virtual time to target_s.

        More than one flow may finish during the interval,
        so the active set and per-flow rate are recomputed
        after every completion event.
        """

        eps = 1e-12

        if not self.active:
            self.last_update_s = target_s
            return

        while (
            self.active
            and self.last_update_s
            < target_s - eps
        ):
            active_count = len(
                self.active
            )

            per_flow_rate_mbps = (
                self.capacity_mbps
                / active_count
            )

            available_s = (
                target_s
                - self.last_update_s
            )

            time_to_first_finish_s = min(
                state["remaining_mbit"]
                / per_flow_rate_mbps
                for state
                in self.active.values()
            )

            if (
                time_to_first_finish_s
                <= available_s + eps
            ):
                elapsed_s = max(
                    0.0,
                    time_to_first_finish_s,
                )
            else:
                elapsed_s = (
                    available_s
                )

            transmitted_mbit = (
                per_flow_rate_mbps
                * elapsed_s
            )

            for state in self.active.values():
                state["remaining_mbit"] -= (
                    transmitted_mbit
                )

            self.last_update_s += (
                elapsed_s
            )

            if (
                time_to_first_finish_s
                > available_s + eps
            ):
                break

            completed_ids = [
                flow_id
                for flow_id, state
                in self.active.items()
                if (
                    state["remaining_mbit"]
                    <= 1e-9
                )
            ]

            if not completed_ids:
                raise RuntimeError(
                    "Backhaul scheduler reached a "
                    "completion event without a "
                    "completed flow."
                )

            for flow_id in completed_ids:
                state = self.active.pop(
                    flow_id
                )

                state[
                    "virtual_completion_s"
                ] = self.last_update_s

                state["done"].set()

        if not self.active:
            self.last_update_s = (
                target_s
            )


    def serve(
        self,
        flow_id,
        payload_mbit,
        timeout_s,
    ):
        payload_mbit = float(
            payload_mbit
        )

        if payload_mbit <= 0:
            raise ValueError(
                "Backhaul payload must be positive."
            )

        done = threading.Event()

        with self.condition:
            now_s = time.perf_counter()

            self._advance_locked(
                now_s
            )

            if flow_id in self.active:
                raise RuntimeError(
                    f"Backhaul flow {flow_id} "
                    f"is already active."
                )

            state = {
                "flow_id": flow_id,
                "remaining_mbit":
                    payload_mbit,
                "release_s":
                    now_s,
                "virtual_completion_s":
                    None,
                "done":
                    done,
            }

            self.active[
                flow_id
            ] = state

            self.condition.notify_all()

        wait_start_s = (
            time.perf_counter()
        )

        completed = done.wait(
            timeout=timeout_s
        )

        wait_measured_s = (
            time.perf_counter()
            - wait_start_s
        )

        if not completed:
            raise RuntimeError(
                f"Backhaul scheduler timeout "
                f"for flow {flow_id}."
            )

        virtual_service_s = (
            state[
                "virtual_completion_s"
            ]
            - state["release_s"]
        )

        return {
            "flow_id":
                flow_id,
            "virtual_service_s":
                virtual_service_s,
            "wait_measured_s":
                wait_measured_s,
        }


    def _run(self):
        while True:

            with self.condition:
                now_s = (
                    time.perf_counter()
                )

                self._advance_locked(
                    now_s
                )

                if (
                    self.closed
                    and not self.active
                ):
                    return

                if not self.active:
                    self.condition.wait()
                    continue

                active_count = len(
                    self.active
                )

                per_flow_rate_mbps = (
                    self.capacity_mbps
                    / active_count
                )

                next_finish_s = min(
                    state["remaining_mbit"]
                    / per_flow_rate_mbps
                    for state
                    in self.active.values()
                )

                self.condition.wait(
                    timeout=max(
                        next_finish_s,
                        1e-6,
                    )
                )


    def close(self):
        with self.condition:
            if self.active:
                raise RuntimeError(
                    "Cannot close backhaul scheduler "
                    "while flows remain active."
                )

            self.closed = True
            self.condition.notify_all()

        self.worker.join(
            timeout=2.0
        )

        if self.worker.is_alive():
            raise RuntimeError(
                "Backhaul scheduler did not stop."
            )


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

default_input_size_mbit = float(
    config["workload"]["input_size_mbit"]
)

configured_input_sizes_mbit = (
    config["workload"].get(
        "input_sizes_mbit"
    )
)

if configured_input_sizes_mbit is not None:

    if (
        len(configured_input_sizes_mbit)
        != num_uavs
    ):
        raise ValueError(
            "workload.input_sizes_mbit must "
            "contain exactly one value per UAV."
        )

    configured_input_sizes_mbit = [
        float(value)
        for value
        in configured_input_sizes_mbit
    ]

    if any(
        value <= 0
        for value
        in configured_input_sizes_mbit
    ):
        raise ValueError(
            "All workload.input_sizes_mbit "
            "values must be positive."
        )


def input_size_for_uav(uav_id):

    if configured_input_sizes_mbit is None:
        return default_input_size_mbit

    return configured_input_sizes_mbit[
        uav_id - 1
    ]


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

sv_capacity_cycles_s = (
    sv_capacity_gcycles_s * 1e9
)

def resolve_mixed_sv_uav_ids():
    configured_ids = (
        config.get(
            "mixed_offloading",
            {},
        ).get(
            "sv_uav_ids"
        )
    )

    # Backward-compatible default:
    # odd UAV IDs are executed at the SV.
    if configured_ids is None:
        configured_ids = [
            uav_id
            for uav_id in range(
                1,
                num_uavs + 1,
            )
            if uav_id % 2 == 1
        ]

    try:
        sv_ids = sorted(
            int(uav_id)
            for uav_id in configured_ids
        )
    except (TypeError, ValueError):
        raise RuntimeError(
            "mixed_offloading.sv_uav_ids "
            "must contain integer UAV IDs."
        )

    if len(sv_ids) != len(set(sv_ids)):
        raise RuntimeError(
            "mixed_offloading.sv_uav_ids "
            "contains duplicate UAV IDs."
        )

    valid_ids = set(
        range(
            1,
            num_uavs + 1,
        )
    )

    invalid_ids = (
        set(sv_ids) - valid_ids
    )

    if invalid_ids:
        raise RuntimeError(
            "Invalid UAV IDs in "
            "mixed_offloading.sv_uav_ids: "
            f"{sorted(invalid_ids)}"
        )

    if not (
        0 < len(sv_ids) < num_uavs
    ):
        raise RuntimeError(
            "S3 requires at least one workload "
            "at the SV and at least one workload "
            "at the RCC."
        )

    return sv_ids


if scenario == "S1":
    mixed_sv_uav_ids = []
    sv_compute_jobs = num_uavs

elif scenario == "S3":
    mixed_sv_uav_ids = (
        resolve_mixed_sv_uav_ids()
    )

    sv_compute_jobs = len(
        mixed_sv_uav_ids
    )

else:
    mixed_sv_uav_ids = []
    sv_compute_jobs = 0


def expected_sv_compute_time_s(
    input_size_mbit,
):
    workload_cycles = (
        float(input_size_mbit)
        * 1e6
        * c_inf
    )

    return (
        sv_compute_jobs
        * workload_cycles
        / sv_capacity_cycles_s
    )


def process_at_sv(workload):
    message = workload["message"]

    compute_expected_s = (
        expected_sv_compute_time_s(
            message["input_size_mbit"]
        )
    )

    start = time.perf_counter()

    # Computation is intentionally model-controlled.
    time.sleep(compute_expected_s)

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
        compute_expected_s
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


# The SV -> RCC shared backhaul is controlled at the
# application level. This avoids coupling the experiment
# to Linux TCP/qdisc scheduling while preserving the
# work-conserving processor-sharing semantics used by
# the analytical model.
backhaul_scheduler = (
    WorkConservingBackhaulScheduler(
        backhaul_capacity_mbps
    )
)

print(
    f"[SV] Work-conserving backhaul scheduler "
    f"configured before START: "
    f"aggregate_rate="
    f"{backhaul_capacity_mbps:.3f} Mbit/s.",
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


# S1 and S2 use synchronized backhaul transmission.
#
# S3 intentionally does not use the global barrier:
# RCC-bound workloads may start backhaul transmission
# immediately, while SV-bound workloads first complete
# inference at the SV.
if scenario == "S3":
    backhaul_barrier = None
else:
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

    # -----------------------------------------------------
    # Instrument synchronization overhead separately.
    #
    # This interval is part of end-to-end latency but is
    # intentionally outside backhaul_measured_s.
    # -----------------------------------------------------

    message["backhaul_barrier_wait_s"] = 0.0
    message["backhaul_barrier_enter_epoch_s"] = None
    message["backhaul_barrier_exit_epoch_s"] = None

    if backhaul_barrier is not None:

        message[
            "backhaul_barrier_enter_epoch_s"
        ] = time.time()

        barrier_wait_start = time.perf_counter()

        try:
            backhaul_barrier.wait(
                timeout=BACKHAUL_BARRIER_TIMEOUT_S
            )
        except threading.BrokenBarrierError as exc:

            message[
                "backhaul_barrier_wait_s"
            ] = (
                time.perf_counter()
                - barrier_wait_start
            )

            message[
                "backhaul_barrier_exit_epoch_s"
            ] = time.time()

            raise RuntimeError(
                f"[SV] Backhaul synchronization "
                f"failed for UAV "
                f"{message['uav_id']}."
            ) from exc

        message[
            "backhaul_barrier_wait_s"
        ] = (
            time.perf_counter()
            - barrier_wait_start
        )

        message[
            "backhaul_barrier_exit_epoch_s"
        ] = time.time()

        print(
            f"[SV] UAV {message['uav_id']} "
            f"backhaul barrier wait="
            f"{message['backhaul_barrier_wait_s']:.6f} s.",
            flush=True,
        )

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

    message["sv_post_access_wait_s"] = (
        message["backhaul_send_start_epoch_s"]
        - message["sv_received_epoch_s"]
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
        "backhaul_transport_release_epoch_s"
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

    # -----------------------------------------------------
    # Shared backhaul service.
    #
    # The real-time scheduler independently implements the
    # fluid work-conserving policy. TCP is used only to
    # deliver the payload after the corresponding amount
    # of shared-link service has been consumed.
    # -----------------------------------------------------

    scheduler_result = (
        backhaul_scheduler.serve(
            flow_id=message["uav_id"],
            payload_mbit=payload_mbit,
            timeout_s=BACKHAUL_ACK_TIMEOUT_S,
        )
    )

    message[
        "backhaul_scheduler_virtual_service_s"
    ] = scheduler_result[
        "virtual_service_s"
    ]

    message[
        "backhaul_scheduler_wait_measured_s"
    ] = scheduler_result[
        "wait_measured_s"
    ]

    print(
        f"[SV] UAV {message['uav_id']} "
        f"shared backhaul service completed: "
        f"virtual="
        f"{scheduler_result['virtual_service_s']:.6f} s, "
        f"measured_wait="
        f"{scheduler_result['wait_measured_s']:.6f} s.",
        flush=True,
    )

    message[
        "backhaul_transport_send_start_epoch_s"
    ] = time.time()

    # Actual Docker-network delivery is unshaped and is
    # expected to contribute only a small implementation
    # overhead after virtual service completion.
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


total_access_bytes = 0


for workload in workloads:
    message = workload["message"]
    payload = workload["payload"]

    configured_input_size_mbit = (
        input_size_for_uav(
            message["uav_id"]
        )
    )

    declared_input_size_mbit = float(
        message["input_size_mbit"]
    )

    if abs(
        declared_input_size_mbit
        - configured_input_size_mbit
    ) > 1e-9:
        raise RuntimeError(
            f"[SV] UAV {message['uav_id']} "
            f"declared input size "
            f"{declared_input_size_mbit} Mbit, "
            f"but configuration expects "
            f"{configured_input_size_mbit} Mbit."
        )

    expected_access_bytes = (
        mbit_to_bytes(
            declared_input_size_mbit
        )
    )

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



backhaul_already_sent = False


if scenario == "S1":

    for workload in workloads:
        workload["message"][
            "offload_target"
        ] = "SV"

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

    expected_batch_compute_s = max(
        float(
            workload["message"][
                "compute_expected_s"
            ]
        )
        for workload
        in workloads
    )

    print(
        f"[SV] Batch inference completed. "
        f"expected_max="
        f"{expected_batch_compute_s:.6f} s, "
        f"measured="
        f"{batch_measured_s:.6f} s.",
        flush=True,
    )


elif scenario == "S2":

    for workload in workloads:
        message = workload["message"]

        message[
            "offload_target"
        ] = "RCC"

        message[
            "current_payload_mbit"
        ] = message[
            "input_size_mbit"
        ]

    print(
        "[SV] S2 selected: preserving "
        "full input payload for RCC.",
        flush=True,
    )


elif scenario == "S3":

    sv_jobs = []
    rcc_jobs = []

    sv_id_set = set(
        mixed_sv_uav_ids
    )

    for workload in workloads:
        message = workload["message"]

        if message["uav_id"] in sv_id_set:
            message[
                "offload_target"
            ] = "SV"

            sv_jobs.append(
                message["uav_id"]
            )

        else:
            message[
                "offload_target"
            ] = "RCC"

            message[
                "current_payload_mbit"
            ] = message[
                "input_size_mbit"
            ]

            rcc_jobs.append(
                message["uav_id"]
            )

    expected_sv_jobs = len(
        mixed_sv_uav_ids
    )

    expected_rcc_jobs = (
        num_uavs
        - expected_sv_jobs
    )

    if (
        len(sv_jobs)
        != expected_sv_jobs
        or len(rcc_jobs)
        != expected_rcc_jobs
    ):
        raise RuntimeError(
            "[SV] S3 workload assignment "
            "does not match the configured "
            "SV/RCC split."
        )

    rho_sv = (
        len(sv_jobs)
        / num_uavs
    )

    print(
        f"[SV] S3 mixed offloading selected. "
        f"SV jobs={sv_jobs}; "
        f"RCC jobs={rcc_jobs}; "
        f"rho_sv={rho_sv:.2f}.",
        flush=True,
    )

    print(
        f"[SV] SV compute share: "
        f"{len(sv_jobs)} jobs. "
        f"Per-workload compute time follows "
        f"the corresponding input size.",
        flush=True,
    )

    def execute_mixed_path(
        workload,
        channel,
    ):
        message = workload["message"]

        if (
            message["offload_target"]
            == "SV"
        ):
            workload = process_at_sv(
                workload
            )

        return transmit_backhaul(
            workload,
            channel,
        )

    print(
        "[SV] Starting mixed SV/RCC paths. "
        "RCC-bound jobs can enter the "
        "backhaul immediately; SV-bound jobs "
        "first complete inference.",
        flush=True,
    )

    with ThreadPoolExecutor(
        max_workers=num_uavs
    ) as executor:

        workloads = list(
            executor.map(
                execute_mixed_path,
                workloads,
                backhaul_channels,
            )
        )

    backhaul_already_sent = True


else:
    raise RuntimeError(
        f"[SV] Unsupported scenario: "
        f"{scenario}"
    )


if not backhaul_already_sent:

    print(
        f"[SV] Starting {num_uavs} synchronized "
        f"backhaul transfers using "
        f"pre-established TCP channels.",
        flush=True,
    )

    with ThreadPoolExecutor(
        max_workers=num_uavs
    ) as executor:

        workloads = list(
            executor.map(
                transmit_backhaul,
                workloads,
                backhaul_channels,
            )
        )


total_backhaul_bytes = sum(
    len(workload["payload"])
    for workload in workloads
)


print(
    f"[SV] Backhaul payload total delivered = "
    f"{total_backhaul_bytes} bytes.",
    flush=True,
)


backhaul_scheduler.close()

print(
    "[SV] Work-conserving backhaul scheduler "
    "completed with no active flows.",
    flush=True,
)


server.close()


print(
    f"[SV] Completed successfully. "
    f"Delivered "
    f"{len(workloads)}/{num_uavs} "
    f"workloads with RCC ACK.",
    flush=True,
)
