import csv
import json
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from ucc_protocol import (
    mbit_to_bytes,
    read_frame,
    send_json_line,
)


def timestamp():
    return datetime.now(timezone.utc).isoformat()



def model_s3_work_conserving_backhaul(
    config,
    workloads,
):
    """
    Analytical model for S3 mixed offloading.

    The SV-to-RCC backhaul is represented as a
    work-conserving processor-sharing server.

    Only currently active flows share C_bh.

    RCC-bound jobs:
        access barrier
        -> tau_bh
        -> raw payload over backhaul
        -> RCC inference

    SV-bound jobs:
        access barrier
        -> SV inference
        -> tau_bh
        -> reduced payload over backhaul

    Times are relative to the common experiment start.
    """

    cbh_mbps = float(
        config[
            "communication"
        ][
            "sv_rcc_backhaul_capacity_mbps"
        ]
    )

    tau_s = (
        float(
            config[
                "communication"
            ][
                "sv_rcc_fixed_delay_ms"
            ]
        )
        / 1000.0
    )

    input_size_mbit = float(
        config[
            "workload"
        ][
            "input_size_mbit"
        ]
    )

    mu = float(
        config[
            "workload"
        ][
            "output_input_ratio"
        ]
    )

    if cbh_mbps <= 0:
        raise RuntimeError(
            "Backhaul capacity must be positive."
        )

    if not workloads:
        raise RuntimeError(
            "S3 analytical model received "
            "no workloads."
        )

    # -----------------------------------------------------
    # The current containerized implementation waits until
    # the complete access batch has arrived at the SV
    # before the mixed execution paths are started.
    # -----------------------------------------------------

    access_barrier_s = max(
        float(
            message["access_expected_s"]
        )
        for message in workloads
    )

    flows = []

    phase_start_s = {}
    post_backhaul_compute_s = {}
    access_wait_s = {}

    for message in workloads:

        uav_id = int(
            message["uav_id"]
        )

        target = message.get(
            "offload_target"
        )

        access_s = float(
            message["access_expected_s"]
        )

        access_wait_s[uav_id] = (
            access_barrier_s
            - access_s
        )

        if target == "RCC":

            # RCC-bound jobs enter the backhaul
            # immediately after the access batch barrier.
            path_start_s = (
                access_barrier_s
            )

            payload_mbit = (
                input_size_mbit
            )

            post_compute_s = float(
                message[
                    "compute_expected_s"
                ]
            )

        elif target == "SV":

            # SV-bound jobs first complete inference.
            path_start_s = (
                access_barrier_s
                + float(
                    message[
                        "compute_expected_s"
                    ]
                )
            )

            payload_mbit = (
                input_size_mbit
                * mu
            )

            post_compute_s = 0.0

        else:
            raise RuntimeError(
                f"Invalid S3 offload target "
                f"for UAV {uav_id}: "
                f"{target}."
            )

        phase_start_s[uav_id] = (
            path_start_s
        )

        post_backhaul_compute_s[
            uav_id
        ] = post_compute_s

        # tau_bh does not consume link capacity.
        # Therefore the TCP flow becomes active only
        # after its controlled fixed delay.
        transport_release_s = (
            path_start_s
            + tau_s
        )

        flows.append(
            {
                "uav_id": uav_id,
                "release_s":
                    transport_release_s,
                "remaining_mbit":
                    payload_mbit,
            }
        )

    # -----------------------------------------------------
    # Fluid processor-sharing model.
    #
    # At every instant:
    #
    #     R_i(t) = C_bh / m(t)
    #
    # where m(t) is the number of currently active flows.
    # -----------------------------------------------------

    pending = sorted(
        flows,
        key=lambda flow:
            (
                flow["release_s"],
                flow["uav_id"],
            ),
    )

    active = {}

    transport_completion_s = {}

    eps = 1e-12

    current_s = (
        pending[0]["release_s"]
    )

    while pending or active:

        # Activate all flows released at current time.
        while (
            pending
            and pending[0]["release_s"]
            <= current_s + eps
        ):
            flow = pending.pop(0)

            active[
                flow["uav_id"]
            ] = float(
                flow["remaining_mbit"]
            )

        # Jump to next release if no flow is active.
        if not active:
            current_s = (
                pending[0]["release_s"]
            )
            continue

        active_count = len(active)

        per_flow_rate_mbps = (
            cbh_mbps
            / active_count
        )

        time_to_first_finish_s = min(
            remaining_mbit
            / per_flow_rate_mbps
            for remaining_mbit
            in active.values()
        )

        first_finish_s = (
            current_s
            + time_to_first_finish_s
        )

        if pending:
            next_release_s = (
                pending[0]["release_s"]
            )
        else:
            next_release_s = float(
                "inf"
            )

        # A new flow becomes active before any
        # currently active flow completes.
        if (
            next_release_s
            < first_finish_s - eps
        ):

            elapsed_s = (
                next_release_s
                - current_s
            )

            transmitted_mbit = (
                per_flow_rate_mbps
                * elapsed_s
            )

            for uav_id in list(
                active.keys()
            ):
                active[uav_id] -= (
                    transmitted_mbit
                )

            current_s = (
                next_release_s
            )

            continue

        # One or more flows complete before the
        # next release event.
        elapsed_s = (
            first_finish_s
            - current_s
        )

        transmitted_mbit = (
            per_flow_rate_mbps
            * elapsed_s
        )

        for uav_id in list(
            active.keys()
        ):
            active[uav_id] -= (
                transmitted_mbit
            )

        current_s = (
            first_finish_s
        )

        completed = [
            uav_id
            for uav_id, remaining
            in active.items()
            if remaining <= 1e-9
        ]

        if not completed:
            raise RuntimeError(
                "S3 processor-sharing model "
                "failed to complete a flow."
            )

        for uav_id in completed:

            transport_completion_s[
                uav_id
            ] = current_s

            del active[uav_id]

    # -----------------------------------------------------
    # Build per-UAV model decomposition.
    # -----------------------------------------------------

    model = {}

    for message in workloads:

        uav_id = int(
            message["uav_id"]
        )

        transport_done_s = (
            transport_completion_s[
                uav_id
            ]
        )

        backhaul_s = (
            transport_done_s
            - phase_start_s[uav_id]
        )

        completion_s = (
            transport_done_s
            + post_backhaul_compute_s[
                uav_id
            ]
        )

        model[uav_id] = {
            "access_barrier_s":
                access_barrier_s,
            "access_wait_s":
                access_wait_s[uav_id],
            "backhaul_s":
                backhaul_s,
            "completion_s":
                completion_s,
        }

    return model

def save_results(
    config,
    scenario,
    workloads,
):
    results_dir = (
        config["experiment"]["results_dir"]
    )

    os.makedirs(
        results_dir,
        exist_ok=True,
    )

    rows = []

    if scenario == "S3":
        s3_model = (
            model_s3_work_conserving_backhaul(
                config,
                workloads,
            )
        )
    else:
        s3_model = None

    for message in sorted(
        workloads,
        key=lambda x: x["uav_id"],
    ):
        access_model_s = (
            message["access_expected_s"]
        )

        access_emulated_s = (
            message["access_measured_s"]
        )

        compute_model_s = (
            message["compute_expected_s"]
        )

        compute_emulated_s = (
            message["compute_measured_s"]
        )

        if s3_model is not None:

            model_row = s3_model[
                message["uav_id"]
            ]

            access_wait_model_s = (
                model_row[
                    "access_wait_s"
                ]
            )

            backhaul_model_s = (
                model_row[
                    "backhaul_s"
                ]
            )

            t_model_s = (
                model_row[
                    "completion_s"
                ]
            )

        else:

            access_wait_model_s = 0.0

            backhaul_model_s = (
                message[
                    "backhaul_expected_s"
                ]
            )

            t_model_s = (
                access_model_s
                + compute_model_s
                + backhaul_model_s
            )

        backhaul_emulated_s = (
            message["backhaul_measured_s"]
        )

        t_emulated_s = (
            message["completion_epoch_s"]
            - message["common_start_epoch_s"]
        )

        error_s = (
            t_emulated_s
            - t_model_s
        )

        error_pct = (
            100.0
            * error_s
            / t_model_s
            if t_model_s > 0
            else 0.0
        )

        rows.append(
            {
                "scenario": scenario,
                "uav_id":
                    message["uav_id"],
                "execution_tier":
                    message["execution_tier"],
                "access_rate_mbps":
                    message[
                        "configured_access_rate_mbps"
                    ],
                "generation_skew_ms":
                    message["generation_skew_s"]
                    * 1000.0,
                "sv_post_access_wait_s":
                    message.get(
                        "sv_post_access_wait_s",
                        0.0,
                    ),
                "rcc_pre_compute_wait_s":
                    message.get(
                        "rcc_pre_compute_wait_s",
                        0.0,
                    ),
                "backhaul_barrier_wait_s":
                    message.get(
                        "backhaul_barrier_wait_s",
                        0.0,
                    ),
                "access_model_s":
                    access_model_s,
                "access_barrier_wait_model_s":
                    access_wait_model_s,
                "access_emulated_s":
                    access_emulated_s,
                "compute_model_s":
                    compute_model_s,
                "compute_emulated_s":
                    compute_emulated_s,
                "backhaul_model_s":
                    backhaul_model_s,
                "backhaul_emulated_s":
                    backhaul_emulated_s,
                "t_model_s":
                    t_model_s,
                "t_emulated_s":
                    t_emulated_s,
                "error_s":
                    error_s,
                "error_pct":
                    error_pct,
                "backhaul_tcp_source_port":
                    message[
                        "backhaul_tcp_source_port"
                    ],
            }
        )

    tmax_model_row = max(
        rows,
        key=lambda row:
            row["t_model_s"],
    )

    tmax_emulated_row = max(
        rows,
        key=lambda row:
            row["t_emulated_s"],
    )

    tmax_model_s = (
        tmax_model_row["t_model_s"]
    )

    tmax_emulated_s = (
        tmax_emulated_row["t_emulated_s"]
    )

    tmax_error_s = (
        tmax_emulated_s
        - tmax_model_s
    )

    tmax_error_pct = (
        100.0
        * tmax_error_s
        / tmax_model_s
    )

    csv_path = os.path.join(
        results_dir,
        f"{scenario}_results.csv",
    )

    with open(
        csv_path,
        "w",
        newline="",
    ) as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "experiment":
            config["experiment"]["name"],
        "scenario":
            scenario,
        "analytical_backhaul_model":
            (
                "work_conserving_processor_sharing"
                if scenario == "S3"
                else "static_equal_sharing"
            ),
        "num_uavs":
            config["experiment"]["num_uavs"],
        "tmax_model_s":
            tmax_model_s,
        "tmax_emulated_s":
            tmax_emulated_s,
        "tmax_error_s":
            tmax_error_s,
        "tmax_error_pct":
            tmax_error_pct,
        "tmax_model_uav":
            tmax_model_row["uav_id"],
        "tmax_emulated_uav":
            tmax_emulated_row["uav_id"],
        "results_csv":
            csv_path,
    }

    json_path = os.path.join(
        results_dir,
        f"{scenario}_summary.json",
    )

    with open(
        json_path,
        "w",
    ) as jsonfile:

        json.dump(
            summary,
            jsonfile,
            indent=2,
        )

    return rows, summary


def receive_backhaul(connection):
    conn = connection["socket"]
    addr = connection["addr"]

    stream = conn.makefile("rb")

    try:
        message, payload = read_frame(
            stream
        )

        received_epoch_s = time.time()

        if message is None:
            raise RuntimeError(
                "[RCC] Backhaul connection closed "
                "before workload was received."
            )

        if message.get("type") != "WORKLOAD":
            raise RuntimeError(
                "[RCC] Expected WORKLOAD "
                "on backhaul flow."
            )

        if message["scenario"] != scenario:
            raise RuntimeError(
                f"[RCC] Scenario mismatch: "
                f"container={scenario}, "
                f"message={message['scenario']}."
            )

        expected_payload_bytes = (
            mbit_to_bytes(
                message["current_payload_mbit"]
            )
        )

        if len(payload) != expected_payload_bytes:
            raise RuntimeError(
                f"[RCC] Payload mismatch for UAV "
                f"{message['uav_id']}: "
                f"expected={expected_payload_bytes}, "
                f"received={len(payload)}."
            )

        message[
            "backhaul_payload_bytes_received"
        ] = len(payload)

        message[
            "rcc_observed_tcp_source_port"
        ] = addr[1]

        message["rcc_received_at"] = (
            timestamp()
        )

        message["rcc_received_epoch_s"] = (
            received_epoch_s
        )

        message["backhaul_measured_s"] = (
            received_epoch_s
            - message[
                "backhaul_send_start_epoch_s"
            ]
        )

        ack = {
            "type": "DELIVERY_ACK",
            "scenario": scenario,
            "uav_id": message["uav_id"],
            "workload_id": message["workload_id"],
            "payload_bytes_received": len(payload),
            "rcc_received_epoch_s": received_epoch_s,
        }

        send_json_line(
            conn,
            ack,
        )

        print(
            f"[RCC] Delivery ACK sent for "
            f"UAV {message['uav_id']}: "
            f"bytes={len(payload)}.",
            flush=True,
        )

        # -------------------------------------------------
        # Start RCC execution as soon as the complete input
        # payload is available.
        #
        # S2 and RCC-bound S3 workloads therefore follow
        # the same execution semantics.
        # -------------------------------------------------

        if scenario == "S2":

            message["rcc_batch_wait_s"] = 0.0

            message = process_at_rcc(
                message
            )

        elif scenario == "S3":

            target = message.get(
                "offload_target"
            )

            if target == "RCC":

                message = process_at_rcc(
                    message
                )

            elif target == "SV":

                if (
                    message.get(
                        "execution_tier"
                    )
                    != "SV"
                ):
                    raise RuntimeError(
                        f"[RCC] S3 workload from UAV "
                        f"{message['uav_id']} was "
                        f"assigned to SV but was not "
                        f"processed there."
                    )

                message[
                    "completion_epoch_s"
                ] = message[
                    "rcc_received_epoch_s"
                ]

            else:
                raise RuntimeError(
                    f"[RCC] S3 workload from UAV "
                    f"{message['uav_id']} has invalid "
                    f"offload target: {target}."
                )

        return message

    finally:
        stream.close()
        conn.close()


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


valid_scenarios = (
    config["experiment"]["scenarios"]
)


if scenario not in valid_scenarios:
    raise ValueError(
        f"Scenario {scenario} is not valid. "
        f"Available scenarios: {valid_scenarios}"
    )


host = "0.0.0.0"

port = (
    config["nodes"]["rcc"]["port"]
)

num_uavs = (
    config["experiment"]["num_uavs"]
)

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

rcc_capacity_gcycles_s = (
    config["computation"]
    ["rcc_capacity_gcycles_per_s"]
)


input_size_bits = (
    input_size_mbit * 1e6
)

workload_cycles = (
    input_size_bits * c_inf
)

rcc_capacity_cycles_s = (
    rcc_capacity_gcycles_s * 1e9
)

if scenario == "S2":
    rcc_compute_jobs = num_uavs

elif scenario == "S3":
    if num_uavs % 2 != 0:
        raise RuntimeError(
            "S3 mixed 50/50 offloading requires "
            "an even number of UAVs."
        )

    rcc_compute_jobs = (
        num_uavs // 2
    )

else:
    rcc_compute_jobs = 0


expected_compute_time_s = (
    rcc_compute_jobs
    * workload_cycles
    / rcc_capacity_cycles_s
)


def process_at_rcc(message):
    message["inference_started_at"] = (
        timestamp()
    )

    message["inference_start_epoch_s"] = (
        time.time()
    )

    message["rcc_pre_compute_wait_s"] = (
        message["inference_start_epoch_s"]
        - message["rcc_received_epoch_s"]
    )

    start = time.perf_counter()

    # Computation remains model-controlled.
    time.sleep(
        expected_compute_time_s
    )

    measured = (
        time.perf_counter()
        - start
    )

    output_size_mbit = (
        message["input_size_mbit"]
        * mu
    )

    message["execution_tier"] = "RCC"

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

    message["completion_epoch_s"] = (
        time.time()
    )

    return message


print(
    f"[RCC] Starting RCC node "
    f"on port {port}. "
    f"Scenario={scenario}.",
    flush=True,
)


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
    (host, port)
)

server.listen(num_uavs)


print(
    f"[RCC] Waiting for {num_uavs} "
    f"independent backhaul TCP flows.",
    flush=True,
)


connections = []


while len(connections) < num_uavs:
    conn, addr = server.accept()

    connections.append(
        {
            "socket": conn,
            "addr": addr,
        }
    )

    print(
        f"[RCC] Accepted backhaul TCP flow "
        f"{len(connections)}/{num_uavs} "
        f"from {addr[0]}:{addr[1]}.",
        flush=True,
    )


with ThreadPoolExecutor(
    max_workers=num_uavs
) as executor:

    workloads = list(
        executor.map(
            receive_backhaul,
            connections,
        )
    )


server.close()


workloads.sort(
    key=lambda message:
        message["uav_id"]
)


for message in workloads:

    print(
        f"[RCC] UAV {message['uav_id']} "
        f"backhaul observed: "
        f"source_port="
        f"{message['rcc_observed_tcp_source_port']}, "
        f"bytes="
        f"{message['backhaul_payload_bytes_received']}, "
        f"model="
        f"{message['backhaul_expected_s']:.6f} s, "
        f"observed="
        f"{message['backhaul_measured_s']:.6f} s.",
        flush=True,
    )

    if (
        message[
            "rcc_observed_tcp_source_port"
        ]
        != message[
            "backhaul_tcp_source_port"
        ]
    ):
        raise RuntimeError(
            f"[RCC] TCP source-port mismatch "
            f"for UAV {message['uav_id']}."
        )


total_backhaul_bytes_received = sum(
    message[
        "backhaul_payload_bytes_received"
    ]
    for message in workloads
)


print(
    f"[RCC] Backhaul payload total received = "
    f"{total_backhaul_bytes_received} bytes.",
    flush=True,
)


if scenario == "S1":

    for message in workloads:

        if message["execution_tier"] != "SV":
            raise RuntimeError(
                f"[RCC] S1 workload from UAV "
                f"{message['uav_id']} "
                f"was not processed at SV."
            )

        message["completion_epoch_s"] = (
            message["rcc_received_epoch_s"]
        )


elif scenario == "S2":

    # All S2 workloads were processed immediately after
    # their complete payload arrived at the RCC.
    for message in workloads:

        if (
            message.get(
                "execution_tier"
            )
            != "RCC"
        ):
            raise RuntimeError(
                f"[RCC] S2 workload from UAV "
                f"{message['uav_id']} was not "
                f"processed at RCC."
            )

    print(
        f"[RCC] S2 inference complete: "
        f"RCC={len(workloads)}.",
        flush=True,
    )


elif scenario == "S3":

    sv_count = 0
    rcc_count = 0

    for message in workloads:

        target = message.get(
            "offload_target"
        )

        if target == "SV":
            sv_count += 1

            if (
                message.get(
                    "execution_tier"
                )
                != "SV"
            ):
                raise RuntimeError(
                    f"[RCC] Invalid S3 SV workload "
                    f"for UAV {message['uav_id']}."
                )

        elif target == "RCC":
            rcc_count += 1

            if (
                message.get(
                    "execution_tier"
                )
                != "RCC"
            ):
                raise RuntimeError(
                    f"[RCC] Invalid S3 RCC workload "
                    f"for UAV {message['uav_id']}."
                )

        else:
            raise RuntimeError(
                f"[RCC] Invalid S3 target for UAV "
                f"{message['uav_id']}: "
                f"{target}."
            )

    if (
        sv_count != num_uavs // 2
        or rcc_count != num_uavs // 2
    ):
        raise RuntimeError(
            "[RCC] S3 result does not contain "
            "an exact 50/50 SV/RCC split."
        )

    print(
        f"[RCC] S3 mixed inference complete: "
        f"SV={sv_count}, RCC={rcc_count}.",
        flush=True,
    )


rows, summary = save_results(
    config,
    scenario,
    workloads,
)


print(
    "[RCC] ===== UCC END-TO-END RESULTS =====",
    flush=True,
)


for row in rows:

    print(
        f"[RCC] UAV {row['uav_id']} | "
        f"T_model={row['t_model_s']:.6f} s | "
        f"T_emulated={row['t_emulated_s']:.6f} s | "
        f"error={row['error_pct']:+.3f}%",
        flush=True,
    )


print(
    f"[RCC] T_max model    = "
    f"{summary['tmax_model_s']:.6f} s "
    f"(UAV {summary['tmax_model_uav']})",
    flush=True,
)


print(
    f"[RCC] T_max emulated = "
    f"{summary['tmax_emulated_s']:.6f} s "
    f"(UAV {summary['tmax_emulated_uav']})",
    flush=True,
)


print(
    f"[RCC] T_max error    = "
    f"{summary['tmax_error_s']:+.6f} s "
    f"({summary['tmax_error_pct']:+.3f}%)",
    flush=True,
)


print(
    f"[RCC] Completed successfully. "
    f"Received {len(workloads)}/{num_uavs} workloads.",
    flush=True,
)
