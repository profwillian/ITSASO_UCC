"""Mathematical model for the UCC maritime simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass


class ModelValidationError(ValueError):
    """Raised when a model input or derived result is invalid."""


@dataclass(frozen=True, slots=True)
class InferenceWorkload:
    """Resolved representation of one homogeneous inference job."""

    chunks_per_job: int
    chunk_size_bits: float
    input_payload_bits: float
    output_input_ratio: float
    output_payload_bits: float
    compute_intensity_cycles_per_bit: float
    workload_cycles: float


def _require_positive_finite(value: float, field_name: str) -> float:
    converted = float(value)

    if not math.isfinite(converted) or converted <= 0.0:
        raise ModelValidationError(
            f"{field_name} must be finite and greater than zero. "
            f"Received: {value!r}"
        )

    return converted

def _require_non_negative_finite(
    value: float,
    field_name: str,
) -> float:
    """Validate a finite numeric value that may be zero."""
    converted = float(value)

    if not math.isfinite(converted) or converted < 0.0:
        raise ModelValidationError(
            f"{field_name} must be finite and non-negative. "
            f"Received: {value!r}"
        )

    return converted


def _require_positive_int(
    value: int,
    field_name: str,
) -> int:
    """Validate a strictly positive integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ModelValidationError(
            f"{field_name} must be an integer. Received: {value!r}"
        )

    if value <= 0:
        raise ModelValidationError(
            f"{field_name} must be greater than zero. Received: {value}"
        )

    return value


def _compute_equal_share(
    total_resource: float,
    number_of_shares: int,
    field_name: str,
) -> float:
    """Divide one positive resource equally among concurrent users."""
    total = _require_positive_finite(
        total_resource,
        field_name,
    )

    shares = _require_positive_int(
        number_of_shares,
        "number_of_shares",
    )

    result = total / shares

    if not math.isfinite(result) or result <= 0.0:
        raise ModelValidationError(
            f"Equal sharing of {field_name} produced an invalid result."
        )

    return result


def build_inference_workload(
    chunks_per_job: int,
    chunk_size_bits: float,
    output_input_ratio: float,
    compute_intensity_cycles_per_bit: float,
) -> InferenceWorkload:
    """
    Build a homogeneous inference workload.

    The model follows:

        P = Q * D
        P_out = mu * P
        W = P * C_inf
    """
    if isinstance(chunks_per_job, bool) or not isinstance(chunks_per_job, int):
        raise ModelValidationError(
            "chunks_per_job must be an integer."
        )

    if chunks_per_job <= 0:
        raise ModelValidationError(
            "chunks_per_job must be greater than zero."
        )

    chunk_size = _require_positive_finite(
        chunk_size_bits,
        "chunk_size_bits",
    )

    ratio = float(output_input_ratio)

    if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ModelValidationError(
            "output_input_ratio must satisfy 0 < value <= 1. "
            f"Received: {output_input_ratio!r}"
        )

    compute_intensity = _require_positive_finite(
        compute_intensity_cycles_per_bit,
        "compute_intensity_cycles_per_bit",
    )

    input_payload = chunks_per_job * chunk_size
    output_payload = ratio * input_payload
    workload = input_payload * compute_intensity

    if output_payload > input_payload:
        raise ModelValidationError(
            "output_payload_bits cannot exceed input_payload_bits."
        )

    if not all(
        math.isfinite(value) and value > 0.0
        for value in (input_payload, output_payload, workload)
    ):
        raise ModelValidationError(
            "Workload derivation produced a non-finite or non-positive value."
        )

    return InferenceWorkload(
        chunks_per_job=chunks_per_job,
        chunk_size_bits=chunk_size,
        input_payload_bits=input_payload,
        output_input_ratio=ratio,
        output_payload_bits=output_payload,
        compute_intensity_cycles_per_bit=compute_intensity,
        workload_cycles=workload,
    )
@dataclass(frozen=True, slots=True)
class AccessLinkState:
    """Resolved physical state of one UAV-to-SV access link."""

    uav_id: int
    distance_m: float
    channel_gain_linear: float
    snr_linear: float
    rate_bps: float
def compute_channel_gain(
    distance_m: float,
    reference_distance_m: float,
    reference_gain_linear: float,
) -> float:
    """
    Compute the UAV-to-SV channel gain.

    The model follows:

        g = rho_0 * (d / d_0)^(-2)
    """
    distance = _require_positive_finite(
        distance_m,
        "distance_m",
    )

    reference_distance = _require_positive_finite(
        reference_distance_m,
        "reference_distance_m",
    )

    reference_gain = _require_positive_finite(
        reference_gain_linear,
        "reference_gain_linear",
    )

    gain = reference_gain * math.pow(
        distance / reference_distance,
        -2.0,
    )

    if not math.isfinite(gain) or gain <= 0.0:
        raise ModelValidationError(
            "Channel gain calculation produced an invalid result."
        )

    return gain


def compute_snr_linear(
    transmit_power_w: float,
    channel_gain_linear: float,
    noise_psd_w_hz: float,
    allocated_bandwidth_hz: float,
) -> float:
    """
    Compute the linear signal-to-noise ratio.

    The model follows:

        SNR = p * g / (N_0 * B_n)
    """
    transmit_power = _require_positive_finite(
        transmit_power_w,
        "transmit_power_w",
    )

    channel_gain = _require_positive_finite(
        channel_gain_linear,
        "channel_gain_linear",
    )

    noise_psd = _require_positive_finite(
        noise_psd_w_hz,
        "noise_psd_w_hz",
    )

    allocated_bandwidth = _require_positive_finite(
        allocated_bandwidth_hz,
        "allocated_bandwidth_hz",
    )

    noise_power = noise_psd * allocated_bandwidth

    if not math.isfinite(noise_power) or noise_power <= 0.0:
        raise ModelValidationError(
            "Noise power calculation produced an invalid result."
        )

    snr = transmit_power * channel_gain / noise_power

    if not math.isfinite(snr) or snr <= 0.0:
        raise ModelValidationError(
            "SNR calculation produced an invalid result."
        )

    return snr


def compute_access_rate_bps(
    allocated_bandwidth_hz: float,
    snr_linear: float,
) -> float:
    """
    Compute the UAV-to-SV rate using the Shannon expression.

    The model follows:

        R = B_n * log2(1 + SNR)
    """
    allocated_bandwidth = _require_positive_finite(
        allocated_bandwidth_hz,
        "allocated_bandwidth_hz",
    )

    snr = _require_positive_finite(
        snr_linear,
        "snr_linear",
    )

    spectral_efficiency = math.log1p(snr) / math.log(2.0)
    rate = allocated_bandwidth * spectral_efficiency

    if not math.isfinite(rate) or rate <= 0.0:
        raise ModelValidationError(
            "Access-rate calculation produced an invalid result."
        )

    return rate


def build_access_link_state(
    uav_id: int,
    distance_m: float,
    reference_distance_m: float,
    reference_gain_linear: float,
    transmit_power_w: float,
    noise_psd_w_hz: float,
    allocated_bandwidth_hz: float,
) -> AccessLinkState:
    """Build the complete physical state of one UAV-to-SV link."""
    if isinstance(uav_id, bool) or not isinstance(uav_id, int):
        raise ModelValidationError(
            f"uav_id must be an integer. Received: {uav_id!r}"
        )

    if uav_id <= 0:
        raise ModelValidationError(
            f"uav_id must be greater than zero. Received: {uav_id}"
        )

    gain = compute_channel_gain(
        distance_m=distance_m,
        reference_distance_m=reference_distance_m,
        reference_gain_linear=reference_gain_linear,
    )

    snr = compute_snr_linear(
        transmit_power_w=transmit_power_w,
        channel_gain_linear=gain,
        noise_psd_w_hz=noise_psd_w_hz,
        allocated_bandwidth_hz=allocated_bandwidth_hz,
    )

    rate = compute_access_rate_bps(
        allocated_bandwidth_hz=allocated_bandwidth_hz,
        snr_linear=snr,
    )

    return AccessLinkState(
        uav_id=uav_id,
        distance_m=float(distance_m),
        channel_gain_linear=gain,
        snr_linear=snr,
        rate_bps=rate,
    )

@dataclass(frozen=True, slots=True)
class ScenarioDefinition:
    """Static definition of one homogeneous execution scenario."""

    scenario: str
    execution_tier: str
    access_payload_kind: str
    backhaul_payload_kind: str


@dataclass(frozen=True, slots=True)
class LatencyComponents:
    """Resolved latency components for one UAV and one scenario."""

    scenario: str
    execution_tier: str

    access_payload_bits: float
    backhaul_payload_bits: float
    effective_compute_capacity_cycles_s: float

    access_time_s: float
    compute_time_s: float
    backhaul_transmission_time_s: float
    backhaul_fixed_delay_s: float

    end_to_end_latency_s: float


def compute_bandwidth_per_uav(
    total_bandwidth_hz: float,
    number_of_uavs: int,
) -> float:
    """
    Compute the equal access bandwidth assigned to each UAV.

        B_n = B / N
    """
    return _compute_equal_share(
        total_resource=total_bandwidth_hz,
        number_of_shares=number_of_uavs,
        field_name="total_bandwidth_hz",
    )


def compute_backhaul_rate_per_flow(
    aggregate_capacity_bps: float,
    number_of_uavs: int,
) -> float:
    """
    Compute the equal SV-to-RCC backhaul rate assigned to each flow.

        R_2 = C_bh / N
    """
    return _compute_equal_share(
        total_resource=aggregate_capacity_bps,
        number_of_shares=number_of_uavs,
        field_name="aggregate_capacity_bps",
    )


def compute_remote_capacity_per_job(
    total_capacity_cycles_s: float,
    number_of_uavs: int,
) -> float:
    """
    Compute the equal remote processing capacity assigned to each job.

        f_eff = f_total / N
    """
    return _compute_equal_share(
        total_resource=total_capacity_cycles_s,
        number_of_shares=number_of_uavs,
        field_name="total_capacity_cycles_s",
    )


def compute_transmission_time(
    payload_bits: float,
    rate_bps: float,
) -> float:
    """
    Compute the transmission time of a payload.

        t_tx = payload / rate
    """
    payload = _require_non_negative_finite(
        payload_bits,
        "payload_bits",
    )

    rate = _require_positive_finite(
        rate_bps,
        "rate_bps",
    )

    transmission_time = payload / rate

    if not math.isfinite(transmission_time):
        raise ModelValidationError(
            "Transmission-time calculation produced an invalid result."
        )

    return transmission_time


def compute_processing_time(
    workload_cycles: float,
    effective_capacity_cycles_s: float,
) -> float:
    """
    Compute the processing time of an inference workload.

        t_comp = W / f_eff
    """
    workload = _require_positive_finite(
        workload_cycles,
        "workload_cycles",
    )

    capacity = _require_positive_finite(
        effective_capacity_cycles_s,
        "effective_capacity_cycles_s",
    )

    processing_time = workload / capacity

    if not math.isfinite(processing_time) or processing_time <= 0.0:
        raise ModelValidationError(
            "Processing-time calculation produced an invalid result."
        )

    return processing_time

_SCENARIO_DEFINITIONS: dict[str, ScenarioDefinition] = {
    "S1": ScenarioDefinition(
        scenario="S1",
        execution_tier="UAV",
        access_payload_kind="output",
        backhaul_payload_kind="output",
    ),
    "S2": ScenarioDefinition(
        scenario="S2",
        execution_tier="SV",
        access_payload_kind="input",
        backhaul_payload_kind="output",
    ),
    "S3": ScenarioDefinition(
        scenario="S3",
        execution_tier="RCC",
        access_payload_kind="input",
        backhaul_payload_kind="input",
    ),
}


def get_scenario_definition(
    scenario: str,
) -> ScenarioDefinition:
    """Return the immutable definition of a supported scenario."""
    if not isinstance(scenario, str):
        raise ModelValidationError(
            f"scenario must be a string. Received: {scenario!r}"
        )

    normalized = scenario.strip().upper()

    try:
        return _SCENARIO_DEFINITIONS[normalized]
    except KeyError as exc:
        raise ModelValidationError(
            f"Unsupported scenario: {scenario!r}. "
            "Expected one of: S1, S2, S3."
        ) from exc

def compute_latency_components(
    scenario: str,
    access_rate_bps: float,
    backhaul_rate_bps: float,
    input_payload_bits: float,
    output_payload_bits: float,
    workload_cycles: float,
    uav_capacity_cycles_s: float,
    sv_capacity_cycles_s: float,
    rcc_capacity_cycles_s: float,
    number_of_uavs: int,
    backhaul_fixed_delay_s: float,
) -> LatencyComponents:
    """Compute all latency components for one UAV and one scenario."""
    definition = get_scenario_definition(scenario)

    access_rate = _require_positive_finite(
        access_rate_bps,
        "access_rate_bps",
    )

    backhaul_rate = _require_positive_finite(
        backhaul_rate_bps,
        "backhaul_rate_bps",
    )

    input_payload = _require_positive_finite(
        input_payload_bits,
        "input_payload_bits",
    )

    output_payload = _require_positive_finite(
        output_payload_bits,
        "output_payload_bits",
    )

    if output_payload > input_payload:
        raise ModelValidationError(
            "output_payload_bits cannot exceed input_payload_bits."
        )

    workload = _require_positive_finite(
        workload_cycles,
        "workload_cycles",
    )

    uav_capacity = _require_positive_finite(
        uav_capacity_cycles_s,
        "uav_capacity_cycles_s",
    )

    sv_capacity = _require_positive_finite(
        sv_capacity_cycles_s,
        "sv_capacity_cycles_s",
    )

    rcc_capacity = _require_positive_finite(
        rcc_capacity_cycles_s,
        "rcc_capacity_cycles_s",
    )

    uav_count = _require_positive_int(
        number_of_uavs,
        "number_of_uavs",
    )

    fixed_delay = _require_non_negative_finite(
        backhaul_fixed_delay_s,
        "backhaul_fixed_delay_s",
    )

    if definition.access_payload_kind == "output":
        access_payload = output_payload
    else:
        access_payload = input_payload

    if definition.backhaul_payload_kind == "output":
        backhaul_payload = output_payload
    else:
        backhaul_payload = input_payload

    if definition.scenario == "S1":
        effective_compute_capacity = uav_capacity
    elif definition.scenario == "S2":
        effective_compute_capacity = compute_remote_capacity_per_job(
            total_capacity_cycles_s=sv_capacity,
            number_of_uavs=uav_count,
        )
    else:
        effective_compute_capacity = compute_remote_capacity_per_job(
            total_capacity_cycles_s=rcc_capacity,
            number_of_uavs=uav_count,
        )

    access_time = compute_transmission_time(
        payload_bits=access_payload,
        rate_bps=access_rate,
    )

    compute_time = compute_processing_time(
        workload_cycles=workload,
        effective_capacity_cycles_s=effective_compute_capacity,
    )

    backhaul_transmission_time = compute_transmission_time(
        payload_bits=backhaul_payload,
        rate_bps=backhaul_rate,
    )

    end_to_end_latency = (
        access_time
        + compute_time
        + backhaul_transmission_time
        + fixed_delay
    )

    if (
        not math.isfinite(end_to_end_latency)
        or end_to_end_latency <= 0.0
    ):
        raise ModelValidationError(
            "End-to-end latency calculation produced an invalid result."
        )

    component_sum = (
        access_time
        + compute_time
        + backhaul_transmission_time
        + fixed_delay
    )

    if not math.isclose(
        end_to_end_latency,
        component_sum,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise ModelValidationError(
            "End-to-end latency is inconsistent with its components."
        )

    return LatencyComponents(
        scenario=definition.scenario,
        execution_tier=definition.execution_tier,
        access_payload_bits=access_payload,
        backhaul_payload_bits=backhaul_payload,
        effective_compute_capacity_cycles_s=(
            effective_compute_capacity
        ),
        access_time_s=access_time,
        compute_time_s=compute_time,
        backhaul_transmission_time_s=(
            backhaul_transmission_time
        ),
        backhaul_fixed_delay_s=fixed_delay,
        end_to_end_latency_s=end_to_end_latency,
    )
