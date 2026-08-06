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
