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
