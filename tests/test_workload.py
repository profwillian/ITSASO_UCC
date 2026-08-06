import pytest

from ucc.model import (
    ModelValidationError,
    build_inference_workload,
)


def test_build_baseline_inference_workload() -> None:
    workload = build_inference_workload(
        chunks_per_job=3,
        chunk_size_bits=2_000_000.0,
        output_input_ratio=0.10,
        compute_intensity_cycles_per_bit=1000.0,
    )

    assert workload.chunks_per_job == 3
    assert workload.chunk_size_bits == 2_000_000.0
    assert workload.input_payload_bits == 6_000_000.0
    assert workload.output_input_ratio == 0.10
    assert workload.output_payload_bits == 600_000.0
    assert workload.compute_intensity_cycles_per_bit == 1000.0
    assert workload.workload_cycles == 6_000_000_000.0


@pytest.mark.parametrize(
    "chunks_per_job",
    [0, -1],
)
def test_chunks_per_job_must_be_positive(
    chunks_per_job: int,
) -> None:
    with pytest.raises(ModelValidationError):
        build_inference_workload(
            chunks_per_job=chunks_per_job,
            chunk_size_bits=2_000_000.0,
            output_input_ratio=0.10,
            compute_intensity_cycles_per_bit=1000.0,
        )


@pytest.mark.parametrize(
    "ratio",
    [0.0, -0.1, 1.1],
)
def test_output_input_ratio_must_be_valid(
    ratio: float,
) -> None:
    with pytest.raises(ModelValidationError):
        build_inference_workload(
            chunks_per_job=3,
            chunk_size_bits=2_000_000.0,
            output_input_ratio=ratio,
            compute_intensity_cycles_per_bit=1000.0,
        )


def test_output_ratio_equal_to_one_is_valid() -> None:
    workload = build_inference_workload(
        chunks_per_job=3,
        chunk_size_bits=2_000_000.0,
        output_input_ratio=1.0,
        compute_intensity_cycles_per_bit=1000.0,
    )

    assert workload.output_payload_bits == workload.input_payload_bits
