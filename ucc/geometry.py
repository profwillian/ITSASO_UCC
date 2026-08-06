"""Spatial model for the UCC maritime simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ucc.config import ConfigurationError, resolve_factorized_grid


@dataclass(frozen=True, slots=True)
class Subregion:
    """One equal-area search subregion assigned to one UAV."""

    subregion_id: int
    row_index: int
    column_index: int

    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float


@dataclass(frozen=True, slots=True)
class UAVPosition:
    """Position of one UAV and its distance to the surface vessel."""

    uav_id: int
    subregion_id: int
    row_index: int
    column_index: int

    subregion_x_min_m: float
    subregion_x_max_m: float
    subregion_y_min_m: float
    subregion_y_max_m: float

    x_m: float
    y_m: float
    altitude_m: float

    sv_x_m: float
    sv_y_m: float

    distance_to_sv_m: float


@dataclass(frozen=True, slots=True)
class SpatialRealization:
    """Complete spatial realization generated from one random seed."""

    seed: int
    grid_rows: int
    grid_columns: int
    uav_positions: tuple[UAVPosition, ...]


def _require_positive_finite(value: float, field_name: str) -> float:
    """Validate a strictly positive finite numeric value."""
    converted = float(value)

    if not math.isfinite(converted) or converted <= 0.0:
        raise ConfigurationError(
            f"{field_name} must be finite and greater than zero. "
            f"Received: {value!r}"
        )

    return converted


def _require_finite(value: float, field_name: str) -> float:
    """Validate a finite numeric value."""
    converted = float(value)

    if not math.isfinite(converted):
        raise ConfigurationError(
            f"{field_name} must be finite. Received: {value!r}"
        )

    return converted


def build_subregions(
    area_side_m: float,
    rows: int,
    columns: int,
) -> tuple[Subregion, ...]:
    """Divide a square search area into equal-area rectangular regions."""
    area_side = _require_positive_finite(
        area_side_m,
        "area_side_m",
    )

    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise ConfigurationError(
            f"rows must be a positive integer. Received: {rows!r}"
        )

    if (
        isinstance(columns, bool)
        or not isinstance(columns, int)
        or columns <= 0
    ):
        raise ConfigurationError(
            f"columns must be a positive integer. Received: {columns!r}"
        )

    region_width = area_side / columns
    region_height = area_side / rows

    subregions: list[Subregion] = []
    subregion_id = 1

    for row_index in range(rows):
        for column_index in range(columns):
            x_min = column_index * region_width
            x_max = (column_index + 1) * region_width
            y_min = row_index * region_height
            y_max = (row_index + 1) * region_height

            subregions.append(
                Subregion(
                    subregion_id=subregion_id,
                    row_index=row_index,
                    column_index=column_index,
                    x_min_m=x_min,
                    x_max_m=x_max,
                    y_min_m=y_min,
                    y_max_m=y_max,
                )
            )

            subregion_id += 1

    return tuple(subregions)


def sample_position_in_subregion(
    subregion: Subregion,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Sample one horizontal UAV position uniformly inside a subregion."""
    x_m = float(
        rng.uniform(
            subregion.x_min_m,
            subregion.x_max_m,
        )
    )

    y_m = float(
        rng.uniform(
            subregion.y_min_m,
            subregion.y_max_m,
        )
    )

    return x_m, y_m


def compute_uav_sv_distance(
    uav_x_m: float,
    uav_y_m: float,
    altitude_m: float,
    sv_x_m: float,
    sv_y_m: float,
) -> float:
    """Compute the three-dimensional Euclidean UAV-to-SV distance."""
    uav_x = _require_finite(uav_x_m, "uav_x_m")
    uav_y = _require_finite(uav_y_m, "uav_y_m")
    altitude = _require_positive_finite(
        altitude_m,
        "altitude_m",
    )
    sv_x = _require_finite(sv_x_m, "sv_x_m")
    sv_y = _require_finite(sv_y_m, "sv_y_m")

    distance = math.sqrt(
        (uav_x - sv_x) ** 2
        + (uav_y - sv_y) ** 2
        + altitude**2
    )

    if not math.isfinite(distance) or distance < altitude:
        raise ConfigurationError(
            "Computed UAV-to-SV distance is invalid."
        )

    return distance


def generate_spatial_realization(
    number_of_uavs: int,
    area_side_m: float,
    altitude_m: float,
    sv_x_m: float,
    sv_y_m: float,
    seed: int,
) -> SpatialRealization:
    """Generate one reproducible spatial realization."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigurationError(
            f"seed must be a non-negative integer. Received: {seed!r}"
        )

    rows, columns = resolve_factorized_grid(number_of_uavs)

    subregions = build_subregions(
        area_side_m=area_side_m,
        rows=rows,
        columns=columns,
    )

    rng = np.random.default_rng(seed)

    positions: list[UAVPosition] = []

    for uav_id, subregion in enumerate(subregions, start=1):
        x_m, y_m = sample_position_in_subregion(
            subregion=subregion,
            rng=rng,
        )

        distance = compute_uav_sv_distance(
            uav_x_m=x_m,
            uav_y_m=y_m,
            altitude_m=altitude_m,
            sv_x_m=sv_x_m,
            sv_y_m=sv_y_m,
        )

        positions.append(
            UAVPosition(
                uav_id=uav_id,
                subregion_id=subregion.subregion_id,
                row_index=subregion.row_index,
                column_index=subregion.column_index,
                subregion_x_min_m=subregion.x_min_m,
                subregion_x_max_m=subregion.x_max_m,
                subregion_y_min_m=subregion.y_min_m,
                subregion_y_max_m=subregion.y_max_m,
                x_m=x_m,
                y_m=y_m,
                altitude_m=float(altitude_m),
                sv_x_m=float(sv_x_m),
                sv_y_m=float(sv_y_m),
                distance_to_sv_m=distance,
            )
        )

    if len(positions) != number_of_uavs:
        raise ConfigurationError(
            "The number of generated UAV positions is inconsistent."
        )

    return SpatialRealization(
        seed=seed,
        grid_rows=rows,
        grid_columns=columns,
        uav_positions=tuple(positions),
    )
