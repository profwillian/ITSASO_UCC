import math

import pytest

from ucc.config import resolve_factorized_grid
from ucc.geometry import (
    build_subregions,
    compute_uav_sv_distance,
    generate_spatial_realization,
)


@pytest.mark.parametrize(
    ("number_of_uavs", "expected"),
    [
        (4, (2, 2)),
        (8, (2, 4)),
        (12, (3, 4)),
        (16, (4, 4)),
    ],
)
def test_resolve_factorized_grid(
    number_of_uavs: int,
    expected: tuple[int, int],
) -> None:
    assert resolve_factorized_grid(number_of_uavs) == expected


def test_build_baseline_subregions() -> None:
    regions = build_subregions(
        area_side_m=1000.0,
        rows=2,
        columns=4,
    )

    assert len(regions) == 8

    first = regions[0]
    assert first.subregion_id == 1
    assert first.row_index == 0
    assert first.column_index == 0
    assert first.x_min_m == 0.0
    assert first.x_max_m == 250.0
    assert first.y_min_m == 0.0
    assert first.y_max_m == 500.0

    fourth = regions[3]
    assert fourth.x_min_m == 750.0
    assert fourth.x_max_m == 1000.0
    assert fourth.y_min_m == 0.0
    assert fourth.y_max_m == 500.0

    fifth = regions[4]
    assert fifth.x_min_m == 0.0
    assert fifth.x_max_m == 250.0
    assert fifth.y_min_m == 500.0
    assert fifth.y_max_m == 1000.0


def test_subregions_cover_complete_area() -> None:
    regions = build_subregions(
        area_side_m=1000.0,
        rows=2,
        columns=4,
    )

    total_area = sum(
        (region.x_max_m - region.x_min_m)
        * (region.y_max_m - region.y_min_m)
        for region in regions
    )

    assert total_area == pytest.approx(1_000_000.0)


def test_distance_directly_above_surface_vessel() -> None:
    distance = compute_uav_sv_distance(
        uav_x_m=500.0,
        uav_y_m=500.0,
        altitude_m=500.0,
        sv_x_m=500.0,
        sv_y_m=500.0,
    )

    assert distance == pytest.approx(500.0)


def test_distance_from_search_area_corner() -> None:
    distance = compute_uav_sv_distance(
        uav_x_m=0.0,
        uav_y_m=0.0,
        altitude_m=500.0,
        sv_x_m=500.0,
        sv_y_m=500.0,
    )

    expected = math.sqrt(
        500.0**2
        + 500.0**2
        + 500.0**2
    )

    assert distance == pytest.approx(
        expected,
        rel=1.0e-12,
    )


def test_same_seed_produces_same_realization() -> None:
    first = generate_spatial_realization(
        number_of_uavs=8,
        area_side_m=1000.0,
        altitude_m=500.0,
        sv_x_m=500.0,
        sv_y_m=500.0,
        seed=1,
    )

    second = generate_spatial_realization(
        number_of_uavs=8,
        area_side_m=1000.0,
        altitude_m=500.0,
        sv_x_m=500.0,
        sv_y_m=500.0,
        seed=1,
    )

    assert first == second


def test_different_seeds_change_at_least_one_position() -> None:
    first = generate_spatial_realization(
        number_of_uavs=8,
        area_side_m=1000.0,
        altitude_m=500.0,
        sv_x_m=500.0,
        sv_y_m=500.0,
        seed=1,
    )

    second = generate_spatial_realization(
        number_of_uavs=8,
        area_side_m=1000.0,
        altitude_m=500.0,
        sv_x_m=500.0,
        sv_y_m=500.0,
        seed=2,
    )

    changed = any(
        first_position.x_m != second_position.x_m
        or first_position.y_m != second_position.y_m
        for first_position, second_position in zip(
            first.uav_positions,
            second.uav_positions,
            strict=True,
        )
    )

    assert changed


def test_positions_are_inside_assigned_subregions() -> None:
    realization = generate_spatial_realization(
        number_of_uavs=8,
        area_side_m=1000.0,
        altitude_m=500.0,
        sv_x_m=500.0,
        sv_y_m=500.0,
        seed=17,
    )

    assert len(realization.uav_positions) == 8
    assert realization.grid_rows == 2
    assert realization.grid_columns == 4

    for position in realization.uav_positions:
        assert (
            position.subregion_x_min_m
            <= position.x_m
            < position.subregion_x_max_m
        )

        assert (
            position.subregion_y_min_m
            <= position.y_m
            < position.subregion_y_max_m
        )

        assert position.altitude_m == 500.0
        assert position.distance_to_sv_m >= 500.0
