from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workflow_modules.ris_pattern_module import calc_ris_pattern


def test_default_shape_matches_tmytek_10x10() -> None:
    result = calc_ris_pattern(
        incident_az_deg=0.0,
        incident_el_deg=0.0,
        reflection_az_deg=10.0,
        reflection_el_deg=5.0,
    )
    assert result.code.shape == (10, 10)
    assert result.payload.shape == (100,)


def test_near_field_with_angles_and_distances() -> None:
    result = calc_ris_pattern(
        incident_az_deg=10.0,
        incident_el_deg=5.0,
        reflection_az_deg=-20.0,
        reflection_el_deg=15.0,
        incident_distance_m=2.5,
        reflection_distance_m=3.2,
        nx=21,
        ny=21,
        bits=1,
    )

    assert result.mode == "near_field"
    assert result.code.shape == (21, 21)
    assert result.payload.shape == (21 * 21,)
    assert np.all((result.code == 0) | (result.code == 1))


def test_far_field_with_angles_only() -> None:
    result = calc_ris_pattern(
        incident_az_deg=0.0,
        incident_el_deg=0.0,
        reflection_az_deg=30.0,
        reflection_el_deg=10.0,
        nx=21,
        ny=21,
        bits=2,
    )

    assert result.mode == "far_field"
    assert result.phase_ris.shape == (21, 21)
    assert result.code.shape == (21, 21)
    assert result.payload.shape == (21 * 21,)
    assert np.min(result.code) >= 0
    assert np.max(result.code) <= 3


def main() -> None:
    test_default_shape_matches_tmytek_10x10()
    test_near_field_with_angles_and_distances()
    test_far_field_with_angles_only()
    print("test_ris_pattern_module: PASS")


if __name__ == "__main__":
    main()
