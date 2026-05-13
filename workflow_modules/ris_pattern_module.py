from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class RisPattern:
    phase_ris: np.ndarray
    code: np.ndarray
    payload: np.ndarray
    mode: str


@dataclass(frozen=True)
class RisPanelConfig:
    nx: int
    ny: int
    dx: float
    dy: float


# TMYTEK sub-6G dynamic RIS common panel (10x10, 42.8cm x 43.1cm aperture).
TMYTEK_PANEL_10X10 = RisPanelConfig(nx=10, ny=10, dx=0.0428, dy=0.0431)


def _dir_from_azel_deg(az_deg: float, el_deg: float) -> np.ndarray:
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    return np.array(
        [
            np.cos(el) * np.sin(az),
            np.sin(el),
            np.cos(el) * np.cos(az),
        ],
        dtype=float,
    )


def _point_from_azelr_deg(az_deg: float, el_deg: float, r_m: float) -> np.ndarray:
    if r_m <= 0:
        raise ValueError("distance must be > 0")
    return r_m * _dir_from_azel_deg(az_deg, el_deg)


def _as_xyz(point: Iterable[float], name: str) -> np.ndarray:
    arr = np.asarray(list(point), dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be length-3 xyz coordinates")
    return arr


def _build_ris_grid(nx: int, ny: int, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    if nx < 1 or ny < 1:
        raise ValueError("nx/ny must be >= 1")
    if dx <= 0 or dy <= 0:
        raise ValueError("dx/dy must be > 0")

    xs = (np.arange(nx) - (nx - 1) / 2.0) * dx
    ys = (np.arange(ny) - (ny - 1) / 2.0) * dy
    return np.meshgrid(xs, ys, indexing="xy")


def build_panel_config_from_size(
    *,
    nx: int,
    ny: int,
    panel_size_x_m: float,
    panel_size_y_m: float,
) -> RisPanelConfig:
    """
    Build panel config from physical aperture size.

    Example:
      panel_size_x_m=0.428, panel_size_y_m=0.431, nx=10, ny=10
      -> dx=0.0428, dy=0.0431
    """
    if nx < 1 or ny < 1:
        raise ValueError("nx/ny must be >= 1")
    if panel_size_x_m <= 0 or panel_size_y_m <= 0:
        raise ValueError("panel_size_x_m/panel_size_y_m must be > 0")
    return RisPanelConfig(
        nx=nx,
        ny=ny,
        dx=panel_size_x_m / nx,
        dy=panel_size_y_m / ny,
    )


def _quantize_phase(phase_ris: np.ndarray, bits: int) -> np.ndarray:
    if bits < 1:
        raise ValueError("bits must be >= 1")

    if bits == 1:
        code = np.zeros_like(phase_ris, dtype=np.int32)
        code[(phase_ris > np.pi) & (phase_ris <= 2 * np.pi)] = 1
        return code

    levels = 2 ** bits
    code = np.round((phase_ris / (2 * np.pi)) * levels) % levels
    return code.astype(np.int32)


def _resolve_point(
    point_m: Iterable[float] | None,
    az_deg: float | None,
    el_deg: float | None,
    distance_m: float | None,
) -> np.ndarray | None:
    if point_m is not None:
        return _as_xyz(point_m, "point_m")
    if az_deg is None or el_deg is None or distance_m is None:
        return None
    return _point_from_azelr_deg(az_deg, el_deg, distance_m)


def calc_ris_pattern(
    *,
    incident_point_m: Iterable[float] | None = None,
    reflection_point_m: Iterable[float] | None = None,
    incident_az_deg: float | None = None,
    incident_el_deg: float | None = None,
    reflection_az_deg: float | None = None,
    reflection_el_deg: float | None = None,
    incident_distance_m: float | None = None,
    reflection_distance_m: float | None = None,
    freq_hz: float = 3.5e9,
    nx: int = TMYTEK_PANEL_10X10.nx,
    ny: int = TMYTEK_PANEL_10X10.ny,
    dx: float = TMYTEK_PANEL_10X10.dx,
    dy: float = TMYTEK_PANEL_10X10.dy,
    bits: int = 1,
    transpose_for_payload: bool = True,
) -> RisPattern:
    """
    Calculate RIS element pattern.

    Near-field mode:
      - Uses `incident_point_m` + `reflection_point_m`, or
      - Uses (incident/reflection az, el, distance) to build the two points.

    Far-field mode:
      - Uses incident/reflection azimuth + elevation only.
    """
    if freq_hz <= 0:
        raise ValueError("freq_hz must be > 0")

    c = 3e8
    wavelength = c / freq_hz
    k = 2 * np.pi / wavelength

    x, y = _build_ris_grid(nx, ny, dx, dy)
    z = np.zeros_like(x)

    p_in = _resolve_point(
        point_m=incident_point_m,
        az_deg=incident_az_deg,
        el_deg=incident_el_deg,
        distance_m=incident_distance_m,
    )
    p_out = _resolve_point(
        point_m=reflection_point_m,
        az_deg=reflection_az_deg,
        el_deg=reflection_el_deg,
        distance_m=reflection_distance_m,
    )

    if p_in is not None and p_out is not None:
        d_in = np.sqrt((x - p_in[0]) ** 2 + (y - p_in[1]) ** 2 + (z - p_in[2]) ** 2)
        d_out = np.sqrt((x - p_out[0]) ** 2 + (y - p_out[1]) ** 2 + (z - p_out[2]) ** 2)
        phase = k * (d_in + d_out)
        mode = "near_field"
    else:
        if None in (incident_az_deg, incident_el_deg, reflection_az_deg, reflection_el_deg):
            raise ValueError(
                "far-field requires incident/reflection azimuth and elevation, "
                "or provide both near-field points."
            )

        s_in = _dir_from_azel_deg(float(incident_az_deg), float(incident_el_deg))
        s_out = _dir_from_azel_deg(float(reflection_az_deg), float(reflection_el_deg))
        ds = s_out - s_in
        phase = k * (ds[0] * x + ds[1] * y + ds[2] * z)
        mode = "far_field"

    phase_ris = np.mod(phase, 2 * np.pi)
    code = _quantize_phase(phase_ris, bits)
    code_for_payload = code.T if transpose_for_payload else code
    payload = code_for_payload.flatten(order="C")

    return RisPattern(phase_ris=phase_ris, code=code, payload=payload, mode=mode)
