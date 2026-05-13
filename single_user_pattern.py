import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class RisPatternResult:
    phase_ris: np.ndarray
    code: np.ndarray
    payload: np.ndarray
    mode: str


def _dir_from_azel_deg(az_deg: float, el_deg: float) -> np.ndarray:
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    return np.array(
        [
            np.cos(el) * np.sin(az),  # x
            np.sin(el),               # y
            np.cos(el) * np.cos(az),  # z
        ],
        dtype=float,
    )


def _point_from_azelr_deg(az_deg: float, el_deg: float, r_m: float) -> np.ndarray:
    direction = _dir_from_azel_deg(az_deg, el_deg)
    return r_m * direction


def _build_ris_grid(
    nx: int,
    ny: int,
    dx: float,
    dy: float,
) -> Tuple[np.ndarray, np.ndarray]:
    xs = (np.arange(nx) - (nx - 1) / 2.0) * dx
    ys = (np.arange(ny) - (ny - 1) / 2.0) * dy
    return np.meshgrid(xs, ys, indexing="xy")


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


def calc_single_user_pattern(
    in_az_deg: float,
    in_el_deg: float,
    out_az_deg: float,
    out_el_deg: float,
    in_r_m: float | None = None,
    out_r_m: float | None = None,
    *,
    freq_hz: float = 3.5e9,
    nx: int = 21,
    ny: int = 21,
    dx: float = 0.0428,
    dy: float = 0.0431,
    bits: int = 1,
    transpose_for_hw: bool = True,
) -> RisPatternResult:
    c = 3e8
    wavelength = c / freq_hz
    k = 2 * np.pi / wavelength

    x, y = _build_ris_grid(nx, ny, dx, dy)
    z = np.zeros_like(x)

    if in_r_m is not None and out_r_m is not None:
        # Near-field model: phase from path sum |Tx-RISn| + |RISn-Target|
        p_in = _point_from_azelr_deg(in_az_deg, in_el_deg, in_r_m)
        p_out = _point_from_azelr_deg(out_az_deg, out_el_deg, out_r_m)

        d_in = np.sqrt((x - p_in[0]) ** 2 + (y - p_in[1]) ** 2 + (z - p_in[2]) ** 2)
        d_out = np.sqrt((x - p_out[0]) ** 2 + (y - p_out[1]) ** 2 + (z - p_out[2]) ** 2)
        phase = k * (d_in + d_out)
        mode = "near_field"
    else:
        # Far-field approximation: phase gradient by direction difference
        s_in = _dir_from_azel_deg(in_az_deg, in_el_deg)
        s_out = _dir_from_azel_deg(out_az_deg, out_el_deg)
        ds = s_out - s_in
        phase = k * (ds[0] * x + ds[1] * y + ds[2] * z)
        mode = "far_field"

    phase_ris = np.mod(phase, 2 * np.pi)

    code = _quantize_phase(phase_ris, bits=bits)
    code_for_hw = code.T if transpose_for_hw else code
    payload = code_for_hw.flatten(order="C")

    return RisPatternResult(phase_ris=phase_ris, code=code, payload=payload, mode=mode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-user RIS pattern calculator (from incident/outgoing az-el)."
    )
    parser.add_argument("--in-az", type=float, required=True, help="Incident azimuth in degree")
    parser.add_argument("--in-el", type=float, required=True, help="Incident elevation in degree")
    parser.add_argument("--out-az", type=float, required=True, help="Outgoing azimuth in degree")
    parser.add_argument("--out-el", type=float, required=True, help="Outgoing elevation in degree")
    parser.add_argument(
        "--in-r",
        type=float,
        default=None,
        help="Incident source range in meter (enable near-field when both --in-r and --out-r are set)",
    )
    parser.add_argument(
        "--out-r",
        type=float,
        default=None,
        help="Target range in meter (enable near-field when both --in-r and --out-r are set)",
    )
    parser.add_argument("--freq", type=float, default=3.5e9, help="Carrier frequency in Hz")
    parser.add_argument("--nx", type=int, default=21, help="RIS element count on x-axis")
    parser.add_argument("--ny", type=int, default=21, help="RIS element count on y-axis")
    parser.add_argument("--dx", type=float, default=0.0428, help="Element pitch on x-axis (m)")
    parser.add_argument("--dy", type=float, default=0.0431, help="Element pitch on y-axis (m)")
    parser.add_argument("--bits", type=int, default=1, help="Quantization bits")
    parser.add_argument(
        "--no-transpose",
        action="store_true",
        help="Do not transpose before hardware payload packing",
    )
    parser.add_argument(
        "--save-code",
        type=str,
        default="",
        help="Optional output txt path for code matrix",
    )
    args = parser.parse_args()

    result = calc_single_user_pattern(
        in_az_deg=args.in_az,
        in_el_deg=args.in_el,
        out_az_deg=args.out_az,
        out_el_deg=args.out_el,
        in_r_m=args.in_r,
        out_r_m=args.out_r,
        freq_hz=args.freq,
        nx=args.nx,
        ny=args.ny,
        dx=args.dx,
        dy=args.dy,
        bits=args.bits,
        transpose_for_hw=not args.no_transpose,
    )

    print(f"mode      : {result.mode}")
    print(f"phase shape: {result.phase_ris.shape}")
    print(f"code shape : {result.code.shape}")
    print(f"payload len: {len(result.payload)}")
    print("payload preview (first 32):")
    print(result.payload[:32].tolist())

    if args.save_code:
        np.savetxt(args.save_code, result.code, fmt="%d", delimiter="\t")
        print(f"saved code matrix to: {args.save_code}")


if __name__ == "__main__":
    main()
