from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workflow_modules.ris_pattern_module import calc_ris_pattern


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float

    def as_tuple(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z


def _angle_to_deg(value: float, unit: str) -> float:
    if unit == "deg":
        return value
    if unit == "rad":
        return float(np.rad2deg(value))
    raise ValueError(f"Unsupported angle unit: {unit}")


def _point_from_polar(r_m: float, az: float, el: float, unit: str) -> Point3D:
    if r_m <= 0:
        raise ValueError("distance/r must be > 0")

    az_deg = _angle_to_deg(az, unit)
    el_deg = _angle_to_deg(el, unit)
    az_rad = np.deg2rad(az_deg)
    el_rad = np.deg2rad(el_deg)

    return Point3D(
        x=float(r_m * np.cos(el_rad) * np.sin(az_rad)),
        y=float(r_m * np.sin(el_rad)),
        z=float(r_m * np.cos(el_rad) * np.cos(az_rad)),
    )


def _load_json_arg(text: str, file_path: str) -> dict[str, Any]:
    if text and file_path:
        raise ValueError("Use either --position-json or --position-json-file, not both")
    if file_path:
        with Path(file_path).open("r", encoding="utf-8") as f:
            data = json.load(f)
    elif text:
        data = json.loads(text)
    else:
        return {}

    if not isinstance(data, dict):
        raise ValueError("Position JSON must be an object")
    return data


def _first_number(data: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return float(value)
    return None


def _point_from_json(data: dict[str, Any], default_unit: str) -> Point3D | None:
    xyz = data.get("xyz") or data.get("point_m") or data.get("target_point_m")
    if xyz is not None:
        if not isinstance(xyz, list | tuple) or len(xyz) != 3:
            raise ValueError("xyz/point_m must be a length-3 list")
        return Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2]))

    x = _first_number(data, "x", "X")
    y = _first_number(data, "y", "Y")
    z = _first_number(data, "z", "Z")
    if x is not None and y is not None and z is not None:
        return Point3D(x, y, z)

    r = _first_number(data, "r", "distance", "distance_m")
    az_deg = _first_number(data, "azimuth_deg", "az_deg")
    el_deg = _first_number(data, "elevation_deg", "el_deg")
    az_rad = _first_number(data, "azimuth_rad", "az_rad")
    el_rad = _first_number(data, "elevation_rad", "el_rad")
    az = _first_number(data, "azimuth", "az")
    el = _first_number(data, "elevation", "el")

    if r is None:
        return None
    if az_deg is not None and el_deg is not None:
        return _point_from_polar(r, az_deg, el_deg, "deg")
    if az_rad is not None and el_rad is not None:
        return _point_from_polar(r, az_rad, el_rad, "rad")
    if az is not None and el is not None:
        return _point_from_polar(r, az, el, default_unit)

    raise ValueError("Position JSON has r/distance but no azimuth/elevation")


def _resolve_target_point(args: argparse.Namespace) -> Point3D:
    data = _load_json_arg(args.position_json, args.position_json_file)
    point = _point_from_json(data, args.angle_unit) if data else None

    if point is None:
        if None not in (args.target_x, args.target_y, args.target_z):
            point = Point3D(args.target_x, args.target_y, args.target_z)
        elif None not in (args.target_r, args.target_az, args.target_el):
            point = _point_from_polar(
                args.target_r,
                args.target_az,
                args.target_el,
                args.angle_unit,
            )
        else:
            raise ValueError(
                "Provide target position via --position-json, --position-json-file, "
                "--target-x/y/z, or --target-r/az/el"
            )

    # Translation-only camera-to-RIS transform. Keep zero if camera origin is RIS center.
    return Point3D(
        x=point.x + args.camera_origin_x,
        y=point.y + args.camera_origin_y,
        z=point.z + args.camera_origin_z,
    )


def _resolve_incident_point(args: argparse.Namespace) -> Point3D:
    if None not in (args.incident_x, args.incident_y, args.incident_z):
        return Point3D(args.incident_x, args.incident_y, args.incident_z)

    return _point_from_polar(
        args.incident_r,
        args.incident_az,
        args.incident_el,
        args.incident_angle_unit,
    )


def _set_pattern(
    *,
    sn: str,
    interface: str,
    working_root: Path,
    module: int,
    pattern: list[list[int]],
    no_verify: bool,
) -> None:
    from tlkcore_ris_control import RISControllerCli, _ret_msg, _ret_ok

    cli = RISControllerCli(working_root)
    target_sn = cli.init_ris(sn.strip(), interface)
    print(f"Using SN: {target_sn}")

    module_pattern: list[list[int]] | dict[int, list[list[int]]]
    if module == 1:
        module_pattern = pattern
    else:
        module_pattern = {module: pattern}

    ret = cli.service.setRISPattern(target_sn, module_pattern)
    if not _ret_ok(ret):
        raise RuntimeError(f"setRISPattern failed: {_ret_msg(ret)}")

    print(f"setRISPattern: OK (module={module})")
    if not no_verify:
        read_back = cli.get_pattern(target_sn, module)
        if read_back != pattern:
            raise RuntimeError("Same-process verify failed: TLKCore pattern cache != target pattern")
        print("same-process verify: PASS")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate RIS pattern from a target position and optionally set it on device."
    )
    parser.add_argument("--working-root", default=".", help="TLKCore working root")
    parser.add_argument("--interface", default="LAN", help="LAN/COMPORT/USB/ALL")
    parser.add_argument("--sn", required=True, help="Target RIS SN")
    parser.add_argument("--module", type=int, default=1, help="Target module index")
    parser.add_argument("--dry-run", action="store_true", help="Calculate and save only; do not set device")
    parser.add_argument("--no-verify", action="store_true", help="Skip same-process TLKCore verify")

    target = parser.add_argument_group("target position")
    target.add_argument("--position-json", default="", help='Target JSON, e.g. {"r":2,"azimuth":0.1,"elevation":0.0}')
    target.add_argument("--position-json-file", default="", help="Path to target position JSON")
    target.add_argument("--angle-unit", choices=("rad", "deg"), default="rad", help="Unit for target az/el")
    target.add_argument("--target-r", type=float, default=None, help="Target distance in meters")
    target.add_argument("--target-az", type=float, default=None, help="Target azimuth")
    target.add_argument("--target-el", type=float, default=None, help="Target elevation")
    target.add_argument("--target-x", type=float, default=None, help="Target X in meters")
    target.add_argument("--target-y", type=float, default=None, help="Target Y in meters")
    target.add_argument("--target-z", type=float, default=None, help="Target Z in meters")
    target.add_argument("--camera-origin-x", type=float, default=0.0, help="Camera origin X in RIS frame")
    target.add_argument("--camera-origin-y", type=float, default=0.0, help="Camera origin Y in RIS frame")
    target.add_argument("--camera-origin-z", type=float, default=0.0, help="Camera origin Z in RIS frame")

    incident = parser.add_argument_group("incident source position")
    incident.add_argument("--incident-r", type=float, default=1.0, help="Incident source distance in meters")
    incident.add_argument("--incident-az", type=float, default=0.0, help="Incident source azimuth")
    incident.add_argument("--incident-el", type=float, default=0.0, help="Incident source elevation")
    incident.add_argument(
        "--incident-angle-unit",
        choices=("rad", "deg"),
        default="deg",
        help="Unit for incident az/el",
    )
    incident.add_argument("--incident-x", type=float, default=None, help="Incident source X in meters")
    incident.add_argument("--incident-y", type=float, default=None, help="Incident source Y in meters")
    incident.add_argument("--incident-z", type=float, default=None, help="Incident source Z in meters")

    pattern = parser.add_argument_group("pattern")
    pattern.add_argument("--freq-mhz", type=float, default=3500.0, help="RIS operating frequency in MHz")
    pattern.add_argument("--bits", type=int, default=1, help="Phase quantization bits")
    pattern.add_argument("--rows", type=int, default=10, help="RIS pattern rows")
    pattern.add_argument("--cols", type=int, default=10, help="RIS pattern cols")
    pattern.add_argument("--save-csv", default="", help="Optional output CSV path")
    pattern.add_argument("--save-json", default="", help="Optional output JSON path")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.module < 1:
        raise ValueError("--module must be >= 1")

    target_point = _resolve_target_point(args)
    incident_point = _resolve_incident_point(args)

    result = calc_ris_pattern(
        incident_point_m=incident_point.as_tuple(),
        reflection_point_m=target_point.as_tuple(),
        freq_hz=args.freq_mhz * 1e6,
        nx=args.cols,
        ny=args.rows,
        bits=args.bits,
    )
    pattern = result.code.astype(int)
    pattern_list = pattern.tolist()

    print(f"mode: {result.mode}")
    print(
        "incident point in RIS frame: "
        f"x={incident_point.x:.3f} m, y={incident_point.y:.3f} m, z={incident_point.z:.3f} m"
    )
    print(
        "target point in RIS frame  : "
        f"x={target_point.x:.3f} m, y={target_point.y:.3f} m, z={target_point.z:.3f} m"
    )
    print(f"pattern shape: {pattern.shape[0]}x{pattern.shape[1]}")
    print(f"pattern ones : {int(pattern.sum())}/{pattern.size}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = Path(args.save_csv) if args.save_csv else Path("files") / f"position_pattern_module{args.module}_{ts}.csv"
    if not out_csv.is_absolute():
        out_csv = Path(os.getcwd()) / out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_csv.resolve(), pattern, fmt="%d", delimiter=",")
    print(f"saved csv: {out_csv.resolve()}")

    if args.save_json:
        out_json = Path(args.save_json)
        if not out_json.is_absolute():
            out_json = Path(os.getcwd()) / out_json
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.resolve().open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "mode": result.mode,
                    "incident_point_m": incident_point.as_tuple(),
                    "target_point_m": target_point.as_tuple(),
                    "freq_mhz": args.freq_mhz,
                    "bits": args.bits,
                    "pattern": pattern_list,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"saved json: {out_json.resolve()}")

    if args.dry_run:
        print("dry-run: skipped setRISPattern")
        return 0

    _set_pattern(
        sn=args.sn,
        interface=args.interface,
        working_root=Path(args.working_root).resolve(),
        module=args.module,
        pattern=pattern_list,
        no_verify=args.no_verify,
    )
    print("Done. Pattern is kept on RIS device.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
