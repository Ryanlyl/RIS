#!/usr/bin/env python3
"""
TLKCore RIS control script.

Features:
1. Scan and list available TLKCore devices.
2. Initialize a RIS device by SN (or auto-pick the first scanned device).
3. Set RIS pattern by incident/reflection angle.
4. Set RIS pattern from CSV (0/1 matrix).
5. Read back current RIS pattern.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any


try:
    from tlkcore import DevInterface, RIS_Dir, RIS_ModuleConfig, RetCode, TLKCoreService
except Exception as exc:  # pragma: no cover
    print("Failed to import tlkcore.")
    print("Please install it first, e.g. `pip install tlkcore`.")
    raise SystemExit(1) from exc


def _ret_ok(ret: Any) -> bool:
    return hasattr(ret, "RetCode") and ret.RetCode == RetCode.OK


def _ret_msg(ret: Any) -> str:
    code = getattr(ret, "RetCode", "UNKNOWN")
    msg = getattr(ret, "RetMsg", "")
    data = getattr(ret, "RetData", None)
    return f"RetCode={code}, RetMsg={msg}, RetData={data}"


def _to_interface(name: str) -> Any:
    key = name.strip().upper()
    if not hasattr(DevInterface, key):
        valid = [x for x in ("LAN", "COMPORT", "USB", "ALL") if hasattr(DevInterface, x)]
        raise ValueError(f"Invalid interface '{name}', valid: {valid}")
    return getattr(DevInterface, key)


def _load_module_rotate(text: str | None) -> dict[str, int] | None:
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("--module-rotate must be a JSON dict, e.g. '{\"1\": 90, \"2\": 180}'") from exc

    if not isinstance(data, dict):
        raise ValueError("--module-rotate must decode to dict")

    result: dict[str, int] = {}
    for k, v in data.items():
        result[str(k)] = int(v)
    return result


def _load_pattern_csv(csv_path: Path) -> list[list[int]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    matrix: list[list[int]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for ridx, row in enumerate(reader, start=1):
            if not row:
                continue
            parsed = [int(x.strip()) for x in row]
            if any(v not in (0, 1) for v in parsed):
                raise ValueError(f"CSV row {ridx} has non-binary value, only 0/1 supported")
            matrix.append(parsed)

    if not matrix:
        raise ValueError("CSV contains no valid rows")

    row_len = len(matrix[0])
    if any(len(r) != row_len for r in matrix):
        raise ValueError("CSV rows must have identical column count")

    return matrix


class RISControllerCli:
    def __init__(self, working_root: Path) -> None:
        self.working_root = working_root
        self.service = self._create_service()
        if not self.service.running:
            raise RuntimeError("TLKCoreService is not running")

    def _create_service(self) -> Any:
        # Compatible with different TLKCore versions:
        # - TLKCoreService(working_root=...)
        # - TLKCoreService("...")
        try:
            return TLKCoreService(working_root=str(self.working_root))
        except TypeError:
            return TLKCoreService(str(self.working_root))

    def scan(self, interface_name: str) -> dict[str, tuple[str, int, bool]]:
        interface = _to_interface(interface_name)
        ret = self.service.scanDevices(interface=interface)
        if not _ret_ok(ret):
            raise RuntimeError(f"scanDevices failed: {_ret_msg(ret)}")

        scan_info = self.service.getScanInfo().RetData
        if not isinstance(scan_info, dict):
            raise RuntimeError(f"getScanInfo returned unexpected data: {scan_info}")
        return scan_info

    def init_ris(self, sn: str | None, interface_name: str) -> str:
        scan_info = self.scan(interface_name)
        if not scan_info:
            raise RuntimeError("No TLKCore device found")

        target_sn = sn or next(iter(scan_info.keys()))

        if target_sn not in scan_info:
            raise RuntimeError(
                f"SN '{target_sn}' not found in scan result: {list(scan_info.keys())}"
            )

        try:
            ret = self.service.initDev(target_sn, is_custom_calibration=False)
        except TypeError:
            ret = self.service.initDev(target_sn)

        if not _ret_ok(ret):
            raise RuntimeError(f"initDev failed: {_ret_msg(ret)}")

        dev_name = self.service.getDevTypeName(target_sn)
        if str(dev_name).upper() != "RIS":
            print(f"Warning: target device type is '{dev_name}', not 'RIS'.")

        return target_sn

    def get_module_info(self, sn: str) -> dict[str, Any]:
        ret = self.service.getRISModuleInfo(sn)
        if not _ret_ok(ret):
            raise RuntimeError(f"getRISModuleInfo failed: {_ret_msg(ret)}")
        if not isinstance(ret.RetData, dict):
            raise RuntimeError("getRISModuleInfo returned non-dict data")
        return ret.RetData

    def set_by_angle(
        self,
        sn: str,
        inc_distance: float,
        inc_theta: float,
        inc_phi: float,
        ref_distance: float,
        ref_theta: float,
        ref_phi: float,
        freq_mhz: int,
        module: int | list[Any],
        module_rotate: dict[str, int] | None,
        save: bool,
    ) -> None:
        incident = RIS_Dir(inc_distance, (inc_theta, inc_phi))
        reflection = RIS_Dir(ref_distance, (ref_theta, ref_phi))
        module_cfg = RIS_ModuleConfig(freq_mhz, module, module_rotate)

        ret = self.service.setRISAngle(sn, incident, reflection, module_cfg, save=save)
        if not _ret_ok(ret):
            raise RuntimeError(f"setRISAngle failed: {_ret_msg(ret)}")

    def set_by_csv(self, sn: str, csv_path: Path, module: int) -> None:
        pattern = _load_pattern_csv(csv_path)
        module_pattern: list[list[int]] | dict[int, list[list[int]]]
        if module == 1:
            module_pattern = pattern
        else:
            module_pattern = {module: pattern}

        ret = self.service.setRISPattern(sn, module_pattern)
        if not _ret_ok(ret):
            raise RuntimeError(f"setRISPattern failed: {_ret_msg(ret)}")

    def get_pattern(self, sn: str, module: int | list[int]) -> Any:
        ret = self.service.getRISPattern(sn, module)
        if not _ret_ok(ret):
            raise RuntimeError(f"getRISPattern failed: {_ret_msg(ret)}")
        return ret.RetData


def _parse_module(text: str) -> int | list[Any]:
    # Supports:
    # "1"
    # "[1,2]"
    # "[[1,2],[3,4]]"
    text = text.strip()
    if text.startswith("["):
        return json.loads(text)
    return int(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Control RIS device via tlkcore")
    parser.add_argument("--working-root", default=".", help="TLKCore working root")
    parser.add_argument("--interface", default="ALL", help="Scan interface: LAN/COMPORT/USB/ALL")
    parser.add_argument("--sn", default="", help="Target SN (empty means auto-pick first)")

    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="Scan devices")
    p_scan.add_argument("--interface", dest="interface_cmd", default=None, help="LAN/COMPORT/USB/ALL")

    p_info = sub.add_parser("module-info", help="Query RIS module info")
    p_info.add_argument("--sn", dest="sn_cmd", default=None, help="Target SN")
    p_info.add_argument("--interface", dest="interface_cmd", default=None, help="LAN/COMPORT/USB/ALL")

    p_angle = sub.add_parser("set-angle", help="Set RIS pattern by angle")
    p_angle.add_argument("--sn", dest="sn_cmd", default=None, help="Target SN")
    p_angle.add_argument("--interface", dest="interface_cmd", default=None, help="LAN/COMPORT/USB/ALL")
    p_angle.add_argument("--inc-distance", type=float, default=1.0, help="Incident distance (m)")
    p_angle.add_argument("--inc-theta", type=float, default=0.0, help="Incident theta (deg)")
    p_angle.add_argument("--inc-phi", type=float, default=0.0, help="Incident phi (deg)")
    p_angle.add_argument("--ref-distance", type=float, default=1.0, help="Reflection distance (m)")
    p_angle.add_argument("--ref-theta", type=float, required=True, help="Reflection theta (deg)")
    p_angle.add_argument("--ref-phi", type=float, default=0.0, help="Reflection phi (deg)")
    p_angle.add_argument("--freq-mhz", type=int, default=28000, help="Central frequency (MHz)")
    p_angle.add_argument(
        "--module",
        default="1",
        help='Target module, e.g. "1", "[1,2]", "[[1,2],[3,4]]"',
    )
    p_angle.add_argument(
        "--module-rotate",
        default="",
        help='JSON dict, e.g. \'{"1": 90, "2": 180}\'',
    )
    p_angle.add_argument("--save", action="store_true", help="Save debug csv (TLKCore option)")

    p_csv = sub.add_parser("set-pattern-csv", help="Set RIS pattern from CSV (0/1 matrix)")
    p_csv.add_argument("--sn", dest="sn_cmd", default=None, help="Target SN")
    p_csv.add_argument("--interface", dest="interface_cmd", default=None, help="LAN/COMPORT/USB/ALL")
    p_csv.add_argument("--csv", required=True, help="CSV path")
    p_csv.add_argument("--module", type=int, default=1, help="Target module index")

    p_get = sub.add_parser("get-pattern", help="Read current RIS pattern")
    p_get.add_argument("--sn", dest="sn_cmd", default=None, help="Target SN")
    p_get.add_argument("--interface", dest="interface_cmd", default=None, help="LAN/COMPORT/USB/ALL")
    p_get.add_argument(
        "--module",
        default="1",
        help='Target module index/list, e.g. "1" or "[1,2]"',
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    interface_name = getattr(args, "interface_cmd", None) or args.interface
    sn_cli = getattr(args, "sn_cmd", None)
    sn_value = (sn_cli if sn_cli is not None else args.sn).strip()

    cli = RISControllerCli(Path(args.working_root).resolve())

    if args.command == "scan":
        scan_info = cli.scan(interface_name)
        print(json.dumps(scan_info, indent=2, ensure_ascii=False))
        return 0

    sn = cli.init_ris(sn_value or None, interface_name)
    print(f"Using SN: {sn}")

    if args.command == "module-info":
        info = cli.get_module_info(sn)
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return 0

    if args.command == "set-angle":
        module = _parse_module(args.module)
        module_rotate = _load_module_rotate(args.module_rotate.strip() or None)
        cli.set_by_angle(
            sn=sn,
            inc_distance=args.inc_distance,
            inc_theta=args.inc_theta,
            inc_phi=args.inc_phi,
            ref_distance=args.ref_distance,
            ref_theta=args.ref_theta,
            ref_phi=args.ref_phi,
            freq_mhz=args.freq_mhz,
            module=module,
            module_rotate=module_rotate,
            save=args.save,
        )
        print("setRISAngle: OK")
        return 0

    if args.command == "set-pattern-csv":
        csv_path = Path(args.csv)
        if not csv_path.is_absolute():
            csv_path = Path(os.getcwd()) / csv_path
        cli.set_by_csv(sn, csv_path.resolve(), args.module)
        print("setRISPattern: OK")
        return 0

    if args.command == "get-pattern":
        module = _parse_module(args.module)
        pattern = cli.get_pattern(sn, module)
        print(json.dumps(pattern, indent=2, ensure_ascii=False))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
