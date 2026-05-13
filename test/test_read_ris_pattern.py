from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tlkcore_ris_control import RISControllerCli, _parse_module


def _shape_of_pattern(pattern: Any) -> tuple[int, int] | None:
    if isinstance(pattern, list) and pattern and isinstance(pattern[0], list):
        rows = len(pattern)
        cols = len(pattern[0]) if pattern[0] else 0
        return rows, cols
    return None


def _save_pattern_json(pattern: Any, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(pattern, f, ensure_ascii=False, indent=2)


def _save_pattern_csv(pattern: Any, out_path: Path) -> None:
    if not (isinstance(pattern, list) and pattern and isinstance(pattern[0], list)):
        raise ValueError("CSV export requires a 2D pattern matrix (e.g. module=1)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(pattern, dtype=int)
    np.savetxt(out_path, arr, fmt="%d", delimiter=",")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auto-read current RIS pattern from device")
    parser.add_argument("--working-root", default=".", help="TLKCore working root")
    parser.add_argument("--interface", default="LAN", help="LAN/COMPORT/USB/ALL")
    parser.add_argument("--sn", required=True, help="Target RIS SN")
    parser.add_argument(
        "--module",
        default="1",
        help='Module index/list, e.g. "1" or "[1,2]"',
    )
    parser.add_argument("--count", type=int, default=1, help="Read count (<=0 means infinite)")
    parser.add_argument("--interval", type=float, default=0.0, help="Interval seconds")
    parser.add_argument("--save-json", default="", help="Optional output json path")
    parser.add_argument("--save-csv", default="", help="Optional output csv path (2D matrix only)")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    module = _parse_module(args.module)
    cli = RISControllerCli(Path(args.working_root).resolve())
    sn = cli.init_ris(args.sn.strip(), args.interface)
    print(f"Using SN: {sn}")

    i = 0
    while True:
        i += 1
        pattern = cli.get_pattern(sn, module)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        shape = _shape_of_pattern(pattern)

        print(f"[{ts}] read #{i}")
        if shape:
            print(f"pattern shape: {shape[0]}x{shape[1]}")
        else:
            print(f"pattern type : {type(pattern).__name__}")
        print(json.dumps(pattern, ensure_ascii=False))

        if args.save_json:
            out_json = Path(args.save_json)
            if not out_json.is_absolute():
                out_json = Path(os.getcwd()) / out_json
            _save_pattern_json(pattern, out_json.resolve())
            print(f"saved json: {out_json.resolve()}")

        if args.save_csv:
            out_csv = Path(args.save_csv)
            if not out_csv.is_absolute():
                out_csv = Path(os.getcwd()) / out_csv
            _save_pattern_csv(pattern, out_csv.resolve())
            print(f"saved csv : {out_csv.resolve()}")

        if args.count > 0 and i >= args.count:
            break
        if args.interval > 0:
            time.sleep(args.interval)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
