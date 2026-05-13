from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tlkcore_ris_control import RISControllerCli, _ret_msg, _ret_ok


def _extract_shape_from_pattern(pattern: Any) -> tuple[int, int] | None:
    if isinstance(pattern, list) and pattern and isinstance(pattern[0], list):
        rows = len(pattern)
        cols = len(pattern[0]) if pattern[0] else 0
        if rows > 0 and cols > 0:
            return rows, cols
    return None


def _extract_shape_from_module_info(module_info: dict[str, Any], module: int) -> tuple[int, int] | None:
    node = module_info.get(str(module))
    if not isinstance(node, dict):
        return None

    antenna_size = node.get("antenna_size")
    if (
        isinstance(antenna_size, list)
        and len(antenna_size) == 2
        and all(isinstance(v, int) and v > 0 for v in antenna_size)
    ):
        cols, rows = antenna_size[0], antenna_size[1]
        return rows, cols
    return None


def _set_pattern(cli: RISControllerCli, sn: str, module: int, pattern: list[list[int]]) -> None:
    module_pattern: list[list[int]] | dict[int, list[list[int]]]
    if module == 1:
        module_pattern = pattern
    else:
        module_pattern = {module: pattern}

    ret = cli.service.setRISPattern(sn, module_pattern)
    if not _ret_ok(ret):
        raise RuntimeError(f"setRISPattern failed: {_ret_msg(ret)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Set RIS to a binary pattern (random/ones/zeros) and keep it on device."
    )
    parser.add_argument("--working-root", default=".", help="TLKCore working root")
    parser.add_argument("--interface", default="LAN", help="LAN/COMPORT/USB/ALL")
    parser.add_argument("--sn", required=True, help="Target RIS SN")
    parser.add_argument("--module", type=int, default=1, help="Target module index, default=1")
    parser.add_argument("--rows", type=int, default=0, help="Pattern rows (optional)")
    parser.add_argument("--cols", type=int, default=0, help="Pattern cols (optional)")
    parser.add_argument(
        "--mode",
        choices=("random", "ones", "zeros"),
        default="random",
        help="Pattern mode: random / ones / zeros",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    parser.add_argument(
        "--save-csv",
        default="",
        help="Optional output csv path for the pattern",
    )
    parser.add_argument(
        "--save-json",
        default="",
        help="Optional output json path for the pattern",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip read-back verification",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.module < 1:
        raise ValueError("--module must be >= 1")

    cli = RISControllerCli(Path(args.working_root).resolve())
    sn = cli.init_ris(args.sn.strip(), args.interface)
    print(f"Using SN: {sn}")

    shape: tuple[int, int] | None = None
    if args.rows > 0 and args.cols > 0:
        shape = (args.rows, args.cols)
    else:
        current = cli.get_pattern(sn, args.module)
        shape = _extract_shape_from_pattern(current)
        if shape is None:
            module_info = cli.get_module_info(sn)
            shape = _extract_shape_from_module_info(module_info, args.module)

    if shape is None:
        raise RuntimeError(
            "Cannot infer pattern shape. Please provide --rows and --cols explicitly."
        )

    rows, cols = shape
    if args.mode == "ones":
        pattern = np.ones((rows, cols), dtype=np.int32)
    elif args.mode == "zeros":
        pattern = np.zeros((rows, cols), dtype=np.int32)
    else:
        rng = np.random.default_rng(args.seed)
        pattern = rng.integers(0, 2, size=(rows, cols), dtype=np.int32)
    pattern_list = pattern.tolist()

    _set_pattern(cli, sn, args.module, pattern_list)
    print(f"setRISPattern: OK (module={args.module}, shape={rows}x{cols})")

    if args.save_csv:
        out_csv = Path(args.save_csv)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = Path("files") / f"{args.mode}_pattern_module{args.module}_{ts}.csv"
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
            json.dump(pattern_list, f, ensure_ascii=False, indent=2)
        print(f"saved json: {out_json.resolve()}")

    if not args.no_verify:
        read_back = cli.get_pattern(sn, args.module)
        if read_back != pattern_list:
            raise RuntimeError("Read-back verify failed: device pattern != target pattern")
        print("verify read-back: PASS")

    print("Done. Pattern is kept on RIS device (no rollback).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
