from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workflow_modules.ris_pattern_module import TMYTEK_PANEL_10X10, calc_ris_pattern


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float

    def as_tuple(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z


@dataclass(frozen=True)
class TargetSnapshot:
    track_id: int
    azimuth_deg: float
    elevation_deg: float
    distance_m: float
    camera_point_m: Point3D
    ris_point_m: Point3D
    score: float
    timestamp: float


def point_from_azelr_deg(az_deg: float, el_deg: float, r_m: float) -> Point3D:
    if r_m <= 0:
        raise ValueError("distance must be > 0")

    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    return Point3D(
        x=float(r_m * np.cos(el) * np.sin(az)),
        y=float(r_m * np.sin(el)),
        z=float(r_m * np.cos(el) * np.cos(az)),
    )


def resolve_incident_point(args: argparse.Namespace) -> Point3D:
    if None not in (args.incident_x, args.incident_y, args.incident_z):
        return Point3D(args.incident_x, args.incident_y, args.incident_z)
    return point_from_azelr_deg(args.incident_az, args.incident_el, args.incident_r)


def target_to_snapshot(target: Any, camera_origin: Point3D) -> TargetSnapshot:
    camera_point = point_from_azelr_deg(
        az_deg=target.azimuth,
        el_deg=target.elevation,
        r_m=target.distance,
    )
    ris_point = Point3D(
        x=camera_point.x + camera_origin.x,
        y=camera_point.y + camera_origin.y,
        z=camera_point.z + camera_origin.z,
    )
    return TargetSnapshot(
        track_id=target.track_id,
        azimuth_deg=target.azimuth,
        elevation_deg=target.elevation,
        distance_m=target.distance,
        camera_point_m=camera_point,
        ris_point_m=ris_point,
        score=target.score,
        timestamp=target.timestamp,
    )


def select_target(
    targets: Sequence[Any],
    *,
    policy: str,
    preferred_track_id: int | None,
) -> Any | None:
    if not targets:
        return None

    if preferred_track_id is not None:
        for target in targets:
            if target.track_id == preferred_track_id:
                return target

    if policy == "nearest":
        return min(targets, key=lambda t: t.distance)
    if policy == "highest-score":
        return max(targets, key=lambda t: t.score)
    if policy == "first":
        return targets[0]

    raise ValueError(f"Unsupported target policy: {policy}")


def should_update(
    previous: TargetSnapshot | None,
    current: TargetSnapshot,
    *,
    angle_threshold_deg: float,
    distance_threshold_m: float,
) -> bool:
    if previous is None:
        return True

    daz = abs(current.azimuth_deg - previous.azimuth_deg)
    delv = abs(current.elevation_deg - previous.elevation_deg)
    ddist = abs(current.distance_m - previous.distance_m)
    return (
        daz >= angle_threshold_deg
        or delv >= angle_threshold_deg
        or ddist >= distance_threshold_m
    )


class RisPatternSetter:
    def __init__(
        self,
        *,
        dry_run: bool,
        working_root: Path,
        interface: str,
        sn: str,
        module: int,
        verify: bool,
    ) -> None:
        self.dry_run = dry_run
        self.module = module
        self.verify = verify
        self.cli: Any | None = None
        self.sn = sn

        if self.module < 1:
            raise ValueError("--module must be >= 1")

        if self.dry_run:
            return

        if not sn.strip():
            raise ValueError("--sn is required unless --dry-run is used")

        from tlkcore_ris_control import RISControllerCli

        self.cli = RISControllerCli(working_root)
        self.sn = self.cli.init_ris(sn.strip(), interface)
        print(f"Using SN: {self.sn}")

    def set_pattern(self, pattern: list[list[int]]) -> None:
        if self.dry_run:
            print("dry-run: skipped setRISPattern")
            return

        if self.cli is None:
            raise RuntimeError("RIS controller is not initialized")

        from tlkcore_ris_control import _ret_msg, _ret_ok

        module_pattern: list[list[int]] | dict[int, list[list[int]]]
        if self.module == 1:
            module_pattern = pattern
        else:
            module_pattern = {self.module: pattern}

        ret = self.cli.service.setRISPattern(self.sn, module_pattern)
        if not _ret_ok(ret):
            raise RuntimeError(f"setRISPattern failed: {_ret_msg(ret)}")

        print(f"setRISPattern: OK (module={self.module})")
        if self.verify:
            read_back = self.cli.get_pattern(self.sn, self.module)
            if read_back != pattern:
                raise RuntimeError("Read-back verify failed: device pattern != target pattern")
            print("verify read-back: PASS")


class TrackingRisExperiment:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.camera_origin = Point3D(
            args.camera_origin_x,
            args.camera_origin_y,
            args.camera_origin_z,
        )
        self.incident_point = resolve_incident_point(args)
        self.setter = RisPatternSetter(
            dry_run=args.dry_run,
            working_root=Path(args.working_root).resolve(),
            interface=args.interface,
            sn=args.sn,
            module=args.module,
            verify=args.verify,
        )
        self.last_update_time = 0.0
        self.last_print_time = 0.0
        self.last_snapshot: TargetSnapshot | None = None
        self.sticky_track_id: int | None = None
        self.update_count = 0

    def __call__(self, targets: list[Any]) -> None:
        now = time.time()
        target = select_target(
            targets,
            policy=self.args.target_policy,
            preferred_track_id=self.sticky_track_id if self.args.sticky_target else None,
        )
        if target is None:
            if self.args.print_empty and self.should_print(now):
                print("target: none")
                self.last_print_time = now
            return

        snapshot = target_to_snapshot(target, self.camera_origin)
        if self.args.sticky_target:
            self.sticky_track_id = snapshot.track_id

        if self.should_print(now):
            self.print_snapshot(snapshot)
            self.last_print_time = now

        if now - self.last_update_time < self.args.min_update_interval:
            return

        if not should_update(
            self.last_snapshot,
            snapshot,
            angle_threshold_deg=self.args.angle_threshold_deg,
            distance_threshold_m=self.args.distance_threshold_m,
        ):
            return

        pattern = self.calculate_pattern(snapshot)
        self.persist_pattern(pattern, snapshot)
        self.setter.set_pattern(pattern.tolist())

        self.last_update_time = now
        self.last_snapshot = snapshot
        self.update_count += 1
        if self.args.max_updates and self.update_count >= self.args.max_updates:
            raise KeyboardInterrupt

    def should_print(self, now: float) -> bool:
        return (
            self.args.print_interval <= 0
            or now - self.last_print_time >= self.args.print_interval
        )

    def calculate_pattern(self, snapshot: TargetSnapshot) -> np.ndarray:
        result = calc_ris_pattern(
            incident_point_m=self.incident_point.as_tuple(),
            reflection_point_m=snapshot.ris_point_m.as_tuple(),
            freq_hz=self.args.freq_mhz * 1e6,
            nx=self.args.cols,
            ny=self.args.rows,
            dx=self.args.dx,
            dy=self.args.dy,
            bits=self.args.bits,
        )
        return result.code.astype(int)

    def persist_pattern(self, pattern: np.ndarray, snapshot: TargetSnapshot) -> None:
        if not self.args.save_csv and not self.args.save_json:
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if self.args.save_csv:
            csv_path = self.resolve_output_path(self.args.save_csv, ts, "csv", snapshot)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(csv_path, pattern, fmt="%d", delimiter=",")
            print(f"saved csv: {csv_path}")

        if self.args.save_json:
            json_path = self.resolve_output_path(self.args.save_json, ts, "json", snapshot)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "track_id": snapshot.track_id,
                        "azimuth_deg": snapshot.azimuth_deg,
                        "elevation_deg": snapshot.elevation_deg,
                        "distance_m": snapshot.distance_m,
                        "camera_point_m": snapshot.camera_point_m.as_tuple(),
                        "ris_point_m": snapshot.ris_point_m.as_tuple(),
                        "incident_point_m": self.incident_point.as_tuple(),
                        "freq_mhz": self.args.freq_mhz,
                        "bits": self.args.bits,
                        "pattern": pattern.tolist(),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"saved json: {json_path}")

    def resolve_output_path(
        self,
        template: str,
        timestamp: str,
        suffix: str,
        snapshot: TargetSnapshot,
    ) -> Path:
        path_text = template.format(
            timestamp=timestamp,
            module=self.args.module,
            track_id=snapshot.track_id,
        )
        path = Path(path_text)
        if path.is_dir() or not path.suffix:
            path = path / f"tracking_pattern_module{self.args.module}_{timestamp}.{suffix}"
        if not path.is_absolute():
            path = Path(os.getcwd()) / path
        return path.resolve()

    def print_snapshot(self, snapshot: TargetSnapshot) -> None:
        payload = {
            "track_id": snapshot.track_id,
            "azimuth_deg": round(snapshot.azimuth_deg, 3),
            "elevation_deg": round(snapshot.elevation_deg, 3),
            "distance_m": round(snapshot.distance_m, 3),
            "score": round(snapshot.score, 3),
            "ris_point_m": [
                round(snapshot.ris_point_m.x, 3),
                round(snapshot.ris_point_m.y, 3),
                round(snapshot.ris_point_m.z, 3),
            ],
        }
        if self.args.output_jsonl:
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print(
                "target "
                f"id={payload['track_id']} "
                f"az={payload['azimuth_deg']:+.3f}deg "
                f"el={payload['elevation_deg']:+.3f}deg "
                f"dist={payload['distance_m']:.3f}m "
                f"score={payload['score']:.3f} "
                f"ris=({payload['ris_point_m'][0]:+.3f},"
                f"{payload['ris_point_m'][1]:+.3f},"
                f"{payload['ris_point_m'][2]:+.3f})m"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Track a person with camera.py, calculate a RIS pattern for the tracked "
            "position, and optionally set it on the RIS with TLKCore."
        )
    )

    ris = parser.add_argument_group("RIS device")
    ris.add_argument("--working-root", default=".", help="TLKCore working root")
    ris.add_argument("--interface", default="LAN", help="LAN/COMPORT/USB/ALL")
    ris.add_argument("--sn", default="", help="Target RIS SN; required unless --dry-run is used")
    ris.add_argument("--module", type=int, default=1, help="Target RIS module index")
    ris.add_argument("--dry-run", action="store_true", help="Print and save only; do not set RIS")
    ris.add_argument("--verify", action="store_true", help="Read back pattern after each set")

    tracking = parser.add_argument_group("tracking")
    tracking.add_argument(
        "--target-policy",
        choices=("nearest", "highest-score", "first"),
        default="nearest",
        help="How to choose one user when multiple people are tracked",
    )
    tracking.add_argument(
        "--sticky-target",
        action="store_true",
        help="Keep using the first selected track id while it remains visible",
    )
    tracking.add_argument("--print-empty", action="store_true", help="Print when no target is visible")
    tracking.add_argument(
        "--print-interval",
        type=float,
        default=0.2,
        help="Minimum seconds between target snapshot prints; 0 prints every frame",
    )
    tracking.add_argument("--output-jsonl", action="store_true", help="Print target snapshots as JSON lines")
    tracking.add_argument(
        "--max-updates",
        type=int,
        default=0,
        help="Stop after N RIS updates; 0 means run until ESC/Ctrl+C",
    )

    transform = parser.add_argument_group("camera-to-RIS transform")
    transform.add_argument("--camera-origin-x", type=float, default=0.0, help="Camera origin X in RIS frame")
    transform.add_argument("--camera-origin-y", type=float, default=0.0, help="Camera origin Y in RIS frame")
    transform.add_argument("--camera-origin-z", type=float, default=0.0, help="Camera origin Z in RIS frame")

    incident = parser.add_argument_group("fixed transmitter position")
    incident.add_argument("--incident-r", type=float, default=1.0, help="Transmitter distance in meters")
    incident.add_argument("--incident-az", type=float, default=0.0, help="Transmitter azimuth in degrees")
    incident.add_argument("--incident-el", type=float, default=0.0, help="Transmitter elevation in degrees")
    incident.add_argument("--incident-x", type=float, default=None, help="Transmitter X in RIS frame")
    incident.add_argument("--incident-y", type=float, default=None, help="Transmitter Y in RIS frame")
    incident.add_argument("--incident-z", type=float, default=None, help="Transmitter Z in RIS frame")

    pattern = parser.add_argument_group("pattern")
    pattern.add_argument("--freq-mhz", type=float, default=3500.0, help="RIS operating frequency in MHz")
    pattern.add_argument("--bits", type=int, default=1, help="Phase quantization bits")
    pattern.add_argument("--rows", type=int, default=TMYTEK_PANEL_10X10.ny, help="RIS pattern rows")
    pattern.add_argument("--cols", type=int, default=TMYTEK_PANEL_10X10.nx, help="RIS pattern cols")
    pattern.add_argument("--dx", type=float, default=TMYTEK_PANEL_10X10.dx, help="Element pitch on x-axis in meters")
    pattern.add_argument("--dy", type=float, default=TMYTEK_PANEL_10X10.dy, help="Element pitch on y-axis in meters")
    pattern.add_argument(
        "--min-update-interval",
        type=float,
        default=0.5,
        help="Minimum seconds between RIS pattern updates",
    )
    pattern.add_argument(
        "--angle-threshold-deg",
        type=float,
        default=1.0,
        help="Skip update unless azimuth/elevation changes by at least this many degrees",
    )
    pattern.add_argument(
        "--distance-threshold-m",
        type=float,
        default=0.1,
        help="Skip update unless distance changes by at least this many meters",
    )
    pattern.add_argument(
        "--save-csv",
        default="",
        help="Optional CSV file or directory. Supports {timestamp}, {module}, {track_id}.",
    )
    pattern.add_argument(
        "--save-json",
        default="",
        help="Optional JSON file or directory. Supports {timestamp}, {module}, {track_id}.",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()
    experiment = TrackingRisExperiment(args)

    print(
        "incident point in RIS frame: "
        f"x={experiment.incident_point.x:.3f}m, "
        f"y={experiment.incident_point.y:.3f}m, "
        f"z={experiment.incident_point.z:.3f}m"
    )
    print("Starting camera tracking. Press ESC in the camera window or Ctrl+C to stop.")

    try:
        from camera import run as run_camera_tracking

        run_camera_tracking(callbacks=[experiment])
    except KeyboardInterrupt:
        print("Stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
