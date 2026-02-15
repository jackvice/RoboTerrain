#!/usr/bin/env python3
"""
Diagnostic: replicates the Nav2 ObstacleLayer pipeline for /scan.

Subscribes to /scan, looks up the TF from lidar_link -> odom
at the scan's timestamp, transforms sample points, and checks
every filter the obstacle layer applies:
  1. TF lookup success/failure
  2. Range filter (obstacle_min_range .. obstacle_max_range)
  3. Height filter (min_obstacle_height .. max_obstacle_height)
  4. Costmap bounds check

Usage (with inspect world running + Nav2):
  python3 costmap_scan_diagnostic.py
"""
from __future__ import annotations

import math
import sys
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time, Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from nav2_msgs.msg import Costmap
import tf2_ros
import numpy as np


# --- Nav2 obstacle-layer defaults (match your YAML) ---
GLOBAL_FRAME: str = "odom"
SENSOR_FRAME: str = "lidar_link"
OBSTACLE_MIN_RANGE: float = 0.15
OBSTACLE_MAX_RANGE: float = 8.0
MIN_OBSTACLE_HEIGHT: float = 0.0
MAX_OBSTACLE_HEIGHT: float = 15.0
TF_TOLERANCE_SEC: float = 0.5   # your costmap transform_tolerance


def quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Quaternion (x,y,z,w) → 3×3 rotation matrix."""
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz),  2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),       1 - 2*(qx*qx + qy*qy)],
    ])


def transform_point(
    local_xyz: np.ndarray,
    tx: float, ty: float, tz: float,
    qx: float, qy: float, qz: float, qw: float,
) -> np.ndarray:
    """Apply TF (translation + rotation) to a point."""
    rot: np.ndarray = quat_to_matrix(qx, qy, qz, qw)
    return rot @ local_xyz + np.array([tx, ty, tz])


class ScanDiagnosticNode(Node):
    def __init__(self) -> None:
        super().__init__(
            "costmap_scan_diagnostic",
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self._on_scan, qos)

        # Also try RELIABLE in case that's what obstacle layer uses
        self.scan_sub_reliable = self.create_subscription(
            LaserScan, "/scan", self._on_scan_reliable, 10
        )
        self.reliable_count: int = 0

        self.costmap_sub = self.create_subscription(
            Costmap, "/local_costmap/costmap_raw", self._on_costmap, 10
        )
        self.costmap_origin: Optional[Tuple[float, float]] = None
        self.costmap_size: Optional[Tuple[int, int]] = None
        self.costmap_resolution: Optional[float] = None
        self.costmap_nonzero: int = 0

        self.msg_count: int = 0
        self.get_logger().info("Diagnostic started — waiting for /scan and TF …")

    # ---- callbacks ----
    def _on_scan_reliable(self, msg: LaserScan) -> None:
        self.reliable_count += 1

    def _on_costmap(self, msg: Costmap) -> None:
        self.costmap_origin = (msg.metadata.origin.position.x,
                               msg.metadata.origin.position.y)
        self.costmap_size = (msg.metadata.size_x, msg.metadata.size_y)
        self.costmap_resolution = msg.metadata.resolution
        self.costmap_nonzero = sum(1 for v in msg.data if v > 0)

    def _on_scan(self, msg: LaserScan) -> None:
        self.msg_count += 1
        if self.msg_count % 20 != 1:          # report ~once per second
            return

        scan_stamp = Time.from_msg(msg.header.stamp)
        sec: int = msg.header.stamp.sec
        nsec: int = msg.header.stamp.nanosec
        now = self.get_clock().now()
        age_ms: float = (now - scan_stamp).nanoseconds / 1e6

        print(f"\n{'='*70}")
        print(f"Scan #{self.msg_count}  stamp={sec}.{nsec:09d}  "
              f"frame_id={msg.header.frame_id}  age={age_ms:.0f} ms")
        print(f"  ranges: {len(msg.ranges)} samples, "
              f"range=[{msg.range_min:.2f}, {msg.range_max:.2f}]")

        # Count valid ranges
        valid: int = sum(
            1 for r in msg.ranges
            if not math.isinf(r) and not math.isnan(r) and r >= msg.range_min
        )
        in_obs_range: int = sum(
            1 for r in msg.ranges
            if OBSTACLE_MIN_RANGE <= r <= OBSTACLE_MAX_RANGE
            and not math.isinf(r) and not math.isnan(r)
        )
        print(f"  valid={valid}/{len(msg.ranges)}, in obstacle range "
              f"[{OBSTACLE_MIN_RANGE},{OBSTACLE_MAX_RANGE}]={in_obs_range}")
        print(f"  RELIABLE sub count: {self.reliable_count} "
              f"(>0 means reliable QoS works too)")

        # --- TF lookup at scan timestamp ---
        print(f"\n  TF lookup: {GLOBAL_FRAME} → {SENSOR_FRAME} at scan stamp …")
        try:
            tf_at_scan = self.tf_buffer.lookup_transform(
                GLOBAL_FRAME, SENSOR_FRAME, scan_stamp,
                timeout=Duration(seconds=0.0),
            )
            t = tf_at_scan.transform.translation
            r = tf_at_scan.transform.rotation
            tf_stamp = tf_at_scan.header.stamp
            print(f"  ✓ SUCCESS  tf_stamp={tf_stamp.sec}.{tf_stamp.nanosec:09d}")
            print(f"    translation=({t.x:.3f}, {t.y:.3f}, {t.z:.3f})")
            print(f"    rotation=({r.x:.4f}, {r.y:.4f}, {r.z:.4f}, {r.w:.4f})")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            # Try with tolerance
            print(f"\n  Retrying with {TF_TOLERANCE_SEC}s tolerance …")
            try:
                tf_at_scan = self.tf_buffer.lookup_transform(
                    GLOBAL_FRAME, SENSOR_FRAME, scan_stamp,
                    timeout=Duration(seconds=TF_TOLERANCE_SEC),
                )
                t = tf_at_scan.transform.translation
                r = tf_at_scan.transform.rotation
                print(f"  ✓ SUCCESS with tolerance")
                print(f"    translation=({t.x:.3f}, {t.y:.3f}, {t.z:.3f})")
            except Exception as e2:
                print(f"  ✗ STILL FAILED: {e2}")
                return

        # Try with time=0 (latest available)
        print(f"\n  TF lookup at time=0 (latest) …")
        try:
            tf_latest = self.tf_buffer.lookup_transform(
                GLOBAL_FRAME, SENSOR_FRAME, Time(),
                timeout=Duration(seconds=0.0),
            )
            tl = tf_latest.transform.translation
            print(f"  ✓ latest: ({tl.x:.3f}, {tl.y:.3f}, {tl.z:.3f})")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")

        # --- Transform sample scan points ---
        print(f"\n  Transforming sample points to {GLOBAL_FRAME} frame:")
        sample_indices = [0, len(msg.ranges)//4, len(msg.ranges)//2,
                          3*len(msg.ranges)//4]

        height_pass: int = 0
        height_fail: int = 0
        bounds_pass: int = 0
        bounds_fail: int = 0

        for idx in sample_indices:
            rng: float = msg.ranges[idx]
            if math.isinf(rng) or math.isnan(rng) or rng < msg.range_min:
                print(f"    [{idx:3d}] range={rng:.2f} — INVALID, skipped")
                continue
            angle: float = msg.angle_min + idx * msg.angle_increment

            # Point in lidar_link frame (2D scan → z=0 in sensor frame)
            local = np.array([rng * math.cos(angle),
                              rng * math.sin(angle),
                              0.0])

            # Transform to odom frame
            global_pt = transform_point(
                local, t.x, t.y, t.z, r.x, r.y, r.z, r.w)

            gx, gy, gz = float(global_pt[0]), float(global_pt[1]), float(global_pt[2])

            # Height filter
            h_ok: bool = MIN_OBSTACLE_HEIGHT <= gz <= MAX_OBSTACLE_HEIGHT
            if h_ok:
                height_pass += 1
            else:
                height_fail += 1

            # Range filter
            rng_ok: bool = OBSTACLE_MIN_RANGE <= rng <= OBSTACLE_MAX_RANGE

            # Costmap bounds check
            b_ok: str = "n/a"
            if self.costmap_origin and self.costmap_size and self.costmap_resolution:
                ox, oy = self.costmap_origin
                sx = self.costmap_size[0] * self.costmap_resolution
                sy = self.costmap_size[1] * self.costmap_resolution
                in_bounds = (ox <= gx <= ox + sx) and (oy <= gy <= oy + sy)
                b_ok = "YES" if in_bounds else "NO"
                if in_bounds:
                    bounds_pass += 1
                else:
                    bounds_fail += 1

            status = "PASS" if (h_ok and rng_ok) else "FILTERED"
            print(f"    [{idx:3d}] rng={rng:.2f}m  angle={math.degrees(angle):+7.1f}°  "
                  f"local=({local[0]:+.2f},{local[1]:+.2f},{local[2]:+.2f})  "
                  f"odom=({gx:+.2f},{gy:+.2f},{gz:+.2f})  "
                  f"height={h_ok}  range={rng_ok}  bounds={b_ok}  → {status}")

        # --- Full scan analysis ---
        print(f"\n  Full scan height analysis (all {in_obs_range} in-range points):")
        all_heights: list[float] = []
        for i, rng in enumerate(msg.ranges):
            if math.isinf(rng) or math.isnan(rng):
                continue
            if not (OBSTACLE_MIN_RANGE <= rng <= OBSTACLE_MAX_RANGE):
                continue
            angle = msg.angle_min + i * msg.angle_increment
            local = np.array([rng * math.cos(angle), rng * math.sin(angle), 0.0])
            gpt = transform_point(local, t.x, t.y, t.z, r.x, r.y, r.z, r.w)
            all_heights.append(float(gpt[2]))

        if all_heights:
            h_arr = np.array(all_heights)
            pass_count = int(np.sum((h_arr >= MIN_OBSTACLE_HEIGHT) &
                                     (h_arr <= MAX_OBSTACLE_HEIGHT)))
            print(f"    z_min={h_arr.min():.3f}  z_max={h_arr.max():.3f}  "
                  f"z_mean={h_arr.mean():.3f}")
            print(f"    pass height filter [{MIN_OBSTACLE_HEIGHT}, "
                  f"{MAX_OBSTACLE_HEIGHT}]: {pass_count}/{len(all_heights)}")
        else:
            print("    No valid in-range points!")

        # Costmap status
        if self.costmap_origin:
            print(f"\n  Costmap: origin=({self.costmap_origin[0]:.1f}, "
                  f"{self.costmap_origin[1]:.1f})  "
                  f"size={self.costmap_size}  res={self.costmap_resolution}")
            print(f"  Non-zero cells: {self.costmap_nonzero}")
        else:
            print(f"\n  Costmap: NOT YET RECEIVED")

        print(f"{'='*70}")


def main() -> None:
    rclpy.init()
    node = ScanDiagnosticNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
