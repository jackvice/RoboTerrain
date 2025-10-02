#!/usr/bin/env python3
"""
Simple metrics collector for 20-minute robot evaluation.
Records velocity, actor distances, and goal count to CSV.
"""

from typing import List, Tuple, Optional
import math
import time
import csv
import rclpy
from std_msgs.msg import String
from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from typing import Optional, Tuple

# add near your imports
from rclpy.qos import QoSProfile, QoSReliabilityPolicy


ActorXY = Optional[Tuple[float, float]]
actor1_xy: ActorXY = None  # /triangle_actor/pose
actor2_xy: ActorXY = None  # /triangle2_actor/pose
actor3_xy: ActorXY = None  # /triangle3_actor/pose

def on_actor1_pose(msg: Pose) -> None:
    """Store actor1 (x,y) from //triangle_actor/pose."""
    global actor1_xy
    actor1_xy = (float(msg.position.x), float(msg.position.y))

def on_actor2_pose(msg: Pose) -> None:
    """Store actor2 (x,y) from /triangle2_actor/pose."""
    global actor2_xy
    actor2_xy = (float(msg.position.x), float(msg.position.y))

def on_actor3_pose(msg: Pose) -> None:
    """Store actor3 (x,y) from /triangle3_actor/pose."""
    global actor3_xy
    actor3_xy = (float(msg.position.x), float(msg.position.y))
    

def calculate_distance(pos1: Tuple[float, float, float], 
                      pos2: Tuple[float, float, float]) -> float:
    """Calculate 2D distance between two positions."""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def calculate_velocity(pos1: Tuple[float, float, float], 
                      pos2: Tuple[float, float, float], 
                      time_diff: float) -> float:
    """Calculate velocity from position change over time."""
    if time_diff <= 0:
        return 0.0
    distance = calculate_distance(pos1, pos2)
    return distance / time_diff


def write_csv_header(filepath: str) -> None:
    """Write CSV header row."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'timestamp', 'robot_x', 'robot_y', 'robot_z', 
            'velocity_mps', 'actor1_dist', 'actor2_dist', 'actor3_dist', 'goals_total'
        ])


def write_csv_row(filepath: str, timestamp: float, robot_pos: Tuple[float, float, float],
                 velocity: float, actor_distances: List[float], goals_count: int) -> None:
    """Write one data row to CSV."""
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            f"{timestamp:.2f}", f"{robot_pos[0]:.3f}", f"{robot_pos[1]:.3f}", f"{robot_pos[2]:.3f}", f"{robot_pos[3]:.3f}",
            f"{velocity:.3f}", f"{actor_distances[0]:.3f}", f"{actor_distances[1]:.3f}", goals_count
        ])


def distance2d(a: ActorXY, b: ActorXY) -> Optional[float]:
    """Euclidean 2D distance or None if either is missing."""
    if a is None or b is None:
        return None
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.hypot(dx, dy)

def safe_float(x: Optional[float]) -> float:
    """Replace None with NaN for CSV/plotting convenience."""
    return float('nan') if x is None else float(x)



def main() -> None:
    """Collect total_minutes  of metrics at 1 Hz and write a timestamped CSV."""

    import csv, math, time
    from typing import Optional, Tuple
    import rclpy
    from geometry_msgs.msg import PoseArray, Pose
    from std_msgs.msg import String

    total_minutes = 30
    
    # --- helpers -------------------------------------------------------------
    def safe_float(x: Optional[float]) -> float:
        return float('nan') if x is None else float(x)

    def distance2d(a: Optional[Tuple[float, float]],
                   b: Optional[Tuple[float, float]]) -> Optional[float]:
        if a is None or b is None:
            return None
        dx, dy = a[0] - b[0], a[1] - b[1]
        return math.hypot(dx, dy)

    # --- config & file -------------------------------------------------------
    world_name = "inspect"
    csv_path = f"no_heat_metrics_{world_name}_{time.strftime('%m_%d_%H-%M')}.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    # CSV header (write once)
    writer.writerow(["step", "time_s", "rx", "ry", "speed_mps",
                     "d1_triangle", "d2_triangle2","d3_triangle3", "dmin", "goals"])
    csv_file.flush()

    # --- state ---------------------------------------------------------------
    robot_xy: Optional[Tuple[float, float]] = None
    last_robot_xy: Optional[Tuple[float, float]] = None
    last_ts: Optional[float] = None

    actor1_xy: Optional[Tuple[float, float]] = None  # /triangle_actor/pose
    actor2_xy: Optional[Tuple[float, float]] = None  # /triangle2_actor/pose
    actor3_xy: Optional[Tuple[float, float]] = None  # /triangle3_actor/pose
    goals_count: int = 0

    # --- ROS setup -----------------------------------------------------------
    rclpy.init()
    node = rclpy.create_node("metrics_collector")

    def on_robot_pose(msg: PoseArray) -> None:
        nonlocal robot_xy
        if msg.poses:
            p = msg.poses[0].position
            robot_xy = (float(p.x), float(p.y))

    def on_actor1_pose(msg: Pose) -> None:
        nonlocal actor1_xy
        actor1_xy = (float(msg.position.x), float(msg.position.y))

    def on_actor2_pose(msg: Pose) -> None:
        nonlocal actor2_xy
        actor2_xy = (float(msg.position.x), float(msg.position.y))

    def on_actor3_pose(msg: Pose) -> None:
        nonlocal actor3_xy
        actor3_xy = (float(msg.position.x), float(msg.position.y))

    def on_event(msg: String) -> None:
        nonlocal goals_count
        if "goal_reached" in msg.data.lower():
            goals_count += 1


    robot_qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
    robot_sub = node.create_subscription(PoseArray, '/rover/pose_array', on_robot_pose, robot_qos)
    
    node.create_subscription(Pose, "/triangle_actor/pose", on_actor1_pose, 10)
    node.create_subscription(Pose, "/triangle2_actor/pose", on_actor2_pose, 10)
    node.create_subscription(Pose, "/triangle3_actor/pose", on_actor3_pose, 10)
    node.create_subscription(String, "/robot/events", on_event, 10)

    # --- loop (1 Hz for 20 minutes) -----------------------------------------
    start = time.time()
    next_log = start + 1.0
    step = 0
    duration_s = total_minutes * 60

    try:
        while (time.time() - start) < duration_s:
            rclpy.spin_once(node, timeout_sec=0.01)
            now = time.time()
            if now >= next_log:
                step += 1

                # speed (m/s) from last position
                speed = float('nan')
                if robot_xy is not None and last_robot_xy is not None and last_ts is not None:
                    dt = now - last_ts
                    if dt > 0.0:
                        dx = robot_xy[0] - last_robot_xy[0]
                        dy = robot_xy[1] - last_robot_xy[1]
                        speed = math.hypot(dx, dy) / dt

                d1 = distance2d(actor1_xy, robot_xy)
                d2 = distance2d(actor2_xy, robot_xy)
                d3 = distance2d(actor3_xy, robot_xy)
                dmin = None if (d1 is None and d2 is None and d3 is None) else min([d for d in (d1, d2, d3) if d is not None])

                writer.writerow([
                    step, now,
                    safe_float(robot_xy[0]) if robot_xy else float('nan'),
                    safe_float(robot_xy[1]) if robot_xy else float('nan'),
                    safe_float(speed),
                    safe_float(d1), safe_float(d2), safe_float(d3),
                    safe_float(dmin),
                    int(goals_count),
                ])
                csv_file.flush()

                last_robot_xy = robot_xy
                last_ts = now
                next_log += 1.0


    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()
        csv_file.close()
        print(f"Wrote metrics to {csv_path}")



if __name__ == '__main__':
    main()
