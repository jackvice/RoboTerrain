#!/usr/bin/env python3
from typing import List, Tuple
import math

# Hardcoded triangle waypoints
WAYPOINTS: List[Tuple[float, float]] = [
    (-2.0, 2.0),
    (-5.4, -4.4),
    (-8.4, -1.0)
]

Z_HEIGHT: float = 1.0          # Actor's vertical pose
VELOCITY: float = 1.3          # m/s
OUTPUT_FILE: str = "triangle_trajectory.sdf"

def calculate_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])

def calculate_yaw(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])

def generate_trajectory() -> str:
    """Generate looped triangle trajectory XML."""
    time_accumulator = 0.0
    trajectory_lines: List[str] = ['<trajectory id="0" type="walk">']

    points = WAYPOINTS + [WAYPOINTS[0]]  # close the loop
    for i in range(len(points) - 1):
        x, y = points[i]
        next_x, next_y = points[i + 1]
        yaw = calculate_yaw((x, y), (next_x, next_y))
        dist = calculate_distance((x, y), (next_x, next_y))
        time_accumulator += dist / VELOCITY

        pose_str = f"{x:.2f} {y:.2f} {Z_HEIGHT:.2f} 0 0 {yaw:.6f}"
        time_str = f"{time_accumulator:.2f}"
        waypoint_block = f"""  <waypoint>
    <time>{time_str}</time>
    <pose>{pose_str}</pose>
  </waypoint>"""
        trajectory_lines.append(waypoint_block)

    trajectory_lines.append("</trajectory>")
    return "\n".join(trajectory_lines)

def write_to_file(content: str, filepath: str) -> None:
    with open(filepath, "w") as f:
        f.write(content)
    print(f"Trajectory written to: {filepath}")

if __name__ == "__main__":
    trajectory_xml = generate_trajectory()
    write_to_file(trajectory_xml, OUTPUT_FILE)
