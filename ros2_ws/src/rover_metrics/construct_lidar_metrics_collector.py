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
from rclpy.parameter import Parameter
from nav2_msgs.srv import ClearEntireCostmap
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy



ActorXY = Optional[Tuple[float, float]]
actor1_xy: ActorXY = None  # /upper_actor/pose
actor2_xy: ActorXY = None  # /lower_actor/pose

def on_actor1_pose(msg: Pose) -> None:
    """Store actor1 (x,y) from //upper_actor/pose."""
    global actor1_xy
    actor1_xy = (float(msg.position.x), float(msg.position.y))

def on_actor2_pose(msg: Pose) -> None:
    """Store actor2 (x,y) from /lower_actor/pose."""
    global actor2_xy
    actor2_xy = (float(msg.position.x), float(msg.position.y))
    
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


def generate_random_goal(
    x_range: Tuple[float, float], 
    y_range: Tuple[float, float]
) -> Tuple[float, float]:
    """Generate random x, y within specified ranges."""
    import random
    x = random.uniform(x_range[0], x_range[1])
    y = random.uniform(y_range[0], y_range[1])
    return (x, y)


def create_goal_msg(x: float, y: float, frame_id: str) -> PoseStamped:
    """Create PoseStamped goal message with random yaw."""
    import random
    import math
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = 0.0
    yaw = random.uniform(-math.pi, math.pi)
    msg.pose.orientation.z = math.sin(yaw / 2.0)
    msg.pose.orientation.w = math.cos(yaw / 2.0)
    return msg


def check_goal_reached(
    robot_xy: Optional[Tuple[float, float]], 
    goal_xy: Tuple[float, float], 
    threshold: float
) -> bool:
    """Return True if robot is within threshold of goal."""
    if robot_xy is None:
        return False
    dx = robot_xy[0] - goal_xy[0]
    dy = robot_xy[1] - goal_xy[1]
    return math.hypot(dx, dy) < threshold


def main() -> None:
    """Collect metrics at 1 Hz and send random navigation goals."""
    import csv, math, time, random
    from typing import Optional, Tuple
    import rclpy
    from geometry_msgs.msg import PoseArray, Pose, PoseStamped
    from std_msgs.msg import String

    # --- island/moon config --------------------------------------------------------------
    TOTAL_MINUTES = 30 #33 for the extra 207 seconds for first goal 90
    GOAL_X_RANGE = (-8.7, -5) 
    GOAL_Y_RANGE = (-6, 3.5)
    GOAL_THRESHOLD = 0.3
    GOAL_TIMEOUT = 207.0
    WORLD_NAME = "default" # default for construct  

    # --- helpers (local) -----------------------------------------------------
    def safe_float(x: Optional[float]) -> float:
        return float('nan') if x is None else float(x)

    def distance2d(a: Optional[Tuple[float, float]],
                   b: Optional[Tuple[float, float]]) -> Optional[float]:
        if a is None or b is None:
            return None
        return math.hypot(a[0] - b[0], a[1] - b[1])

    # --- CSV setup -----------------------------------------------------------
    csv_path = f"metrics_data/construct_csv/Nav2_lidar/Nav2_lidar_{WORLD_NAME}_{time.strftime('%m_%d_%H-%M')}.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["step", "time_s", "rx", "ry", "speed_mps",
                     "d1_lower", "d2_upper", "dmin", "goals"])
    #writer.writerow(["step", "time_s", "rx", "ry", "speed_mps",  # island
    #                 "d1_triangle", "d2_triangle2", "d3_triangle3", "dmin", "goals"])
    csv_file.flush()

    # --- state ---------------------------------------------------------------
    robot_xy: Optional[Tuple[float, float]] = None
    last_robot_xy: Optional[Tuple[float, float]] = None
    last_ts: Optional[float] = None
    actor1_xy: Optional[Tuple[float, float]] = None
    actor2_xy: Optional[Tuple[float, float]] = None
    #actor3_xy: Optional[Tuple[float, float]] = None
    goals_count: int = 0
    goals_failed: int = 0
    current_goal_xy: Optional[Tuple[float, float]] = None
    goal_start_time: float = 0.0

    # --- ROS setup -----------------------------------------------------------
    rclpy.init()
    
    #node = rclpy.create_node("metrics_collector")
    
    node = rclpy.create_node(
        "metrics_collector",
        parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
    )


    goal_qos = QoSProfile(
        depth=10,
        reliability=QoSReliabilityPolicy.BEST_EFFORT
    )
    
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


    robot_qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
    node.create_subscription(PoseArray, '/rover/pose_array', on_robot_pose, robot_qos)
    node.create_subscription(Pose, "/lower_actor/pose", on_actor1_pose, 10)
    node.create_subscription(Pose, "/upper_actor/pose", on_actor2_pose, 10)
    
    goal_pub = node.create_publisher(PoseStamped, '/goal_pose', goal_qos)

    # After node creation, add service clients:
    clear_global_costmap = node.create_client(
        ClearEntireCostmap, 
        '/global_costmap/clear_entirely_global_costmap'
    )
    clear_local_costmap = node.create_client(
        ClearEntireCostmap,
        '/local_costmap/clear_entirely_local_costmap'
    )
    
    # --- goal helper ---------------------------------------------------------
    def send_new_goal() -> Tuple[float, float]:
        nonlocal current_goal_xy, goal_start_time
        current_goal_xy = generate_random_goal(GOAL_X_RANGE, GOAL_Y_RANGE)
        msg = create_goal_msg(current_goal_xy[0], current_goal_xy[1], "odom")
        msg.header.stamp = node.get_clock().now().to_msg()
        goal_pub.publish(msg)
        goal_start_time = time.time()
        print(f"New goal: ({current_goal_xy[0]:.2f}, {current_goal_xy[1]:.2f})")
        return current_goal_xy

    # --- main loop -----------------------------------------------------------
    start = time.time()
    next_log = start + 1.0
    step = 0
    duration_s = TOTAL_MINUTES * 60

    print("Waiting for Nav2 to connect...")
    while goal_pub.get_subscription_count() == 0:
        rclpy.spin_once(node, timeout_sec=0.1)
    print("Nav2 connected.")

    #Wait for valid sim time
    print("Waiting for sim time...")
    while node.get_clock().now().nanoseconds == 0:
        rclpy.spin_once(node, timeout_sec=0.1)
    print(f"Sim time active: {node.get_clock().now().nanoseconds / 1e9:.2f}s")

    send_new_goal()
    time.sleep(5)
    goal_start_time = time.time() - GOAL_TIMEOUT - 1  # Force immediate timeout
    goals_failed = -1 
    
    # In main loop, add periodic clear (e.g., every 2 or 3 seconds):
    COSTMAP_CLEAR_INTERVAL = 3.0
    last_clear_time = time.time()
    
    try:
        while (time.time() - start) < duration_s:
            rclpy.spin_once(node, timeout_sec=0.01)
            now = time.time()

            # Goal checking
            if current_goal_xy is not None:
                if check_goal_reached(robot_xy, current_goal_xy, GOAL_THRESHOLD):
                    goals_count += 1
                    print(f"Goal reached! Total: {goals_count}, Failed: {goals_failed}")
                    send_new_goal()
                elif (now - goal_start_time) > GOAL_TIMEOUT:
                    goals_failed += 1
                    print(f"Goal timeout! Total: {goals_count}, Failed: {goals_failed}")
                    send_new_goal()

            # Logging at 1 Hz
            if now >= next_log:
                step += 1
                speed = float('nan')
                if robot_xy is not None and last_robot_xy is not None and last_ts is not None:
                    dt = now - last_ts
                    if dt > 0.0:
                        speed = math.hypot(robot_xy[0] - last_robot_xy[0], 
                                          robot_xy[1] - last_robot_xy[1]) / dt

                d1 = distance2d(actor1_xy, robot_xy)
                d2 = distance2d(actor2_xy, robot_xy)
                dists = [d for d in (d1, d2, d3) if d is not None]
                dmin = min(dists) if dists else None

                writer.writerow([
                    step, f"{now:.2f}",
                    safe_float(robot_xy[0]) if robot_xy else float('nan'),
                    safe_float(robot_xy[1]) if robot_xy else float('nan'),
                    safe_float(speed),
                    safe_float(d1), safe_float(d2),
                    safe_float(dmin),
                    goals_count,
                ])
                csv_file.flush()

                last_robot_xy = robot_xy
                last_ts = now
                next_log += 1.0
            # Inside the while loop:
            if (now - last_clear_time) > COSTMAP_CLEAR_INTERVAL:
                if clear_global_costmap.service_is_ready():
                    clear_global_costmap.call_async(ClearEntireCostmap.Request())
                if clear_local_costmap.service_is_ready():
                    clear_local_costmap.call_async(ClearEntireCostmap.Request())
                last_clear_time = now


    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        csv_file.close()
        print(f"Wrote metrics to {csv_path}")
        print(f"Final: {goals_count} succeeded, {goals_failed} failed")


if __name__ == '__main__':
    main()

