#!/usr/bin/env python3
"""Nav2 LiDAR metrics collector for social navigation experiments."""

import csv
import math
import random
import time
from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Tuple

import rclpy
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from nav2_msgs.srv import ClearEntireCostmap
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from rclpy.parameter import Parameter
from rclpy.qos import QoSHistoryPolicy, QoSReliabilityPolicy, QoSProfile
from transforms3d.euler import quat2euler
from action_msgs.msg import GoalStatusArray, GoalStatus as ActionGoalStatus


# Type aliases
RobotXY = Tuple[float, float]
ActorXY = Optional[Tuple[float, float]]
GoalXY = Tuple[float, float]
Distance = Optional[float]


# Enums and Types
class GoalStatus(Enum):
    IDLE = "idle"
    WAITING = "waiting"
    ACHIEVED = "achieved"
    TIMEOUT = "timeout"
    FAILED = "failed"


class EnvironmentConfig(NamedTuple):
    world_name: str
    spawn_x: float
    spawn_y: float
    spawn_z: float
    goal_x_range: Tuple[float, float]
    goal_y_range: Tuple[float, float]
    goal_threshold: float
    goal_timeout_sec: float
    actor_topics: Dict[str, str]
    flip_threshold_rad: float = 1.48
    total_duration_min: int = 30
    costmap_clear_interval_sec: float = 0.0 # 2.0 #60.0


class MetricsCollectorState(NamedTuple):
    robot_xy: Optional[RobotXY]
    actor_positions: Dict[str, ActorXY]
    current_goal: Optional[GoalXY]
    goal_status: GoalStatus
    goal_start_time: float
    goals_succeeded: int
    goals_failed: int
    current_roll: float
    current_pitch: float
    last_robot_xy: Optional[RobotXY]
    last_metric_time: float
    discard_first_timeout: bool


class MetricsRow(NamedTuple):
    step: int
    time_s: float
    robot_x: float
    robot_y: float
    speed_mps: float
    actor_distances: Dict[str, Distance]
    dmin: Distance
    goals_count: int


# Environment configurations
ENVIRONMENT_CONFIGS: Dict[str, EnvironmentConfig] = {
    "construct": EnvironmentConfig(
        world_name="default",
        spawn_x=-7.0,
        spawn_y=0.82,
        spawn_z=0.75,
        goal_x_range=(-8.7, -5),
        goal_y_range=(-6, 3.5),
        goal_threshold=0.3,
        goal_timeout_sec=207.0,
        actor_topics={"lower_actor": "/lower_actor/pose", "upper_actor": "/upper_actor/pose"},
    ),
    "island": EnvironmentConfig(
        world_name="moon",
        spawn_x=-6.2,
        spawn_y=-1.52,
        spawn_z=1.0,
        goal_x_range=(-8.0, 8.0),
        goal_y_range=(-8.0, 8.0),
        goal_threshold=0.3,
        goal_timeout_sec=207.0,
        actor_topics={
            "triangle_actor": "/triangle_actor/pose",
            "triangle2_actor": "/triangle2_actor/pose",
            "triangle3_actor": "/triangle3_actor/pose",
        },
    ),
     "inspect": EnvironmentConfig(
        world_name="inspect",
        spawn_x=-19.5,
        spawn_y=-22.1,
        spawn_z=6.0,
        goal_x_range=(-27, -14),
        goal_y_range=(-25, -18),
        goal_threshold=0.3,
        goal_timeout_sec=207.0,
        actor_topics={
            "linear_actor": "/linear_actor/pose",
            "diag_actor": "/diag_actor/pose",
            "triangle_actor": "/triangle_actor/pose",
        },
    ),
}


# Pure functions: Message extraction
def extract_robot_xy(pose_array: PoseArray) -> Optional[RobotXY]:
    """Extract robot position from PoseArray."""
    if not pose_array.poses:
        return None
    p = pose_array.poses[0].position
    return (float(p.x), float(p.y))


def extract_orientation(pose: Pose) -> Tuple[float, float]:
    """Extract roll and pitch from pose quaternion."""
    q = pose.orientation
    quat = [float(q.w), float(q.x), float(q.y), float(q.z)]
    norm = math.sqrt(sum(v * v for v in quat))
    if norm > 0:
        quat = [v / norm for v in quat]
        roll, pitch, _ = quat2euler(quat, axes="sxyz")
        return (float(roll), float(pitch))
    return (0.0, 0.0)


def extract_actor_xy(pose: Pose) -> ActorXY:
    """Extract actor position from Pose."""
    p = pose.position
    return (float(p.x), float(p.y))


# Pure functions: Metrics computation
def compute_distance(p1: RobotXY, p2: RobotXY) -> float:
    """Compute 2D Euclidean distance."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def compute_speed(prev_xy: Optional[RobotXY], curr_xy: RobotXY, dt_sec: float) -> float:
    """Compute speed in m/s; return NaN if prev position unavailable."""
    if prev_xy is None or dt_sec <= 0:
        return float("nan")
    return compute_distance(prev_xy, curr_xy) / dt_sec


def compute_actor_distances(robot_xy: Optional[RobotXY], actor_positions: Dict[str, ActorXY]) -> Dict[str, Distance]:
    """Compute distances to all actors."""
    if robot_xy is None:
        return {actor_id: None for actor_id in actor_positions}
    return {actor_id: compute_distance(robot_xy, pos) if pos else None for actor_id, pos in actor_positions.items()}


def compute_min_distance(distances: Dict[str, Distance]) -> Distance:
    """Compute minimum distance across all actors."""
    dists = [d for d in distances.values() if d is not None]
    return min(dists) if dists else None


# Pure functions: Predicates
def is_goal_reached(robot_xy: Optional[RobotXY], goal: GoalXY, threshold: float) -> bool:
    """Check if robot is within threshold of goal."""
    if robot_xy is None:
        return False
    return compute_distance(robot_xy, goal) < threshold


def is_goal_timeout(elapsed_sec: float, threshold_sec: float) -> bool:
    """Check if goal has timed out."""
    return elapsed_sec > threshold_sec


def is_flipped(roll: float, pitch: float, threshold: float) -> bool:
    """Check if robot has flipped."""
    return abs(roll) > threshold or abs(pitch) > threshold


# Pure functions: State transitions
def update_robot_xy(state: MetricsCollectorState, xy: Optional[RobotXY]) -> MetricsCollectorState:
    """Update robot position."""
    return state._replace(robot_xy=xy)


def update_actor_xy(state: MetricsCollectorState, actor_id: str, xy: ActorXY) -> MetricsCollectorState:
    """Update actor position."""
    new_actors = dict(state.actor_positions)
    new_actors[actor_id] = xy
    return state._replace(actor_positions=new_actors)


def update_orientation(state: MetricsCollectorState, roll: float, pitch: float) -> MetricsCollectorState:
    """Update robot orientation."""
    return state._replace(current_roll=roll, current_pitch=pitch)




def on_goal_timeout(state: MetricsCollectorState, is_first: bool) -> MetricsCollectorState:
    """Handle goal timeout."""
    if is_first and state.discard_first_timeout:
        return state._replace(discard_first_timeout=False)
    return state._replace(goals_failed=state.goals_failed + 1, goal_status=GoalStatus.FAILED)


def on_robot_flipped(state: MetricsCollectorState) -> MetricsCollectorState:
    """Mark robot as failed due to flip."""
    return state._replace(goals_failed=state.goals_failed + 1, goal_status=GoalStatus.FAILED)


# Pure functions: Metrics assembly
def build_metrics_row(step: int, time_s: float, state: MetricsCollectorState) -> MetricsRow:
    """Build metrics row from state."""
    speed = compute_speed(state.last_robot_xy, state.robot_xy, time_s - state.last_metric_time) if state.robot_xy else float("nan")
    actor_dists = compute_actor_distances(state.robot_xy, state.actor_positions)
    dmin = compute_min_distance(actor_dists)

    return MetricsRow(
        step=step,
        time_s=time_s,
        robot_x=state.robot_xy[0] if state.robot_xy else float("nan"),
        robot_y=state.robot_xy[1] if state.robot_xy else float("nan"),
        speed_mps=speed,
        actor_distances=actor_dists,
        dmin=dmin,
        goals_count=state.goals_succeeded,
    )


# I/O functions
def write_metrics_header(writer: csv.writer, actor_ids: List[str]) -> None:
    """Write CSV header with actor distances in sorted order."""
    actor_cols = [f"{actor_id}_dist" for actor_id in sorted(actor_ids)]
    header = ["step", "time_s", "robot_x", "robot_y", "speed_mps"] + actor_cols + ["dmin", "goals_count"]
    writer.writerow(header)


def write_metrics_row(writer: csv.writer, row: MetricsRow) -> None:
    """Write metrics row to CSV with actor distances in sorted order."""
    sorted_actor_ids = sorted(row.actor_distances.keys())
    actor_dists = [row.actor_distances[actor_id] for actor_id in sorted_actor_ids]
    values = [
        row.step,
        f"{row.time_s:.2f}",
        f"{row.robot_x:.2f}",
        f"{row.robot_y:.2f}",
        f"{row.speed_mps:.2f}",
    ] + [f"{d:.2f}" if d is not None else "nan" for d in actor_dists] + [
        f"{row.dmin:.2f}" if row.dmin is not None else "nan",
        row.goals_count,
    ]
    writer.writerow(values)


def generate_random_goal(x_range: Tuple[float, float], y_range: Tuple[float, float]) -> GoalXY:
    """Generate random goal within ranges."""
    x = random.uniform(x_range[0], x_range[1])
    y = random.uniform(y_range[0], y_range[1])
    return (x, y)


def create_goal_msg(x: float, y: float, frame_id: str) -> PoseStamped:
    """Create goal PoseStamped message."""
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = 0.0
    yaw = random.uniform(-math.pi, math.pi)
    msg.pose.orientation.z = math.sin(yaw / 2.0)
    msg.pose.orientation.w = math.cos(yaw / 2.0)
    return msg


# ROS initialization
def init_state(config: EnvironmentConfig) -> MetricsCollectorState:
    """Initialize collector state."""
    actors = {actor_id: None for actor_id in config.actor_topics}
    first_goal = generate_random_goal(config.goal_x_range, config.goal_y_range)
    return MetricsCollectorState(
        robot_xy=None,
        actor_positions=actors,
        current_goal=first_goal,
        goal_status=GoalStatus.WAITING,
        #goal_start_time=time.time(),
        goal_start_time=0.0, # placeholder; will be set after sim time becomes active
        goals_succeeded=0,
        goals_failed=0,
        current_roll=0.0,
        current_pitch=0.0,
        last_robot_xy=None,
        last_metric_time=0.0,
        discard_first_timeout=False #True,
    )


class MetricsCollectorNode:
    """ROS node for metrics collection."""

    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.state = init_state(config)
        
        self.node = rclpy.create_node(
            "nav2_metrics_collector",
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )

        robot_qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        goal_qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)

        self.node.create_subscription(PoseArray, "/rover/pose_array", self._on_robot_pose, robot_qos)
        for actor_id, topic in config.actor_topics.items():
            self.node.create_subscription(Pose, topic, lambda msg,
                                          aid=actor_id: self._on_actor_pose(aid, msg), 10)

        self.goal_pub = self.node.create_publisher(PoseStamped, "/goal_pose", goal_qos)
        self.set_pose_client = self.node.create_client(SetEntityPose,
                                                       f"/world/{config.world_name}/set_pose")
        self.clear_global_costmap = self.node.create_client(ClearEntireCostmap,
                                                            "/global_costmap/clear_entirely_global_costmap")
        self.clear_local_costmap = self.node.create_client(ClearEntireCostmap,
                                                           "/local_costmap/clear_entirely_local_costmap")

        self.nav2_goal_failed: bool = False
        self.nav2_goal_succeeded: bool = False
        self.controller_abort_hard: bool = False

        self.node.create_subscription(
            GoalStatusArray,
            '/navigate_to_pose/_action/status',
            self._on_nav_status,
            10,
        )

        self.controller_abort_count: int = 0
        self.node.create_subscription(
            GoalStatusArray,
            '/follow_path/_action/status',
            self._on_controller_status,
            10,
        )

    def _on_robot_pose(self, msg: PoseArray) -> None:
        """Update robot position and orientation."""
        xy = extract_robot_xy(msg)
        self.state = update_robot_xy(self.state, xy)
        if xy and msg.poses:
            roll, pitch = extract_orientation(msg.poses[0])
            self.state = update_orientation(self.state, roll, pitch)

    def _on_actor_pose(self, actor_id: str, msg: Pose) -> None:
        """Update actor position."""
        xy = extract_actor_xy(msg)
        self.state = update_actor_xy(self.state, actor_id, xy)

    def send_goal(self) -> None:
        """Send new navigation goal."""
        goal_xy = generate_random_goal(self.config.goal_x_range, self.config.goal_y_range)
        msg = create_goal_msg(goal_xy[0], goal_xy[1], "odom")
        msg.header.stamp = self.node.get_clock().now().to_msg()
        self.goal_pub.publish(msg)
        self.state = self.state._replace(current_goal=goal_xy, goal_status=GoalStatus.WAITING,
                                         goal_start_time=self.node.get_clock().now().nanoseconds / 1e9)
        self.controller_abort_count = 0
        self.nav2_goal_failed = False
        self.nav2_goal_succeeded = False
        print(f"New goal: ({goal_xy[0]:.2f}, {goal_xy[1]:.2f})")


    def respawn_robot(self) -> None:
        """Respawn robot at spawn location."""
        if not self.set_pose_client.service_is_ready():
            self.set_pose_client.wait_for_service(timeout_sec=1.0)

        req = SetEntityPose.Request()
        req.entity.name = "leo_rover"
        req.entity.type = Entity.MODEL
        req.pose.position.x = self.config.spawn_x
        req.pose.position.y = self.config.spawn_y
        req.pose.position.z = self.config.spawn_z
        req.pose.orientation.x = 0.0
        req.pose.orientation.y = 0.0
        req.pose.orientation.z = 0.0
        req.pose.orientation.w = 1.0

        fut = self.set_pose_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, fut, timeout_sec=2.0)

        # Spin to let odom and costmap update to new position
        for _ in range(50):
            rclpy.spin_once(self.node, timeout_sec=0.05)

        if self.clear_local_costmap.service_is_ready():
            self.clear_local_costmap.call_async(ClearEntireCostmap.Request())

        # Spin again after clear to let costmap re-center
        for _ in range(50):
            rclpy.spin_once(self.node, timeout_sec=0.05)
        
    def clear_costmaps(self) -> None:
        """Clear costmaps."""
        if self.clear_global_costmap.service_is_ready():
            self.clear_global_costmap.call_async(ClearEntireCostmap.Request())
        if self.clear_local_costmap.service_is_ready():
            self.clear_local_costmap.call_async(ClearEntireCostmap.Request())

    def spin_once(self) -> None:
        """Process one ROS callback."""
        rclpy.spin_once(self.node, timeout_sec=0.01)
        
    def _on_nav_status(self, msg: GoalStatusArray) -> None:
        """Detect Nav2 goal abort/failure and success."""
        if msg.status_list:
            latest = msg.status_list[-1]
            if latest.status in (5, 6):  # CANCELED or ABORTED
                self.nav2_goal_failed = True
                self.controller_abort_hard = False
            elif latest.status == 4:  # SUCCEEDED
                self.nav2_goal_succeeded = True
                self.controller_abort_count = 0
            elif latest.status == 2:  # ACTIVE
                self.controller_abort_count = 0

    def _on_controller_status(self, msg: GoalStatusArray) -> None:
        """Count consecutive controller aborts."""
        if msg.status_list:
            latest = msg.status_list[-1]
            if latest.status == 6:  # STATUS_ABORTED
                self.controller_abort_count += 1
                if self.controller_abort_count >= 2:
                    self.nav2_goal_failed = True
                    self.controller_abort_hard = True
                    self.controller_abort_count = 0


def main(env_name: str = "construct") -> None:
    """Main metrics collection loop."""
    config = ENVIRONMENT_CONFIGS[env_name]
    rclpy.init()
    collector = MetricsCollectorNode(config)

    csv_path = f"metrics_data/{env_name}/Nav2_lidar/{env_name}_nav2_{time.strftime('%m_%d_%H-%M')}.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    write_metrics_header(writer, sorted(config.actor_topics.keys()))
    csv_file.flush()

    # Wait for Nav2 and sim time
    print("Waiting for Nav2 to connect...")
    while collector.goal_pub.get_subscription_count() == 0:
        rclpy.spin_once(collector.node, timeout_sec=0.1)
    print("Nav2 connected.")

    print("Waiting for sim time...")
    while collector.node.get_clock().now().nanoseconds == 0:
        rclpy.spin_once(collector.node, timeout_sec=0.1)
    print(f"Sim time active: {collector.node.get_clock().now().nanoseconds / 1e9:.2f}s")
    
    start_sim_ns: int = collector.node.get_clock().now().nanoseconds
    duration_ns: int = int(config.total_duration_min * 60 * 1e9)

    next_log_ns: int = start_sim_ns + int(1e9)
    last_clear_sim_ns: int = start_sim_ns
    clear_interval_ns: int = int(config.costmap_clear_interval_sec * 1e9)

    collector.state = collector.state._replace(goal_start_time=start_sim_ns / 1e9)

    print("Respawning robot at configured spawn...")
    collector.respawn_robot()
    time.sleep(5)
    collector.send_goal()

    step: int = 0
    duration_s: float = config.total_duration_min * 60

    try:
        while (collector.node.get_clock().now().nanoseconds - start_sim_ns) < duration_ns:
            collector.spin_once()
            now_sim_ns: int = collector.node.get_clock().now().nanoseconds
            now: float = now_sim_ns / 1e9  # keep existing code that uses 'now' as seconds

            # Check flip
            if is_flipped(collector.state.current_roll, collector.state.current_pitch, config.flip_threshold_rad):
                collector.state = on_robot_flipped(collector.state)
                print(f"FLIPPED! Respawning... Total: {collector.state.goals_succeeded}, Failed: {collector.state.goals_failed}")
                collector.respawn_robot()
                collector.send_goal()
                continue

            # Check Nav2 success (trust Nav2's goal checker)
            if collector.nav2_goal_succeeded:
                collector.nav2_goal_succeeded = False
                collector.state = collector.state._replace(goals_succeeded=collector.state.goals_succeeded + 1)
                print(f"Goal reached (Nav2)! Total: {collector.state.goals_succeeded}, Failed: {collector.state.goals_failed}")
                collector.send_goal()
            # Check goal reached by distance (backup
            elif collector.state.current_goal and collector.state.robot_xy:
                if is_goal_reached(collector.state.robot_xy, collector.state.current_goal, config.goal_threshold):
                    collector.state = collector.state._replace(goals_succeeded=collector.state.goals_succeeded + 1)
                    print(f"Goal reached! Total: {collector.state.goals_succeeded}, Failed: {collector.state.goals_failed}")
                    collector.send_goal()
                elif is_goal_timeout(now - collector.state.goal_start_time, config.goal_timeout_sec):
                    is_first: bool = collector.state.discard_first_timeout
                    collector.state = on_goal_timeout(collector.state, is_first)
                    if is_first:
                        print("Discarding first goal timeout (startup); sending new goal.")
                    else:
                        print(f"Goal timeout! Respawning... Total: {collector.state.goals_succeeded}, Failed: {collector.state.goals_failed}")
                        collector.respawn_robot()
                    collector.send_goal()

            # Check Nav2 abort (immediate re-goal without waiting for 207s timeout)
            if collector.nav2_goal_failed:
                collector.nav2_goal_failed = False
                collector.state = collector.state._replace(goals_failed=collector.state.goals_failed + 1)

                if collector.controller_abort_hard:
                    collector.controller_abort_hard = False
                    print(f"Nav2 controller aborted twice — respawning. Total: {collector.state.goals_succeeded}, Failed: {collector.state.goals_failed}")
                    collector.respawn_robot()
                else:
                    print(f"Nav2 goal aborted — clearing + re-goal. Total: {collector.state.goals_succeeded}, Failed: {collector.state.goals_failed}")
                    if collector.clear_local_costmap.service_is_ready():
                        collector.clear_local_costmap.call_async(ClearEntireCostmap.Request())

                collector.send_goal()
                
            # Log metrics at 1 Hz
            if now_sim_ns >= next_log_ns:
                step += 1
                row = build_metrics_row(step, (now_sim_ns - start_sim_ns) / 1e9, collector.state)
                write_metrics_row(writer, row)
                csv_file.flush()
                collector.state = collector.state._replace(
                    last_robot_xy=collector.state.robot_xy,
                    last_metric_time=(now_sim_ns - start_sim_ns) / 1e9,  # was: now
                )
                next_log_ns += int(1e9)

            # Clear costmaps periodically
            if clear_interval_ns > 0 and (now_sim_ns - last_clear_sim_ns) > clear_interval_ns:
                collector.clear_costmaps()
                last_clear_sim_ns = now_sim_ns


    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        collector.node.destroy_node()
        rclpy.shutdown()
        csv_file.close()
        print(f"Wrote metrics to {csv_path}")
        print(f"Final: {collector.state.goals_succeeded} succeeded, {collector.state.goals_failed} failed")

        

if __name__ == "__main__":
    import sys
    env = sys.argv[1] if len(sys.argv) > 1 else "construct"
    main(env)
