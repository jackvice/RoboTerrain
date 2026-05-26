#!/usr/bin/env python3
"""
nav2_diag.py — Tagged diagnostic logger for Nav2 / Nav2CAN runs.

Subscribes to the Nav2 stack and emits ONE tagged line per event so the
output is greppable after a run.

Usage
-----
    # Run in a separate terminal from the main launch, after the stack is up:
    python3 src/rover_metrics/nav2_diag.py 2>&1 | tee nav2_diag_$(date +%Y%m%d_%H%M%S).log

Tags emitted
------------
    [INIT]                startup
    [STATE]               1 Hz snapshot: pos, cmd_vel, actual_vel, footprint cost,
                          ped dist, AND closest /scan return + bearing
    [GOAL_START]          new goal received on /goal_pose
    [GOAL_OK]             goal succeeded
    [GOAL_FAIL]           goal aborted/cancelled (followed by [CONTEXT] dumps)
    [STUCK_START]         |cmd_vel| > 0.05 m/s but actual speed < 0.05 m/s for > 2 s
    [STUCK_END]           stuck condition cleared
    [FWD_COLLISION]       emitted alongside STUCK_START when cmd_vx > +0.05
                          (the robot was being driven *forward* into
                          something it can't pass).  Carries a RCA payload:
                          min_scan range + bearing, local + global costmap
                          cost at robot + 0.3 / 0.6 / 1.0 m in the
                          heading direction, active BT node, distance to
                          goal.  Interpret the dump per the key in
                          _log_fwd_collision_context's docstring.
    [HIGH_COST]           global costmap cost under robot crossed 200
    [LETHAL_FOOT]         global costmap cost under robot >= 253
    [CLOSE_PED]           distance to nearest pedestrian dropped below 0.5 m
    [OBSTACLE_INVISIBLE]  /scan sees a return within INVISIBLE_RANGE_MAX of the
                          robot but the local_costmap cell at that return's xy
                          has cost < INVISIBLE_COST_MAX.  This is the "thin
                          post the costmap doesn't know about" failure mode
                          that drives the robot directly into objects.
                          Throttled to <= 1 per second.
    [BACKUP_START]        cmd_vel.linear.x has been < BACKUP_ENTER_VX for at
                          least BACKUP_DEBOUNCE_S.  Logs the BT node that is
                          currently active so we can distinguish DWB-chosen
                          reverse during FollowPath (bad) from recovery-issued
                          reverse during BackUp / Spin (expected).
    [BACKUP_END]          cmd_vel.linear.x came back above BACKUP_EXIT_VX.
                          Includes duration and integrated reverse distance.
    [BT]                  behavior_tree_log transition (recovery-relevant nodes only)
    [ROSOUT]              Nav2 planner/controller warning matching a keyword filter
    [CONTEXT]             last 10 s of [STATE] history, dumped after a [GOAL_FAIL]

Post-run analysis cookbook
--------------------------
    grep '\\[GOAL_FAIL\\]'  nav2_diag*.log | wc -l            # total failures
    grep '\\[STUCK_START\\]' nav2_diag*.log | wc -l           # stuck events
    grep '\\[ROSOUT\\] .*lethal'  nav2_diag*.log | wc -l      # lethal-start aborts
    grep '\\[ROSOUT\\] .*No valid' nav2_diag*.log | wc -l     # DWB failures
    grep -B1 '\\[GOAL_FAIL\\]' nav2_diag*.log                 # what happened just before each failure
    grep '\\[CLOSE_PED\\]'   nav2_diag*.log                   # all 0.5 m proxemic crossings
    awk '/\\[STATE\\]/ {print $NF}' nav2_diag*.log | sort -n  # distribution of ped distances
"""

import math
import re
import sys
import time
from collections import deque
from typing import Optional, Tuple

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy, qos_profile_sensor_data)
from rclpy.time import Time

import tf2_ros

from action_msgs.msg import GoalStatus, GoalStatusArray
from geometry_msgs.msg import PoseArray, PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
from nav2_msgs.msg import BehaviorTreeLog
from rcl_interfaces.msg import Log
from sensor_msgs.msg import LaserScan

try:
    from multi_person_tracker_interfaces.msg import People
    HAVE_PEOPLE = True
except ImportError:
    HAVE_PEOPLE = False


# Keywords to capture from /rosout, indexed by tag suffix.
ROSOUT_FILTERS = {
    'lethal_start': re.compile(r'lethal space', re.IGNORECASE),
    'no_valid_traj': re.compile(r'No valid trajectories', re.IGNORECASE),
    'patience': re.compile(r'patience exceeded', re.IGNORECASE),
    'planner_rate': re.compile(r'Planner loop missed.*desired rate', re.IGNORECASE),
    'plan_failed': re.compile(r'failed to (create plan|generate.*path)', re.IGNORECASE),
    'controller_abort': re.compile(r'\[ActionServer\] Aborting', re.IGNORECASE),
    'transform_old': re.compile(r'Transform data.*too old', re.IGNORECASE),
}

# Goal status codes (action_msgs/msg/GoalStatus)
GOAL_STATUS_NAME = {
    GoalStatus.STATUS_UNKNOWN:    'UNKNOWN',
    GoalStatus.STATUS_ACCEPTED:   'ACCEPTED',
    GoalStatus.STATUS_EXECUTING:  'EXECUTING',
    GoalStatus.STATUS_CANCELING:  'CANCELING',
    GoalStatus.STATUS_SUCCEEDED:  'SUCCEEDED',
    GoalStatus.STATUS_CANCELED:   'CANCELED',
    GoalStatus.STATUS_ABORTED:    'ABORTED',
}

STUCK_VEL_THRESHOLD = 0.05      # m/s
STUCK_DURATION_THRESHOLD = 2.0  # s before declaring stuck

# FWD_COLLISION is logged when a STUCK_START fires AND the most recent
# commanded vx is at least this positive — i.e. "commanded forward, not
# moving, for >= STUCK_DURATION_THRESHOLD seconds".  The 0.05 m/s floor
# matches STUCK_VEL_THRESHOLD on purpose so the two detectors agree on
# what "commanded to move" means.
FWD_COLLISION_VX_THRESHOLD = 0.05  # m/s

# Distances (in metres) ahead of the robot to sample for RCA when a
# FWD_COLLISION event is logged.  Three points along the heading vector
# let us tell apart (a) costmap saw the obstacle right up against the
# robot, (b) costmap saw it at medium range, (c) costmap is completely
# blind to it.  These are world-frame offsets, computed from
# robot_xy + d * (cos yaw, sin yaw).
FWD_COLLISION_FWD_SAMPLES_M = (0.3, 0.6, 1.0)
HIGH_COST_THRESHOLD = 200       # uint8 costmap cost
LETHAL_COST_THRESHOLD = 253     # INSCRIBED_INFLATED_OBSTACLE and above
CLOSE_PED_THRESHOLD = 0.5       # m

STATE_RING_SECONDS = 10.0       # how much history to dump on failure

# ----- OBSTACLE_INVISIBLE thresholds -----
# Self-returns from the Leo's camera mast / IMU mast can reach ~0.20 m at the
# lidar.  Anything closer than this is filtered out as a self-return.
SCAN_SELF_RETURN_MIN = 0.20     # m
# A return must be at most this close to the robot to be considered a candidate
# for "imminent collision the costmap doesn't see."  Beyond this, the cost
# inflation gradient is supposed to be doing the work and false positives spike.
INVISIBLE_RANGE_MAX = 0.60      # m
# If the local_costmap cost at the beam endpoint is BELOW this, the costmap
# essentially treats that cell as free.  100 ~ moderate inflation;  50 is well
# below that.  Tune up if too few events; down if too many.
INVISIBLE_COST_MAX = 50         # 0..255 scale
# Don't spam more than one OBSTACLE_INVISIBLE per this many seconds.
INVISIBLE_LOG_THROTTLE = 1.0    # s
# Process every Nth scan to keep CPU low.  10 Hz lidar / 2 -> ~5 Hz processing.
SCAN_PROCESS_PERIOD = 0.18      # s

# ----- BACKUP detection -----
# Enter "pending backup" when commanded linear x falls below this value.
BACKUP_ENTER_VX = -0.05         # m/s
# Exit any backup-related state once commanded linear x rises above this value.
BACKUP_EXIT_VX = -0.01          # m/s
# Must remain in pending backup for at least this long before we log
# BACKUP_START.  Filters out cmd_vel jitter and 100 ms reverse blips.
BACKUP_DEBOUNCE_S = 0.3


class Nav2Diagnostic(Node):
    def __init__(self):
        super().__init__('nav2_diag')

        # State
        self.robot_xy: Optional[Tuple[float, float]] = None
        self.robot_yaw: Optional[float] = None       # rad, world frame
        self.robot_speed: float = 0.0
        self.last_pose_time: float = 0.0
        self.cmd_vx: float = 0.0
        self.cmd_wz: float = 0.0
        self.global_costmap: Optional[OccupancyGrid] = None
        self.local_costmap: Optional[OccupancyGrid] = None
        self.people = []
        self.current_goal: Optional[Tuple[float, float]] = None
        # Time the most recent /goal_pose was published.  Kept as a
        # fallback only — duration in GOAL_OK / GOAL_FAIL prefers the
        # per-UUID start time captured the first time the action
        # server reported a status for that UUID (see
        # goal_start_time_per_uuid below).  Before the per-UUID
        # tracking was added, this single field was overwritten on
        # every new goal, which made every GOAL_FAIL log
        # duration ≈ 0 s for the *previous* goal whenever the
        # collector issued a new goal during a fail-and-respawn — a
        # major source of confusion when reading analysis logs.
        self.goal_start_time: float = 0.0
        self.last_goal_status_per_uuid: dict = {}
        self.goal_start_time_per_uuid: dict = {}
        self.stuck_since: Optional[float] = None
        self.high_cost_active: bool = False
        self.lethal_foot_active: bool = False
        self.close_ped_active: bool = False
        self.state_ring: deque = deque()
        # bt_recent stores (timestamp, "NodeName: prev->cur") for every
        # BT event the bt_navigator publishes — condition nodes,
        # control nodes, and action nodes alike.  Used by
        # _active_bt_node to walk back to the most recent *driving*
        # node (FollowPath / BackUp / Spin / ...).  maxlen was 20,
        # which is roughly 0.2 s of BT history at the rate the
        # navigate_to_pose_w_replanning_and_recovery.xml ticks
        # (~100 events / s through GoalUpdated, IsPathValid,
        # Sequence, RoundRobin, etc).  Between a BackUp:IDLE->RUNNING
        # event and the BACKUP_START event firing 0.3 s later, the
        # driving-node transition was reliably evicted, leaving
        # _active_bt_node to fall back to a scaffolding node and
        # label every backup as no_driver_GoalUpdated.  100 covers
        # ~1 s of BT history which comfortably outlasts the 0.3 s
        # backup debounce.
        self.bt_recent: deque = deque(maxlen=100)

        # /scan related state
        self.last_scan_process_t: float = 0.0
        self.min_scan_range: Optional[float] = None       # m  (closest non-self return)
        self.min_scan_bearing: Optional[float] = None     # rad in lidar_link frame
        self.invisible_last_log_t: float = 0.0
        self.scan_frame: str = 'lidar_link'

        # BACKUP detection state.  pending_t is the wall time the rover first
        # commanded a reverse velocity in this episode.  active is True after
        # the debounce period when we've actually emitted a [BACKUP_START].
        self.backup_pending_t: Optional[float] = None
        self.backup_pending_xy: Optional[Tuple[float, float]] = None
        self.backup_active: bool = False
        self.backup_active_bt: Optional[str] = None

        # TF buffer for transforming /scan returns into the local_costmap
        # frame.  Without this we can't look up the cell cost at the beam
        # endpoint.  Listener spins in its own thread; just creating it
        # is enough.
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=2.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # QoS profiles. Costmaps publish TRANSIENT_LOCAL on the costmap topic.
        costmap_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        # /rover/pose_array is published BEST_EFFORT by pose_topic /
        # ign_ros2_Nav2_topics.py (see rover_metrics/nav2_lidar_metrics_collector.py
        # lines 326–334).  A default RELIABLE subscription gives:
        #   "New publisher discovered on topic '/rover/pose_array', offering
        #    incompatible QoS.  Last incompatible policy: RELIABILITY"
        # and no messages are ever delivered.
        pose_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )

        # Subscriptions
        self.create_subscription(PoseArray, '/rover/pose_array', self._on_pose, pose_qos)
        self.create_subscription(Twist, '/cmd_vel', self._on_cmd_vel, 10)
        self.create_subscription(OccupancyGrid, '/global_costmap/costmap',
                                  self._on_global_costmap, costmap_qos)
        self.create_subscription(OccupancyGrid, '/local_costmap/costmap',
                                  self._on_local_costmap, costmap_qos)
        self.create_subscription(PoseStamped, '/goal_pose', self._on_goal_pose, 10)
        self.create_subscription(GoalStatusArray,
                                  '/navigate_to_pose/_action/status',
                                  self._on_goal_status, 10)
        self.create_subscription(BehaviorTreeLog, '/behavior_tree_log',
                                  self._on_bt_log, 10)
        self.create_subscription(Log, '/rosout', self._on_rosout, 100)
        # /scan typically publishes BEST_EFFORT (sensor_data QoS).  Subscribing
        # with the matching profile avoids a "no messages delivered" trap.
        self.create_subscription(LaserScan, '/scan', self._on_scan,
                                  qos_profile_sensor_data)
        if HAVE_PEOPLE:
            self.create_subscription(People, '/people', self._on_people, 10)

        self.create_timer(1.0, self._on_periodic)
        self.log('INIT', f'started; people topic available: {HAVE_PEOPLE}')

    # ---------- output ----------
    def now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def log(self, tag: str, msg: str) -> None:
        t = self.now()
        wall = time.time()
        print(f't={t:.3f} wall={wall:.3f} [{tag}] {msg}', flush=True)

    # ---------- subscription callbacks ----------
    def _on_pose(self, msg: PoseArray) -> None:
        if not msg.poses:
            return
        p = msg.poses[0].position
        q = msg.poses[0].orientation
        x, y = float(p.x), float(p.y)
        t = self.now()
        if self.robot_xy is not None and t > self.last_pose_time:
            dt = t - self.last_pose_time
            dx = x - self.robot_xy[0]
            dy = y - self.robot_xy[1]
            self.robot_speed = math.hypot(dx, dy) / max(dt, 1e-6)
        self.robot_xy = (x, y)
        # Yaw in world frame.  Needed to project "0.3 m forward of the
        # robot" into the costmap frames for FWD_COLLISION RCA sampling.
        # If the pose publisher leaves orientation unset (all-zero
        # quaternion) the conversion yields yaw = 0; we treat that as
        # "unknown" downstream.
        if abs(q.w) + abs(q.x) + abs(q.y) + abs(q.z) > 0.0:
            self.robot_yaw = self._quat_to_yaw(q)
        self.last_pose_time = t

    def _on_cmd_vel(self, msg: Twist) -> None:
        self.cmd_vx = float(msg.linear.x)
        self.cmd_wz = float(msg.angular.z)
        self._update_backup_state()

    def _update_backup_state(self) -> None:
        """State machine: cmd_vx -> {BACKUP_START, BACKUP_END} events.

        Three states, encoded by (backup_pending_t, backup_active):
          (None,  False) — not reversing
          (t>0,   False) — reversing but still inside the debounce window
          (t>0,   True)  — confirmed backup; we've logged BACKUP_START

        Transitions on every cmd_vel sample.  Hysteresis comes from the gap
        between BACKUP_ENTER_VX (-0.05) and BACKUP_EXIT_VX (-0.01) which
        prevents oscillation near zero from generating a chain of START/END
        pairs.  Debounce filters out very short reverse blips (<0.3 s).
        """
        t = self.now()

        if self.cmd_vx < BACKUP_ENTER_VX:
            if self.backup_pending_t is None:
                self.backup_pending_t = t
                self.backup_pending_xy = self.robot_xy
            elif (not self.backup_active
                  and (t - self.backup_pending_t) > BACKUP_DEBOUNCE_S):
                self.backup_active = True
                self.backup_active_bt = self._active_bt_node()
                rx, ry = self.robot_xy if self.robot_xy else (None, None)
                self.log('BACKUP_START',
                         f'robot=({rx},{ry}) cmd_vx={self.cmd_vx:+.2f} '
                         f'active_bt={self.backup_active_bt}')
        elif self.cmd_vx > BACKUP_EXIT_VX:
            if self.backup_pending_t is not None:
                if self.backup_active:
                    duration = t - self.backup_pending_t
                    if (self.robot_xy is not None
                            and self.backup_pending_xy is not None):
                        dx = self.robot_xy[0] - self.backup_pending_xy[0]
                        dy = self.robot_xy[1] - self.backup_pending_xy[1]
                        reverse_dist = math.hypot(dx, dy)
                        rd_str = f'{reverse_dist:.2f}'
                    else:
                        rd_str = 'None'
                    rx, ry = self.robot_xy if self.robot_xy else (None, None)
                    self.log('BACKUP_END',
                             f'robot=({rx},{ry}) duration={duration:.1f}s '
                             f'reverse_dist={rd_str}m '
                             f'active_bt={self.backup_active_bt}')
                self.backup_pending_t = None
                self.backup_pending_xy = None
                self.backup_active = False
                self.backup_active_bt = None

    def _active_bt_node(self) -> str:
        """Most recent BT node that actually drives cmd_vel.

        The naive 'last event in bt_recent' approach is unreliable
        because the bt_recent deque captures ALL BT events including
        scaffolding nodes (RateController, RoundRobin, PipelineSequence,
        RecoveryFallback, NavigateRecovery, ComputePathToPose) that
        merely sequence other nodes and never publish to /cmd_vel.
        analysis5/6 had many BACKUP_START lines labelled
        ``active_bt=RateController`` even though RateController only
        gates *replanning* frequency — the velocity was coming from
        FollowPath (DWB) or BackUp (recovery).

        We instead walk the deque backwards and return the first
        entry whose node_name contains a 'driving' substring.  If
        nothing matches (e.g. very early in startup), we fall back
        to the last event's name with an explicit prefix so the
        ambiguity is visible in the log.
        """
        driving = ('FollowPath', 'BackUp', 'Spin',
                   'DriveOnHeading', 'AssistedTeleop', 'Wait')
        for _, line in reversed(self.bt_recent):
            node = line.partition(':')[0].strip()
            if any(d in node for d in driving):
                return node
        if not self.bt_recent:
            return 'unknown'
        _, line = self.bt_recent[-1]
        fallback = line.partition(':')[0].strip() or 'unknown'
        # Keep the fallback to a single ``[A-Za-z_]+`` token so the
        # existing analyze_diag.sh BACKUP-analysis regex still groups
        # it correctly.  The "no_driver_" prefix preserves the
        # diagnostic distinction: the last BT event was a scaffolding
        # node, not a driving one.
        return f'no_driver_{fallback}'

    def _on_global_costmap(self, msg: OccupancyGrid) -> None:
        self.global_costmap = msg

    def _on_local_costmap(self, msg: OccupancyGrid) -> None:
        self.local_costmap = msg

    def _on_goal_pose(self, msg: PoseStamped) -> None:
        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        self.current_goal = (x, y)
        self.goal_start_time = self.now()
        gc = self._cost_at(x, y, self.global_costmap)
        self.log('GOAL_START', f'goal=({x:.2f},{y:.2f}) goal_cell_cost={gc}')

    def _on_goal_status(self, msg: GoalStatusArray) -> None:
        for s in msg.status_list:
            uuid = bytes(s.goal_info.goal_id.uuid).hex()[:12]
            # Stamp the per-UUID start time on first sighting so the
            # duration of every goal is measured from when bt_navigator
            # first reported a status for it (ACCEPTED / EXECUTING
            # arrive ~10 ms after the /goal_pose publication).  This
            # decouples each goal's duration from any *later* goal's
            # send time, fixing the "duration=0.0s" misnomer that
            # dominated GOAL_FAIL logs in runs 4–6.
            if uuid not in self.goal_start_time_per_uuid:
                self.goal_start_time_per_uuid[uuid] = self.now()

            prev = self.last_goal_status_per_uuid.get(uuid)
            if prev == s.status:
                continue
            self.last_goal_status_per_uuid[uuid] = s.status
            name = GOAL_STATUS_NAME.get(s.status, str(s.status))
            start_t = self.goal_start_time_per_uuid.get(uuid, self.goal_start_time)
            duration = self.now() - start_t

            if s.status == GoalStatus.STATUS_SUCCEEDED:
                self.log('GOAL_OK', f'goal_id={uuid} duration={duration:.1f}s')
            elif s.status in (GoalStatus.STATUS_ABORTED,
                              GoalStatus.STATUS_CANCELED):
                gx, gy = self.current_goal if self.current_goal else (None, None)
                rx, ry = self.robot_xy if self.robot_xy else (None, None)
                self.log('GOAL_FAIL',
                         f'goal_id={uuid} status={name} duration={duration:.1f}s '
                         f'goal=({gx},{gy}) robot=({rx},{ry})')
                self._dump_context()
            else:
                self.log('GOAL_STATE', f'goal_id={uuid} -> {name}')

    def _on_bt_log(self, msg: BehaviorTreeLog) -> None:
        for e in msg.event_log:
            line = f'{e.node_name}: {e.previous_status}->{e.current_status}'
            self.bt_recent.append((self.now(), line))
            # Only emit recovery-relevant BT transitions to keep noise down.
            interesting_nodes = ('FollowPath', 'ComputePathToPose', 'BackUp',
                                 'Spin', 'Wait', 'RecoveryFallback',
                                 'RoundRobin', 'NavigateRecovery')
            if any(n in e.node_name for n in interesting_nodes):
                if e.current_status in ('FAILURE', 'SUCCESS', 'RUNNING', 'IDLE'):
                    self.log('BT', line)

    def _on_rosout(self, msg: Log) -> None:
        # Filter: WARN+ from planner_server / controller_server only.
        # NOTE: In rclpy on Humble the uint8 message constants (Log.WARN etc.)
        # are exposed as `bytes` while msg.level is `int`, so a direct
        # comparison raises TypeError.  Use the integer literal directly:
        #   DEBUG=10, INFO=20, WARN=30, ERROR=40, FATAL=50
        if msg.level < 30:
            return
        if 'planner' not in msg.name and 'controller' not in msg.name \
                and 'bt_navigator' not in msg.name:
            return
        for tag_suffix, pat in ROSOUT_FILTERS.items():
            if pat.search(msg.msg):
                self.log(f'ROSOUT_{tag_suffix.upper()}',
                         f'node={msg.name} msg={msg.msg!r}')
                return

    def _on_people(self, msg) -> None:
        self.people = list(msg.people)

    # ---------- /scan: OBSTACLE_INVISIBLE + min_scan_range cache ----------
    def _on_scan(self, msg: LaserScan) -> None:
        """Process /scan at ~SCAN_PROCESS_PERIOD Hz.

        Two products:
          1.  self.min_scan_range / self.min_scan_bearing — cached for the
              next [STATE] tick so the 1 Hz snapshot includes how close ANY
              return is, independent of the costmap.  This lets us see the
              robot approaching an obstacle even if the costmap is empty.
          2.  [OBSTACLE_INVISIBLE] event — when the closest non-self return
              is within INVISIBLE_RANGE_MAX of the robot AND the local
              costmap cell at that return's xy has cost < INVISIBLE_COST_MAX.
              This is the smoking-gun measurement for "scan sees the post,
              costmap doesn't" — the failure mode that causes the rover to
              drive directly into thin fence posts.
        """
        t = self.now()
        if t - self.last_scan_process_t < SCAN_PROCESS_PERIOD:
            return
        self.last_scan_process_t = t
        # Remember frame_id from the first scan we see in case it differs
        # from the default 'lidar_link'.
        if msg.header.frame_id:
            self.scan_frame = msg.header.frame_id

        # Find closest valid, non-self return.
        rmin = max(float(msg.range_min), SCAN_SELF_RETURN_MIN)
        rmax = float(msg.range_max)
        # Cap a hair below range_max to drop "no return" sentinels some
        # drivers publish at exactly range_max.
        rmax_eff = rmax * 0.99
        best_r = math.inf
        best_idx = -1
        for i, r in enumerate(msg.ranges):
            if not math.isfinite(r):
                continue
            if r < rmin or r > rmax_eff:
                continue
            if r < best_r:
                best_r = r
                best_idx = i

        if best_idx < 0:
            self.min_scan_range = None
            self.min_scan_bearing = None
            return

        bearing = msg.angle_min + best_idx * msg.angle_increment
        self.min_scan_range = best_r
        self.min_scan_bearing = bearing

        # OBSTACLE_INVISIBLE only when the return is close AND we have a
        # local costmap to compare against.
        if best_r > INVISIBLE_RANGE_MAX:
            return
        if self.local_costmap is None:
            return
        if t - self.invisible_last_log_t < INVISIBLE_LOG_THROTTLE:
            return

        # Project beam endpoint from lidar_link into the local costmap frame.
        x_lidar = best_r * math.cos(bearing)
        y_lidar = best_r * math.sin(bearing)
        endpoint = self._lidar_to_costmap_xy(x_lidar, y_lidar,
                                             self.local_costmap.header.frame_id)
        if endpoint is None:
            return
        x_cm, y_cm = endpoint
        cost = self._cost_at(x_cm, y_cm, self.local_costmap)
        if cost is None:
            # Endpoint outside the local costmap — can't say whether the
            # costmap knows about it.  Don't log; not actionable.
            return
        if cost >= INVISIBLE_COST_MAX:
            # Costmap sees it.  This is the working case; nothing to flag.
            return

        self.invisible_last_log_t = t
        rx, ry = self.robot_xy if self.robot_xy else (None, None)
        bearing_deg = math.degrees(bearing)
        self.log('OBSTACLE_INVISIBLE',
                 f'range={best_r:.2f} bearing={bearing_deg:+.1f}deg '
                 f'endpoint=({x_cm:.2f},{y_cm:.2f}) cell_cost={cost} '
                 f'robot=({rx},{ry})')

    def _lidar_to_costmap_xy(self, x: float, y: float,
                              target_frame: str) -> Optional[Tuple[float, float]]:
        """Transform a point (x, y, 0) in self.scan_frame into target_frame
        using the latest available TF.  Returns None if the transform
        isn't available yet (early startup) or the frames are missing.

        Done manually with planar yaw + translation instead of dragging in
        tf2_geometry_msgs to keep the dependency footprint small.  Both the
        lidar and the local costmap are horizontal planes for our purposes,
        so a 2D transform is sufficient.
        """
        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame, self.scan_frame, Time(),
                timeout=Duration(seconds=0.05))
        except (tf2_ros.LookupException,
                tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException,
                tf2_ros.TransformException):
            return None
        yaw = self._quat_to_yaw(tf.transform.rotation)
        c, s = math.cos(yaw), math.sin(yaw)
        tx = tf.transform.translation.x
        ty = tf.transform.translation.y
        return (tx + c * x - s * y, ty + s * x + c * y)

    @staticmethod
    def _quat_to_yaw(q) -> float:
        """Yaw extracted from a geometry_msgs/Quaternion (planar
        assumption: roll/pitch ≈ 0).  Used both for the TF rotation in
        _lidar_to_costmap_xy and for the robot's body orientation
        coming in on /rover/pose_array."""
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    # ---------- FWD_COLLISION RCA dump ----------
    def _log_fwd_collision_context(self, duration_s: float) -> None:
        """Emit a [FWD_COLLISION] line with everything needed to do
        root-cause analysis on a forward-stuck event.  Called from
        _on_periodic only when STUCK_START fires AND cmd_vx > 0.

        Interpretation key (use this when reading the log):
          1.  min_scan = X m at bearing close to 0° ⇒ lidar sees a real
              obstacle ahead.  min_scan > 0.5 m or bearing pointing
              sideways ⇒ no actual obstacle in front; the stuck is from
              terrain / wheel slip / sim weirdness, not a collision.
          2.  local_fwd_0.3m / global_fwd_0.3m HIGH (>200) ⇒ both costmap
              layers know the obstacle is right in front.  The planner
              and controller chose to drive in anyway — that's a
              critic-weight / inflation-gradient issue (BaseObstacle,
              ObstacleFootprint, cost_travel_multiplier).
          3.  min_scan close AND local_fwd_*M all LOW (<50) ⇒ the lidar
              sees it but the local costmap doesn't.  Perception failure:
              observation_persistence too short, min_obstacle_height
              filtering it out, or the obstacle is below the lidar plane.
          4.  local_fwd_0.3m LOW but global_fwd_0.3m HIGH (or vice
              versa) ⇒ the two layers disagree.  Usually means a
              transient observation that one layer has cleared and the
              other hasn't.
        """
        if self.robot_xy is None:
            return  # without pose we can't sample forward of the robot
        rx, ry = self.robot_xy
        # If pose has no orientation, we can't project forward in world
        # frame.  Bail with a marker so we still get a line and can
        # diagnose the pose-publisher problem.
        if self.robot_yaw is None:
            self.log('FWD_COLLISION',
                     f'duration={duration_s:.1f}s '
                     f'cmd_vx={self.cmd_vx:+.2f} '
                     f'actual_v={self.robot_speed:.2f} '
                     f'robot=({rx:.2f},{ry:.2f},UNKNOWN_YAW) '
                     f'NOTE=pose_array has zero orientation; '
                     f'cannot project forward')
            return

        cy, sy = math.cos(self.robot_yaw), math.sin(self.robot_yaw)
        ms = (f'{self.min_scan_range:.2f}'
              if self.min_scan_range is not None else 'None')
        mb = (f'{math.degrees(self.min_scan_bearing):+.0f}'
              if self.min_scan_bearing is not None else 'None')

        # Sample local + global costmap at robot + d * heading for each
        # d in FWD_COLLISION_FWD_SAMPLES_M.  Each entry becomes its own
        # named field so grep/awk can pull them out trivially.
        sample_fields = []
        for d in FWD_COLLISION_FWD_SAMPLES_M:
            fx = rx + d * cy
            fy = ry + d * sy
            lc = self._cost_at(fx, fy, self.local_costmap)
            gc = self._cost_at(fx, fy, self.global_costmap)
            sample_fields.append(f'local_fwd_{d:.1f}m={lc}')
            sample_fields.append(f'global_fwd_{d:.1f}m={gc}')

        active_bt = self._active_bt_node()
        gx, gy = self.current_goal if self.current_goal else (None, None)
        if gx is not None and gy is not None:
            goal_dist_str = f'{math.hypot(gx - rx, gy - ry):.2f}m'
        else:
            goal_dist_str = 'None'

        head = (f'duration={duration_s:.1f}s '
                f'cmd_vx={self.cmd_vx:+.2f} '
                f'actual_v={self.robot_speed:.2f} '
                f'robot=({rx:.2f},{ry:.2f},'
                f'{math.degrees(self.robot_yaw):+.0f}deg) '
                f'min_scan={ms}m@{mb}deg')
        tail = (f'active_bt={active_bt} goal_dist={goal_dist_str}')
        self.log('FWD_COLLISION',
                 f'{head} {" ".join(sample_fields)} {tail}')

    # ---------- cost lookup ----------
    @staticmethod
    def _cost_at(x: float, y: float,
                 cm: Optional[OccupancyGrid]) -> Optional[int]:
        if cm is None:
            return None
        info = cm.info
        mx = int((x - info.origin.position.x) / info.resolution)
        my = int((y - info.origin.position.y) / info.resolution)
        if mx < 0 or my < 0 or mx >= info.width or my >= info.height:
            return None
        idx = my * info.width + mx
        if idx < 0 or idx >= len(cm.data):
            return None
        raw = cm.data[idx]
        # nav2 publishes int8 with -1 meaning NO_INFORMATION and 0..100 mapped
        # from 0..255 by the costmap_2d publisher.  We want the original 0..255
        # interpretation: -1 -> 255 (no info), otherwise rescale.
        if raw < 0:
            return 255
        return int(round(raw * 255 / 100))

    def _nearest_ped_dist(self) -> Optional[float]:
        if not self.people or self.robot_xy is None:
            return None
        rx, ry = self.robot_xy
        best = None
        for p in self.people:
            d = math.hypot(p.position.x - rx, p.position.y - ry)
            if best is None or d < best:
                best = d
        return best

    # ---------- periodic ----------
    def _on_periodic(self) -> None:
        if self.robot_xy is None:
            return
        t = self.now()
        x, y = self.robot_xy
        gc = self._cost_at(x, y, self.global_costmap)
        lc = self._cost_at(x, y, self.local_costmap)
        pd = self._nearest_ped_dist()

        ms = (f'{self.min_scan_range:.2f}'
              if self.min_scan_range is not None else 'None')
        mb = (f'{math.degrees(self.min_scan_bearing):+.0f}'
              if self.min_scan_bearing is not None else 'None')
        state_line = (f'pos=({x:.2f},{y:.2f}) cmd_vx={self.cmd_vx:+.2f} '
                      f'cmd_wz={self.cmd_wz:+.2f} actual_v={self.robot_speed:.2f} '
                      f'gcost={gc} lcost={lc} ped_dist={pd} '
                      f'min_scan={ms}m@{mb}deg')
        self.log('STATE', state_line)

        # Maintain ring buffer for failure context dumps.
        self.state_ring.append((t, state_line))
        while self.state_ring and (t - self.state_ring[0][0]) > STATE_RING_SECONDS:
            self.state_ring.popleft()

        # ----- stuck detection -----
        # State machine encoded by `stuck_since`:
        #   None       — robot is moving (or not commanded to move): clear
        #   t > 0      — pending: commanded-but-stationary started at t,
        #                STUCK_START not yet fired
        #   -1         — latched: STUCK_START already fired for this
        #                episode, waiting for the stuck condition to
        #                clear before declaring STUCK_END
        # The latched case must guard the re-fire elif explicitly:
        # the naive `(t - stuck_since) > THRESHOLD` test evaluates
        # to (t - (-1)) = t + 1 ≫ 2.0 when latched, which re-fires
        # STUCK_START every periodic tick and prints a garbage
        # duration like 'duration=1779744861.9s' (the unix
        # timestamp).  In analysis7.txt this inflated STUCK_START
        # from ~19 (true episodes) to 45.
        commanded_to_move = abs(self.cmd_vx) > STUCK_VEL_THRESHOLD
        actually_stationary = self.robot_speed < STUCK_VEL_THRESHOLD
        if commanded_to_move and actually_stationary:
            if self.stuck_since is None:
                self.stuck_since = t
            elif (self.stuck_since != -1
                  and (t - self.stuck_since) > STUCK_DURATION_THRESHOLD):
                self.log('STUCK_START',
                         f'duration={t - self.stuck_since:.1f}s '
                         f'cmd_vx={self.cmd_vx:+.2f} actual_v={self.robot_speed:.2f} '
                         f'gcost={gc} lcost={lc}')
                # Forward-stuck case → dump RCA payload immediately on a
                # separate [FWD_COLLISION] line.  Keeping the two events
                # distinct (rather than overloading STUCK_START) lets the
                # analyze script count forward-collisions independently
                # and preserves backwards-compatible STUCK_START parsing.
                if self.cmd_vx > FWD_COLLISION_VX_THRESHOLD:
                    self._log_fwd_collision_context(t - self.stuck_since)
                self.stuck_since = -1  # latch: don't repeat until cleared
        else:
            if self.stuck_since == -1:
                self.log('STUCK_END', f'cmd_vx={self.cmd_vx:+.2f} '
                                       f'actual_v={self.robot_speed:.2f}')
            # else: was None or a still-pending positive timestamp;
            # either way the next non-stuck tick clears it silently.
            self.stuck_since = None

        # ----- high cost under robot -----
        if gc is not None and gc >= LETHAL_COST_THRESHOLD:
            if not self.lethal_foot_active:
                self.log('LETHAL_FOOT',
                         f'robot=({x:.2f},{y:.2f}) gcost={gc} lcost={lc}')
                self.lethal_foot_active = True
        else:
            self.lethal_foot_active = False

        if gc is not None and gc >= HIGH_COST_THRESHOLD \
                and gc < LETHAL_COST_THRESHOLD:
            if not self.high_cost_active:
                self.log('HIGH_COST',
                         f'robot=({x:.2f},{y:.2f}) gcost={gc} lcost={lc}')
                self.high_cost_active = True
        elif gc is None or gc < HIGH_COST_THRESHOLD:
            self.high_cost_active = False

        # ----- close pedestrian -----
        if pd is not None and pd < CLOSE_PED_THRESHOLD:
            if not self.close_ped_active:
                self.log('CLOSE_PED', f'ped_dist={pd:.2f} robot=({x:.2f},{y:.2f})')
                self.close_ped_active = True
        else:
            self.close_ped_active = False

    def _dump_context(self) -> None:
        for ts, line in self.state_ring:
            print(f't={ts:.3f} [CONTEXT_STATE] {line}', flush=True)
        for ts, ev in self.bt_recent:
            print(f't={ts:.3f} [CONTEXT_BT] {ev}', flush=True)
        print('[CONTEXT_END]', flush=True)


def main() -> None:
    rclpy.init()
    node = Nav2Diagnostic()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
