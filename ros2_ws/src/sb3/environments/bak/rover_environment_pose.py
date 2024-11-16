import gymnasium as gym
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from transforms3d.euler import quat2euler
from gymnasium import spaces
import time
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_srvs.srv import Empty
import math
from geometry_msgs.msg import Pose, Twist
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState


class RoverEnv(gym.Env):
    """Custom Environment that follows gymnasium interface"""
    metadata = {'render_modes': ['human']}

    def __init__(self, size=(64, 64), length=6000, scan_topic='/scan', imu_topic='/imu/data',
                 cmd_vel_topic='/cmd_vel', odom_topic='/odometry/wheels',
                 camera_topic='/camera/image_raw',
                 connection_check_timeout=30, lidar_points=640, max_lidar_range=12.0):
        super().__init__()
        
        # Initialize ROS2 node and publishers/subscribers
        #rclpy.init()
        self.bridge = CvBridge()
        self.node = rclpy.create_node('turtlebot_controller')

        # Create reset service client
        #self.reset_simulation_client = self.node.create_client(Empty, '/world/maze/reset')

        # Initialize publishers and subscribers
        self.publisher = self.node.create_publisher(
            Twist,
            cmd_vel_topic,
            10)
        
        self.lidar_subscriber = self.node.create_subscription(
            LaserScan,
            scan_topic,
            self.lidar_callback,
            10)
        
        self.odom_subscription = self.node.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            10)
        
        self.imu_subscriber = self.node.create_subscription(
            Imu,
            imu_topic,
            self.imu_callback,
            10)
        
        self.pose_array_subscriber = self.node.create_subscription(
            PoseArray,
            '/rover/pose_array',
            self.pose_array_callback,
            10)

        #self.set_entity_state_client = self.node.create_client(SetEntityState, '/set_entity_state')
        
        # Initialize environment parameters
        self.pose_node = None
        self.lidar_points = lidar_points
        self.max_lidar_range = max_lidar_range
        self.lidar_data = np.zeros(self.lidar_points, dtype=np.float32)
        self._length = length
        self._step = 0
        self._received_scan = False
        self.first = False
        self.desired_distance = 0.5  # wall following
        self.total_steps = 0
        self.last_linear_velocity = 0.0
        self.current_pitch = 0.0
        self.current_roll = 0.0
        self.current_yaw = 0.0
        self.rover_position = (0, 0, 0)
        self.last_angular_velocity = 0.0
        # Cooldown mechanism
        self.cooldown_steps = 100
        self.steps_since_correction = self.cooldown_steps

        # Flip detection parameters
        self.flip_threshold = math.pi / 3  # 60 degrees in radians
        self.is_flipped = False
        self.initial_position = None
        self.initial_orientation = None

        #point navigation
        #self.target_positions = [(-9,8),(-3,9),(-2,6),(-9,-5),(-2,-8),(-3,-1)]
        self.target_positions = [(-2,6), (-4,3), (-2,-3)]
        self.current_target_idx = 0
        self.success_distance = 0.5  # Distance threshold to consider target reached
        self.previous_distance = None  # For progress reward
        
        # Define action space
        # [linear_velocity, angular_velocity]
        self.action_space = spaces.Box(
            low=np.array([-0.3, -1.0]),  # [min_linear_vel, min_angular_vel]
            high=np.array([0.3, 1.0]),   # [max_linear_vel, max_angular_vel]
            dtype=np.float32
        )

        # Define observation space
        # [lidar_data (640), position (3), orientation (3)]
        self.observation_space = spaces.Dict({
            'lidar': spaces.Box(
                low=0,
                high=max_lidar_range,
                shape=(lidar_points,),
                dtype=np.float32
            ),
            'odom': spaces.Box(
                low=-float('inf'),
                high=float('inf'),
                shape=(3,),
                dtype=np.float32
            ),
            'imu': spaces.Box(
                low=-float('inf'),
                high=float('inf'),
                shape=(3,),
                dtype=np.float32
            )
        })

        # Check robot connection
        self._robot_connected = self._check_robot_connection(timeout=connection_check_timeout)
        if not self._robot_connected:
            self.node.get_logger().warn("No actual robot detected. Running in simulation mode.")

        
    def task_reward(self):
        """PointNav reward function"""
        if self.current_pose is None:
            return 0.0  # No pose data available
            
        reward = 0.0
        
        # Get current position
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        
        # Get current target
        target_x, target_y = self.target_positions[self.current_target_idx]
        
        # Calculate distance to current target
        current_distance = math.sqrt(
            (current_x - target_x)**2 + 
            (current_y - target_y)**2
        )
        
        # Initialize previous_distance if not set
        if self.previous_distance is None:
            self.previous_distance = current_distance
            return 0.0
            
        # Progress reward: positive reward for getting closer, negative for getting further
        distance_delta = self.previous_distance - current_distance
        progress_reward = distance_delta * 1.0  # Scale factor for progress
        reward += progress_reward
        
        # Success reward: if reached target
        if current_distance < self.success_distance:
            reward += 10.0  # Bonus for reaching target
            self.current_target_idx = (self.current_target_idx + 1) % len(self.target_positions)
            self.node.get_logger().info(f'Target reached! Moving to target {self.current_target_idx}')
            
            # Reset previous_distance for new target
            self.previous_distance = None
            return reward
            
        # Update previous distance
        self.previous_distance = current_distance
        
        # Add a small negative reward for each step to encourage efficiency
        reward -= 0.01
        
        # Optional: Add heading reward
        target_heading = math.atan2(target_y - current_y, target_x - current_x)
        current_yaw = self.current_yaw  # Assuming this is updated by your IMU callback
        heading_diff = abs(math.atan2(math.sin(target_heading - current_yaw), 
                                    math.cos(target_heading - current_yaw)))
        heading_reward = (math.pi - heading_diff) / math.pi
        reward += heading_reward * 0.1  # Scale factor for heading
        
        # Debug info (every 100 steps or so)
        if self.total_steps % 100 == 0:
            self.node.get_logger().info(
                f'Target: {self.current_target_idx}, '
                f'Distance: {current_distance:.2f}, '
                f'Reward: {reward:.2f}, '
                f'Progress: {distance_delta:.2f}'
            )
            
        return reward

    def reset(self, seed=None, options=None):
        """Add to your existing reset function"""

        

            
    def step(self, action):
        """Execute one time step within the environment"""
        self.total_steps += 1

        # Wait for new sensor data
        while not (self._received_scan):
            rclpy.spin_once(self.node, timeout_sec=0.01)
            
        self._received_scan = False

        
        # Check if robot has flipped
        if self.check_flip_status():
            print("Robot has flipped! Initiating reset...")
            self.node.get_logger().warn("Robot has flipped! Initiating reset...")
            # Create and publish zero velocity command
            stop_cmd = Twist()
            self.publisher.publish(stop_cmd)
            
            # Reset the simulation
            rclpy.spin_once(self.node)  # Process any pending callbacks

            #success = self.reset_simulation()
            if not success:
                self.node.get_logger().error("Failed to reset simulation after flip")
            # Return observation with large negative reward
            
            observation = {
                'lidar': self.lidar_data,
                'odom': np.array(self.rover_position, dtype=np.float32),
                'imu': np.array([self.current_pitch, self.current_roll, self.current_yaw],
                                dtype=np.float32)
            }
            return observation, -100.0, True, False, {'reset_reason': 'flip'}
        
        self.last_angular_velocity = float(action[1]) 
        # Check for climbing and adjust movement with cooldown
        climbing_status, climbing_severity = self.is_climbing_wall()
        if climbing_status:# and self.steps_since_correction >= self.cooldown_steps:
            # Execute corrective action
            twist = Twist()
            if climbing_status == 'forward':
                twist.linear.x = -0.1
                twist.angular.z = 0.0 #-self.current_roll * 1.0
            elif climbing_status == 'reverse':
                twist.linear.x = 0.1
                twist.angular.z = 0.0 #self.current_roll * 1.0
            self.publisher.publish(twist)
            self.steps_since_correction = 0
        else:
            # Normal operation: publish agent's action
            twist = Twist()
            twist.linear.x = float(action[0])  # linear velocity
            twist.angular.z = float(action[1])  # angular velocity
            self.publisher.publish(twist)
            self.last_linear_velocity = twist.linear.x
            self.steps_since_correction += 1


        # Calculate reward
        reward = self.task_reward()

        # 2. Collision penalty
        min_distance = np.min(lidar_ranges[np.isfinite(lidar_ranges)])
        if min_distance < collision_threshold:
            reward = -2.0

        # Check if episode is done
        self._step += 1
        done = (self._step >= self._length)
        
        # Get observation
        observation = {
            'lidar': self.lidar_data,
            'odom': np.array(self.rover_position, dtype=np.float32),
            'imu': np.array([self.current_pitch, self.current_roll, self.current_yaw],
                            dtype=np.float32)
        }

        # Info dict for additional information
        info = {
            'steps': self._step,
            'total_steps': self.total_steps
        }
        
        if climbing_status:
            reward -= 1.0 * climbing_severity  # Scales penalty with tilt severity

        # Debug print statement
        if self.total_steps % 10_000 == 0:
            print(
                #f"climbing_status: {climbing_status},  climbing_severity: {climbing_severity},  "
                #f"Pitch: {round(self.current_pitch,3)},  Roll: {round(self.current_roll,3)},  "
                #f"min lidar: {round(np.nanmin(self.lidar_data),3)}   Yaw: {round(self.current_yaw,3)},  "
                f"current target: {self.current_target_idx},  "
                f"previous distance: {self.previous_distance},  "
                f"Reward: {reward},  "
            )
            
        return observation, reward, done, False, info  # False is for truncated


    def check_flip_status(self):
        """Check if the robot has flipped based on IMU data"""
        # Use absolute roll and pitch to detect if robot is tilted too much
        if (abs(self.current_roll) > self.flip_threshold
            or abs(self.current_pitch) > self.flip_threshold):
            self.is_flipped = True
            return True
        return False


    def reset(self, seed=None, options=None):
        """Reset the environment to its initial state"""
        super().reset(seed=seed)

        # Reset internal state
        self._step = 0
        self.last_linear_velocity = 0.0
        self.steps_since_correction = self.cooldown_steps
        self.is_flipped = False

        # Reset PointNav-specific variables
        self.current_target_idx = 0
        self.previous_distance = None

        # Ensure we get fresh sensor data after reset
        for _ in range(3):  # Spin a few times to get fresh data
            rclpy.spin_once(self.node, timeout_sec=0.1)
    
        observation = {
            'lidar': self.lidar_data,
            'odom': np.array(self.rover_position, dtype=np.float32),
            'imu': np.array([self.current_pitch, self.current_roll, self.current_yaw],
                            dtype=np.float32)
        }        
    
        return observation, {}


    def is_climbing_wall(self):
        if self.lidar_data is None:
            return False, 0.0
        
        min_distance = np.nanmin(self.lidar_data)
        collision_threshold = 0.2
        pitch_threshold = 0.2
        roll_threshold = 0.2
        
        is_too_close = min_distance < collision_threshold
        is_pitch_steep = abs(self.current_pitch) > pitch_threshold
        is_roll_steep = abs(self.current_roll) > roll_threshold
        
        climbing_status = False
        severity = 0.0
        
        if is_too_close and (is_pitch_steep or is_roll_steep):
            if self.current_pitch > pitch_threshold:
                climbing_status = 'reverse'
                severity = self.current_pitch
            elif self.current_pitch < -pitch_threshold:
                climbing_status = 'forward'
                severity = abs(self.current_pitch)
            else:
                if self.current_roll > roll_threshold:
                    climbing_status = 'right_tilt'
                    severity = self.current_roll
                elif self.current_roll < -roll_threshold:
                    climbing_status = 'left_tilt'
                    severity = abs(self.current_roll)
        if climbing_status:
            print('climbing status:', climbing_status, severity)
        return climbing_status, severity


    def render(self):
        """Render the environment (optional)"""
        pass


    def close(self):
        """Clean up resources"""
        self.node.destroy_node()
        rclpy.shutdown()

    def pose_array_callback(self, msg):
        """Callback for processing pose array messages"""
        if msg.poses:  # Check if we have any poses
            self.current_pose = msg.poses[0]  # Take the first pose (rover_zero4wd)

    def lidar_callback(self, msg):
        if not self.first:
            print("First scan received:")
            print(f"Number of points: {len(msg.ranges)}")
            print(f"Angle min: {msg.angle_min}, Angle max: {msg.angle_max}")
            print(f"Angle increment: {msg.angle_increment}")
            print(f"Range min: {msg.range_min}, Range max: {msg.range_max}")
            print(f"First 20 ranges: {msg.ranges[:20]}")
            print(f"Last 20 ranges: {msg.ranges[-20:]}")
            self.angle_min = msg.angle_min
            self.angle_max = msg.angle_max
            self.angle_increment = msg.angle_increment
            self.first = True

        lidar_data = np.array(msg.ranges, dtype=np.float32)
        lidar_data = np.clip(lidar_data, 0, self.max_lidar_range)
        
        if len(lidar_data) != self.lidar_points:
            lidar_data = np.resize(lidar_data, (self.lidar_points,))

        self.lidar_data = lidar_data
        self._received_scan = True


    def imu_callback(self, msg):
        try:
            quat = np.array([msg.orientation.w, msg.orientation.x, msg.orientation.y,
                             msg.orientation.z])
            norm = np.linalg.norm(quat)
            if norm == 0:
                raise ValueError("Received a zero-length quaternion")
            quat_normalized = quat / norm
            roll, pitch, yaw = quat2euler(quat_normalized, axes='sxyz')
            self.current_pitch = pitch
            self.current_roll = roll
            self.current_yaw = yaw
        except Exception as e:
            self.node.get_logger().error(f"Error processing IMU data: {e}")


    def odom_callback(self, msg):
        self.rover_position = (msg.pose.pose.position.x, msg.pose.pose.position.y,
                               msg.pose.pose.position.z)
        self.last_linear_velocity = msg.twist.twist.linear.x


    def _check_robot_connection(self, timeout):
        start_time = time.time()
        while not self._received_scan:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if time.time() - start_time > timeout:
                return False
            if self._received_scan:
                return True
        return False


