import gymnasium as gym
import numpy as np
import rclpy
import subprocess
import time
import math
from geometry_msgs.msg import Twist, Pose, PoseArray, Point, Quaternion
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from transforms3d.euler import quat2euler
from gymnasium import spaces
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState




class RoverEnv(gym.Env):
    """Custom Environment that follows gymnasium interface"""
    metadata = {'render_modes': ['human']}
    def __init__(self, size=(64, 64), length=6000, scan_topic='/scan', imu_topic='/imu/data',
                 cmd_vel_topic='/cmd_vel', camera_topic='/camera/image_raw',
                 connection_check_timeout=30, lidar_points=640, max_lidar_range=12.0):

        super().__init__()
        
        # Initialize ROS2 node and publishers/subscribers
        rclpy.init()
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
        
        self.imu_subscriber = self.node.create_subscription(
            Imu,
            imu_topic,
            self.imu_callback,
            10)
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.pose_array_subscriber = self.node.create_subscription(
            PoseArray,
            '/rover/pose_array',
            self.pose_array_callback,
            qos_profile
        )

        # Create service client for resetting robot pose
        #self.reset_client = self.node.create_client(SetEntityState, '/gazebo/set_entity_state')
        # Wait for service to become available
        #while not self.reset_client.wait_for_service(timeout_sec=1.0):
        #    print('Waiting for /gazebo/set_entity_state service...')

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
        self.cooldown_steps = 20
        self.steps_since_correction = self.cooldown_steps
        self.corrective_linear = 0.0
        self.corrective_angular = 0.0
        # Flip detection parameters
        self.flip_threshold = math.pi / 3  # 60 degrees in radians
        self.is_flipped = False
        self.initial_position = None
        self.initial_orientation = None

        #ground truth pose
        self.current_pose = Pose()
        self.current_pose.position.x = 0.0
        self.current_pose.position.y = 0.0
        self.current_pose.position.z = 0.0
        self.current_pose.orientation.x = 0.0
        self.current_pose.orientation.y = 0.0
        self.current_pose.orientation.z = 0.0
        self.current_pose.orientation.w = 1.0  # w=1 represents no rotation
        
        #point navigation
        #self.target_positions = [(-9,9),(0,0),(-9,-9),(-1,-2),(-9,-9)]
        #self.target_positions = [(-2,6), (-4,3), (-2,-3)]
        self.target_positions = [(-3,8),(-5,2),(0.0,1)]
        self.current_target_idx = 0
        self.success_distance = 0.5  # Distance threshold to consider target reached
        self.previous_distance = None  # For progress reward
        
        # Define action space
        # [linear_velocity, angular_velocity]
        self.action_space = spaces.Box(
            #low=np.array([-0.3, -1.0]),  # [min_linear_vel, min_angular_vel] old slow values so no climb
            #high=np.array([0.3, 1.0]),   # [max_linear_vel, max_angular_vel] old slow values so no climb
            low=np.array([-0.3, -3.0]),  # [min_linear_vel, min_angular_vel]
            high=np.array([0.3, 3.0]),   # [max_linear_vel, max_angular_vel]
            dtype=np.float32
        )

        self.observation_space = spaces.Dict({
            'lidar': spaces.Box(
                low=0,
                high=max_lidar_range,
                shape=(lidar_points,),
                dtype=np.float32
            ),
            'pose': spaces.Box(  # Replace 'odom' with 'pose'
                low=np.array([-20.0, -20.0, 0.0]),  # Based on your 20x15m environment
                high=np.array([20.0, 20.0, 0.0]),
                dtype=np.float32
            ),
            'imu': spaces.Box(
                low=np.array([-np.pi, -np.pi, -np.pi]),  # roll, pitch, yaw
                high=np.array([np.pi, np.pi, np.pi]),
                dtype=np.float32
            ),
            'target': spaces.Box(  # Your updated target bounds look good
                low=np.array([0, -np.pi]),
                high=np.array([100, np.pi]),
                shape=(2,),
                dtype=np.float32
            )
        })
        


        # Check robot connection
        self._robot_connected = self._check_robot_connection(timeout=connection_check_timeout)
        if not self._robot_connected:
            self.node.get_logger().warn("No actual robot detected. Running in simulation mode.")


    def task_reward(self):
        collision_threshold = 0.46
        
        """Simplified reward function for point navigation task"""
        if self.current_pose is None:
            return 0.0
    
        # Get current position and target
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
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
        
        # Success reward: if reached target
        if current_distance < self.success_distance:
            reward = 50.0 #100.0
            self.current_target_idx = (self.current_target_idx + 1) % len(self.target_positions)
            print('######################################################################')
            self.node.get_logger().info(f'Target reached! Moving to target {self.current_target_idx}')
            print('######################################################################')
            self.previous_distance = None
            return reward

        # 2. Collision penalty
        min_distance = np.min(self.lidar_data[np.isfinite(self.lidar_data)])
        if min_distance < collision_threshold:
            return -1.0 #-10.0


        # Calculate reward
        reward = 0.0
    
        # Calculate heading
        target_heading = math.atan2(target_y - current_y, target_x - current_x)
        heading_diff = abs(math.atan2(math.sin(target_heading - self.current_yaw), 
                                 math.cos(target_heading - self.current_yaw)))
    
        # Progress reward only when facing within 90 degrees of target
        distance_delta = self.previous_distance - current_distance
        if heading_diff < math.pi/2:  # 90 degrees
            reward += distance_delta * 80.0
    
        # Update previous distance
        self.previous_distance = current_distance
    
        # Step penalty
        reward -= 0.01

        # Debug info (simplified)
        if self.total_steps % 1000 == 0:
            self.node.get_logger().info(
                f'Status: Target: {self.current_target_idx}, '
                f'Distance: {current_distance:.2f}m, '
                f'Heading diff: {math.degrees(heading_diff):.1f}°, '
                f'Heading: {math.degrees(heading_diff):.1f}°, '
                f'Distance delat: {distance_delta:.3f}m, '
                f'Progress reward: {(distance_delta *2):.3f}m, '
                f'Reward: {reward:.3f}'
            )
            
        return reward
            

    def get_target_info(self):
        """Calculate distance and azimuth to current target"""
        if self.current_pose is None:
            return np.array([0.0, 0.0], dtype=np.float32)
        
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        target_x, target_y = self.target_positions[self.current_target_idx]
    
        # Calculate distance
        distance = math.sqrt(
            (current_x - target_x)**2 + 
            (current_y - target_y)**2
        )
    
        # Calculate azimuth (relative angle to target)
        target_heading = math.atan2(target_y - current_y, target_x - current_x)
        relative_angle = math.atan2(math.sin(target_heading - self.current_yaw), 
                               math.cos(target_heading - self.current_yaw))
    
        return np.array([distance, relative_angle], dtype=np.float32)


    def is_robot_flipped(self):
        """Detect if robot has flipped forward/backward past 85 degrees"""
        FLIP_THRESHOLD = 1.48  # ~85 degrees in radians
        
        if self.current_pitch < -FLIP_THRESHOLD:
            return 'forward'
        elif self.current_pitch > FLIP_THRESHOLD:  
            return 'backward'
        return False
    

    def step(self, action):
        """Execute one time step within the environment"""
        self.total_steps += 1
        
        # Wait for new sensor data
        while not (self._received_scan):
            rclpy.spin_once(self.node, timeout_sec=0.01)

        flip_status = self.is_robot_flipped()
        if flip_status:
            print('Robot flipped', flip_status, ', exiting')
            exit()
        

        self._received_scan = False
        self.last_angular_velocity = float(action[1]) 
        # Check for climbing and adjust movement with cooldown
        climbing_status, climbing_severity = self.is_climbing_wall()
        twist = Twist()
    
        if climbing_status or self.steps_since_correction < self.cooldown_steps:
            if climbing_status:
                # Save the corrective action based on climbing type
                if climbing_status == 'forward':
                    self.corrective_linear = -1.0
                    self.corrective_angular = 0.0
                elif climbing_status == 'reverse':
                    self.corrective_linear = 1.0
                    self.corrective_angular = 0.0
                elif climbing_status in ['right_tilt', 'left_tilt']:
                    self.corrective_linear = -1.0
                    self.corrective_angular = 0.0
                self.steps_since_correction = 0
            
            # Use stored corrective values
            twist.linear.x = self.corrective_linear
            twist.angular.z = self.corrective_angular
            self.steps_since_correction += 1
        else:
            # Normal operation
            twist.linear.x = float(action[0])
            twist.angular.z = float(action[1])

        self.publisher.publish(twist)
        self.last_linear_velocity = twist.linear.x
        
        # Calculate reward
        reward = self.task_reward()

        # Check if episode is done
        self._step += 1
        done = (self._step >= self._length)
        # Get observation
    
        observation = {
            'lidar': self.lidar_data,
            'pose': self.rover_position,  # Changed from 'odom' to 'pose'
            'imu': np.array([self.current_pitch, self.current_roll, self.current_yaw],
                            dtype=np.float32),
            'target': self.get_target_info()
        }

        
        # Info dict for additional information
        info = {
            'steps': self._step,
            'total_steps': self.total_steps
        }
        
        if climbing_status:
            reward -= 10.0 * climbing_severity  # Scales penalty with tilt severity

        #if self.total_steps % 10_000 == 0:
        #    print(observation)
        # Debug print statement
        if self.total_steps % 1000 == 0:
            temp_obs_target = observation["target"]
            print(
                #f"climbing_status: {climbing_status},  climbing_severity: {climbing_severity},  "
                #f"Pitch: {round(self.current_pitch,3)},  Roll: {round(self.current_roll,3)},  "
                #f"min lidar: {round(np.nanmin(self.lidar_data),3)}   Yaw: {round(self.current_yaw,3)},  "
                f"current target: {self.target_positions[self.current_target_idx]},  "
                f"distance and angle to target: {temp_obs_target},  "
                #f"previous distance: {self.previous_distance},  "
                f"Reward: {reward},  "
            )
        #print(observation)
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
    
        # Reset robot pose using ign service
        try:
            reset_cmd = [
                'ign', 'service', '-s', '/world/maze/set_pose',
                '--reqtype', 'ignition.msgs.Pose',
                '--reptype', 'ignition.msgs.Boolean',
                '--timeout', '2000',
                '--req', 'name: "rover_zero4wd", position: {x: 0, y: 0, z: 0.1}, orientation: {x: 0, y: 0, z: 0, w: 1}'
            ]
            result = subprocess.run(reset_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Failed to reset robot pose: {result.stderr}")
        except Exception as e:
            print(f"Error executing reset command: {str(e)}")

        # Reset internal state
        self._step = 0
        self.last_linear_velocity = 0.0
        self.steps_since_correction = self.cooldown_steps
        self.is_flipped = False
        # Reset PointNav-specific variables
        self.current_target_idx = 0
        self.previous_distance = None
        
        # Add a small delay to ensure the robot has time to reset
        for _ in range(5):  # Increased from 3 to 5 to allow more time for pose reset
            rclpy.spin_once(self.node, timeout_sec=0.1)
    
        observation = {
            'lidar': self.lidar_data,
            'pose': self.rover_position,
            'imu': np.array([self.current_pitch, self.current_roll, self.current_yaw],
                            dtype=np.float32),
            'target': self.get_target_info()
        }

        return observation, {}
    
    def resetOld(self, seed=None, options=None):
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
            'pose': self.rover_position,  # Changed from 'odom' to 'pose'
            'imu': np.array([self.current_pitch, self.current_roll, self.current_yaw],
                            dtype=np.float32),
            'target': self.get_target_info()
        }
    
        return observation, {}

    def is_climbing_wall(self):
        """Detect if robot is climbing based solely on IMU data"""
        pitch_threshold = 0.2
        roll_threshold = 0.2
        
        # Check IMU angles
        is_pitch_steep = abs(self.current_pitch) > pitch_threshold
        is_roll_steep = abs(self.current_roll) > roll_threshold
        
        climbing_status = False
        severity = 0.0
        
        # Only check IMU angles and velocity
        if is_pitch_steep or is_roll_steep:
            #if abs(self.last_linear_velocity) > 0.05:  # Confirm we're moving
            if self.current_pitch > pitch_threshold:
                climbing_status = 'reverse'
                severity = self.current_pitch * abs(self.last_linear_velocity)
            elif self.current_pitch < -pitch_threshold:
                climbing_status = 'forward'
                severity = abs(self.current_pitch) * abs(self.last_linear_velocity)
            elif self.current_roll > roll_threshold:
                climbing_status = 'right_tilt'
                severity = self.current_roll * abs(self.last_linear_velocity)
            elif self.current_roll < -roll_threshold:
                climbing_status = 'left_tilt'
                severity = abs(self.current_roll) * abs(self.last_linear_velocity)
        
        if climbing_status:
            self.node.get_logger().info(
                f'Climbing detected: {climbing_status}, '
                f'Severity: {severity:.2f}, '
                f'Pitch: {self.current_pitch:.2f}, '
                f'Roll: {self.current_roll:.2f}, '
                f'Velocity: {self.last_linear_velocity:.2f}'
            )
            
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
            self.current_pose = msg.poses[0]  # Take the first pose
            # UPDATE - Store position as numpy array
            self.rover_position = np.array([
                self.current_pose.position.x,
                self.current_pose.position.y,
                self.current_pose.position.z
            ], dtype=np.float32)
        

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


    def _check_robot_connection(self, timeout):
        start_time = time.time()
        while not self._received_scan:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if time.time() - start_time > timeout:
                return False
            if self._received_scan:
                return True
        return False


