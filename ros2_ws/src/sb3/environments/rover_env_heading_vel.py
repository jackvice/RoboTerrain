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
                 connection_check_timeout=30, lidar_points=16, max_lidar_range=12.0):

        super().__init__()
        
        # Initialize ROS2 node and publishers/subscribers
        rclpy.init()
        self.bridge = CvBridge()
        self.node = rclpy.create_node('turtlebot_controller')

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

        # Initialize environment parameters
        self.pose_node = None
        self.lidar_points = lidar_points
        self.max_lidar_range = max_lidar_range
        self.lidar_data = np.zeros(self.lidar_points, dtype=np.float32)
        self._length = length
        self._step = 0
        self._received_scan = False
        self.first = False
        self.total_steps = 0
        self.last_speed = 0.0
        self.last_heading = 0.0
        self.current_pitch = 0.0
        self.current_roll = 0.0
        self.current_yaw = 0.0
        self.rover_position = (0, 0, 0)
        
        # PID control parameters for heading
        self.Kp = 2.0  # Proportional gain
        self.Ki = 0.0  # Integral gain
        self.Kd = 0.1  # Derivative gain
        self.integral_error = 0.0
        self.last_error = 0.0
        self.max_angular_velocity = 7.0
        
        # Cooldown mechanism
        self.cooldown_steps = 20
        self.steps_since_correction = self.cooldown_steps
        self.corrective_speed = 0.0
        self.corrective_heading = 0.0
        
        # Flip detection parameters
        self.flip_threshold = math.pi / 3  # 60 degrees in radians
        self.is_flipped = False
        self.initial_position = None
        self.initial_orientation = None

        # Ground truth pose
        self.current_pose = Pose()
        self.current_pose.position.x = 0.0
        self.current_pose.position.y = 0.0
        self.current_pose.position.z = 0.0
        self.current_pose.orientation.x = 0.0
        self.current_pose.orientation.y = 0.0
        self.current_pose.orientation.z = 0.0
        self.current_pose.orientation.w = 1.0
        
        # Navigation parameters
        self.target_positions_x = 0
        self.target_positions_y = 0
        self.success_distance = 1.0
        self.previous_distance = None
        self.the_world = 'maze'
        self.world_pose_path = '/world/' + self.the_world + '/set_pose'
        
        # Define action space
        # [speed, desired_heading]
        self.action_space = spaces.Box(
            low=np.array([-0.6, -np.pi]),  # [min_speed, min_heading]
            high=np.array([0.6, np.pi]),   # [max_speed, max_heading]
            dtype=np.float32
        )

        # Observation space remains the same
        self.observation_space = spaces.Dict({
            'lidar': spaces.Box(
                low=0,
                high=max_lidar_range,
                shape=(lidar_points,),
                dtype=np.float32
            ),
            'pose': spaces.Box(
                low=np.array([-20.0, -20.0, 0.0]),
                high=np.array([20.0, 20.0, 0.0]),
                dtype=np.float32
            ),
            'imu': spaces.Box(
                low=np.array([-np.pi, -np.pi, -np.pi]),
                high=np.array([np.pi, np.pi, np.pi]),
                dtype=np.float32
            ),
            'target': spaces.Box(
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

    def heading_controller(self, desired_heading, current_heading):
        """PID controller for heading with time step consideration"""
        dt = 0.05  # 20 Hz -> 0.05 seconds per step
        
        # Normalize angle difference to [-pi, pi]
        error = math.atan2(math.sin(desired_heading - current_heading), 
                           math.cos(desired_heading - current_heading))
    
        # Update integral and derivative terms with dt
        self.integral_error += error * dt
        derivative_error = (error - self.last_error) / dt
        self.last_error = error
        
        # Calculate control output
        control = (self.Kp * error + 
                   self.Ki * self.integral_error + 
                   self.Kd * derivative_error)
        
        # Limit the control output
        return np.clip(control, -self.max_angular_velocity, self.max_angular_velocity)


    def step(self, action):
        """Execute one time step within the environment"""
        self.total_steps += 1
        if self.current_pose.position.z < -20.0:
            print('fell off the world, z is', self.current_pose.position.z, ', resetting.')
            self.reset()
        
        # Wait for new sensor data
        while not self._received_scan:
            rclpy.spin_once(self.node, timeout_sec=0.01)

        flip_status = self.is_robot_flipped()
        if flip_status:
            print('Robot flipped', flip_status, ', resetting')
            self.reset()
        
        self._received_scan = False
        
        # Extract speed and desired heading from action
        speed = float(action[0])
        desired_heading = float(action[1])
        
        # Store last actions
        self.last_speed = speed
        self.last_heading = desired_heading
        
        # Calculate angular velocity using PID controller
        angular_velocity = self.heading_controller(desired_heading, self.current_yaw)
        
        # Check for climbing and adjust movement with cooldown
        climbing_status, climbing_severity = self.is_climbing_wall()
        twist = Twist()
    
        if climbing_status or self.steps_since_correction < self.cooldown_steps:
            if climbing_status:
                # Save the corrective action based on climbing type
                if climbing_status == 'forward':
                    self.corrective_speed = -1.0
                    self.corrective_heading = self.current_yaw
                elif climbing_status == 'reverse':
                    self.corrective_speed = 1.0
                    self.corrective_heading = self.current_yaw
                elif climbing_status in ['right_tilt', 'left_tilt']:
                    self.corrective_speed = -1.0
                    self.corrective_heading = self.current_yaw
                self.steps_since_correction = 0
            
            # Use stored corrective values
            twist.linear.x = self.corrective_speed
            twist.angular.z = self.heading_controller(self.corrective_heading, self.current_yaw)
            self.steps_since_correction += 1
        else:
            # Normal operation
            twist.linear.x = speed
            twist.angular.z = angular_velocity
        
        self.publisher.publish(twist)
        
        # Calculate reward
        reward = self.task_reward()

        # Check if episode is done
        self._step += 1
        done = (self._step >= self._length)
        
        # Get observation
        observation = {
            'lidar': self.lidar_data,
            'pose': self.rover_position,
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
            reward = -100.0
            
        if self.total_steps % 1000 == 0:
            temp_obs_target = observation["target"]
            print(
                f"current target x,y: ({self.target_positions_x}, {self.target_positions_y}), "
                f"distance and angle to target: {temp_obs_target}, "
                f"Final Reward: {reward:.3f}, "
                f"Speed: {speed:.2f}, Heading: {math.degrees(desired_heading):.1f}°"
            )
            
        return observation, reward, done, False, info


    def task_reward(self):
        """Improved reward function for point navigation task with balanced rewards and penalties"""
        # Constants
        collision_threshold = 0.4
        collision_reward = -2.0
        warning_distance = collision_threshold * 2
        distance_scaling_factor = 20
        goal_reward = 100
        #turn_scale = 0.1
        step_penalty = -0.01
        heading_bonus = 0.05
        reverse_penalty = -0.2
        
        if self.current_pose is None:
            return 0.0

        # Get current position and target
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        target_x = self.target_positions_x
        target_y = self.target_positions_y 

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
            reward = goal_reward
            # Set new random target
            self.target_positions_x = np.random.uniform(-3.5, 2.5)
            self.target_positions_y = np.random.uniform(-2.5, 5.0)
            print('######################################################################')
            self.node.get_logger().info(f'Target reached! Moving to target x,y: ({self.target_positions_x}, {self.target_positions_y})')
            print('######################################################################')
            self.previous_distance = None
            return reward

        # Initialize reward
        reward = 0.0

        # Collision checking and graduated penalty
        min_distance = np.min(self.lidar_data[np.isfinite(self.lidar_data)])
        if min_distance < collision_threshold:
            return collision_reward
        elif min_distance < warning_distance:
            # Graduated penalty in warning zone
            warning_factor = (warning_distance - min_distance) / collision_threshold
            reward += collision_reward * 0.5 * warning_factor

        # Movement rewards/penalties
        if self.last_speed > 0.0:  # Forward motion reward
            reward += self.last_speed / 5  # Increased from /10 to encourage more movement
        else:  # Reverse penalty
            reward += reverse_penalty
        
        # Graduated turn penalty based on angular velocity magnitude
        #reward += -turn_scale * abs(self.last_angular_velocity)
        
        # Calculate heading to target
        target_heading = math.atan2(target_y - current_y, target_x - current_x)
        heading_diff = abs(math.atan2(math.sin(target_heading - self.current_yaw), 
                                 math.cos(target_heading - self.current_yaw)))

        # Calculate progress towards goal
        distance_delta = self.previous_distance - current_distance
        
        # Progress rewards
        if heading_diff < math.pi/2:  # Facing generally towards target
            reward += heading_bonus  # Bonus for facing target
            if distance_delta > 0:  # Making progress towards target
                # Scale progress reward by distance to target
                progress_scale = min(current_distance / self.success_distance, 1.0)
                reward += distance_delta * distance_scaling_factor * progress_scale

        # Update previous distance
        self.previous_distance = current_distance

        # Small penalty per step to encourage efficiency
        reward += step_penalty
            
        # Debug info (every 1000 steps)
        if self.total_steps % 1000 == 0:
            self.node.get_logger().info(
                f'Status: Target x,y: ({self.target_positions_x:.2f}, {self.target_positions_y:.2f}), '
                f'Distance: {current_distance:.2f}m, '
                f'Target Heading: {math.degrees(target_heading):.1f}°, '
                f'Heading diff: {math.degrees(heading_diff):.1f}°, '
                f'Distance delta: {distance_delta:.3f}m, '
                f'Progress reward: {(distance_delta * distance_scaling_factor):.3f}, '
                f'Min lidar distance: {min_distance:.2f}m, '
                f'Total Reward: {reward:.3f}'
            )
            
        return reward
            

    def get_target_info(self):
        """Calculate distance and azimuth to current target"""
        if self.current_pose is None:
            return np.array([0.0, 0.0], dtype=np.float32)
        
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        #target_x, target_y = self.target_positions[self.current_target_idx]
        target_x = self.target_positions_x
        target_y = self.target_positions_y 
    
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
    

    def check_flip_status(self):
        """Check if the robot has flipped based on IMU data"""
        # Use absolute roll and pitch to detect if robot is tilted too much
        if (abs(self.current_roll) > self.flip_threshold
            or abs(self.current_pitch) > self.flip_threshold):
            self.is_flipped = True
            return True
        return False
    

    def reset(self, seed=None, options=None):
        print('################ Environment Reset')
        print('')
        """Reset the environment to its initial state"""
        super().reset(seed=seed)
    
        # Reset robot pose using ign service
        try:
            reset_cmd = [
                'ign', 'service', '-s', self.world_pose_path,
                '--reqtype', 'ignition.msgs.Pose',
                '--reptype', 'ignition.msgs.Boolean',
                '--timeout', '2000',
                '--req', 'name: "rover_zero4wd", position: {x: 0, y: 0, z: 1.0}, orientation: {x: 0, y: 0, z: 0, w: 1}'
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
        self.target_positions_x = np.random.uniform(-3.5, 2.5)  # Random x between -3.5 and 2.5
        self.target_positions_y = np.random.uniform(-2.5, 5.0)   # Random y between -2.5 and 5.0
        #self.current_target_idx = 0
        self.previous_distance = None
        
        # Add a small delay to ensure the robot has time to reset
        for _ in range(100):  # Increased from 3 to 5 to allow more time for pose reset
            rclpy.spin_once(self.node, timeout_sec=0.1)
    
        observation = {
            'lidar': self.lidar_data,
            'pose': self.rover_position,
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

        # Convert to numpy array and clip values
        lidar_data = np.array(msg.ranges, dtype=np.float32)
        lidar_data = np.clip(lidar_data, 0, self.max_lidar_range)
        
        # Reshape into 16 segments and take mean of each segment
        segment_size = len(lidar_data) // 16
        reshaped_data = lidar_data[:segment_size * 16].reshape(16, segment_size)
        self.lidar_data = np.mean(reshaped_data, axis=1)
        
        self._received_scan = True
            
            
    def lidar_callbackOLD(self, msg):
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


