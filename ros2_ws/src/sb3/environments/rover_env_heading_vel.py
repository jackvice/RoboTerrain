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
                 connection_check_timeout=30, lidar_points=32, max_lidar_range=12.0):

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
        self.rand_x_range = (-23,-18) #x(-5.4, -1) # moon y(-9.3, -0.5) # moon,  x(-3.5, 2.5) 
        self.rand_y_range = (-25,-20) # -27,-19 for inspection
        self.target_positions_x = 0
        self.target_positions_y = 0
        self.previous_distance = None
        self.the_world = 'default'
        self.world_pose_path = '/world/' + self.the_world + '/set_pose'
        self.too_far_away_low_x = -29 # 17 for inspection
        self.too_far_away_high_x =-13  # 29 for inspection
        self.too_far_away_low_y = -29 # for inspection
        self.too_far_away_high_y = -17.5  # 29 for inspection
        self.too_far_away_penilty = -5.0
        # Define action space
        # [speed, desired_heading]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -np.pi]),  # [min_speed, min_heading]
            high=np.array([1.0, np.pi]),   # [max_speed, max_heading]
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


    def too_far_away(self):
        if ( self.current_pose.position.x < self.too_far_away_low_x or
             self.current_pose.position.x > self.too_far_away_high_x or
             self.current_pose.position.y < self.too_far_away_low_y or
             self.current_pose.position.y > self.too_far_away_high_y):
            print('too far, x, y is', self.current_pose.position.x,
                  self.current_pose.position.y, ', episode done. ********************')
            return True
        else:
            return False


    def step(self, action):
        """Execute one time step within the environment"""
        self.total_steps += 1

        if self.too_far_away():
            return self.get_observation(), self.too_far_away_penilty, True, False, {}  

        flip_status = self.is_robot_flipped()
        if flip_status:
            print('Robot flipped', flip_status, ', episode done')
            if self.total_steps > 100:
                return self.get_observation(), -100, True, False, {}
            else:
                return self.get_observation(), 0, True, False, {} 

        
        # Wait for new sensor data
        while not self._received_scan:
            rclpy.spin_once(self.node, timeout_sec=0.01)

        
        self._received_scan = False
        
        # Extract speed and desired heading from action
        speed = float(action[0])
        desired_heading = float(action[1])
        
        # Store last actions
        self.last_speed = speed
        self.last_heading = desired_heading
        
        # Calculate angular velocity using PID controller
        angular_velocity = self.heading_controller(desired_heading, self.current_yaw)
        
        twist = Twist()

        # Normal operation
        twist.linear.x = speed
        twist.angular.z = angular_velocity
        
        self.publisher.publish(twist)
        

        # Calculate reward and components
        reward, reward_components = self.task_reward()

        # Check if episode is done
        self._step += 1
        done = (self._step >= self._length)
        
        # Get observation
        observation = self.get_observation()

        
        # Info dict for additional information
        info = {
            'steps': self._step,
            'total_steps': self.total_steps,
            'reward_components': reward_components
        }
        

        if self.total_steps % 1000 == 0:
            temp_obs_target = observation["target"]
            print(
                f"current target x,y: ({self.target_positions_x}, {self.target_positions_y}), "
                f"distance and angle to target: {temp_obs_target}, "
                f"Speed: {speed:.2f}, Heading: {math.degrees(desired_heading):.1f}°"
                f"Final Reward: {reward:.3f}, "
            )
        if self.total_steps % 2000 == 0:
            print('Observation: Pose:', observation['pose'],
                  ', IMU:', observation['imu'],
                  ', target:', observation['target'],
                  )
        return observation, reward, done, False, info


    def get_observation(self):
        return {
            'lidar': self.lidar_data,
            'pose': self.rover_position,
            'imu': np.array([self.current_pitch, self.current_roll, self.current_yaw],
                            dtype=np.float32),
            'target': self.get_target_info()
        }    
    

    def task_reward(self):
        """Reward function with balanced rewards and detailed logging"""
        # Constants
        collision_threshold = 0.4
        collision_reward = -1.0        # Increased due to recovery time
        warning_distance = 1.0         # 2x collision threshold
        distance_scaling_factor = 20.0  # Reduced from 20
        goal_reward = 50.0           # Kept as requested
        step_penalty = -0.01        # Further reduced
        heading_bonus = 0.2         # Further reduced
        reverse_penalty = -0.25       # Reduced but still discouraged
        success_distance = 0.5
        max_possible_distance = 18.0

        
        # Initialize reward components dictionary for logging
        reward_components = {
            'collision': 0.0,
            'heading': 0.0,
            'progress': 0.0,
            'step': step_penalty,
            'motion': 0.0,
            'goal': 0.0
        }
        
        if self.current_pose is None:
            return 0.0, reward_components

        # Calculate distance to current target
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y

        distance_heading_angle = self.get_target_info()
        current_distance = distance_heading_angle[0]
        heading_diff = distance_heading_angle[1]
        
        # Initialize previous_distance if not set
        if self.previous_distance is None:
            self.previous_distance = current_distance
            return 0.0, reward_components
        
        # Success reward
        if current_distance < success_distance:
            reward_components['goal'] = goal_reward
            print('###################################################### GOAL ACHIVED!')
            self.target_positions_x = np.random.uniform(*self.rand_x_range)
            self.target_positions_y = np.random.uniform(*self.rand_y_range)
            self.previous_distance = None
            return goal_reward, reward_components

        # Collision checking with graduated penalty
        min_distance = np.min(self.lidar_data[np.isfinite(self.lidar_data)])
        if min_distance < collision_threshold:
            reward_components['collision'] = collision_reward
            return collision_reward, reward_components
        elif min_distance < warning_distance:
            warning_factor = (warning_distance - min_distance) / collision_threshold
            reward_components['collision'] = collision_reward * 0.5 * warning_factor

        # Movement rewards/penalties
        if self.last_speed > 0.0:
            motion_reward = self.last_speed / 40  # Reduced from /5
            reward_components['motion'] = motion_reward
        else:
            reward_components['motion'] = reverse_penalty

        # Progress rewards
        distance_delta = self.previous_distance - current_distance
        if heading_diff < math.pi/2:  # Facing generally towards target
            reward_components['heading'] = heading_bonus
            if distance_delta > 0:
                progress_scale = min(current_distance / max_possible_distance, 1.0)
                progress_reward = distance_delta * distance_scaling_factor * progress_scale
                reward_components['progress'] = progress_reward

        # Update previous distance
        self.previous_distance = current_distance

        # Calculate total reward
        total_reward = sum(reward_components.values())

        # Debug logging (every 100 steps)
        if self.total_steps % 1000 == 0:
            self.node.get_logger().info(
                f'\nReward Components:'
                f'\n- Collision: {reward_components["collision"]:.3f}'
                f'\n- Heading: {reward_components["heading"]:.3f}'
                f'\n- Progress: {reward_components["progress"]:.3f}'
                f'\n- Step: {reward_components["step"]:.3f}'
                f'\n- Motion: {reward_components["motion"]:.3f}'
                f'\n- Goal: {reward_components["goal"]:.3f}'
                f'\nDistance to target: {current_distance:.2f}m'
                f'\nHeading difference: {math.degrees(heading_diff):.1f}°'
                f'\nTotal Reward: {total_reward:.3f}'
            )
            
        return total_reward, reward_components

    
    def get_target_info(self):
        """Calculate distance and azimuth to current target"""
        if self.current_pose is None:
            return np.array([0.0, 0.0], dtype=np.float32)
        
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        target_x = self.target_positions_x
        target_y = self.target_positions_y 
    
        # Calculate distance
        distance = math.sqrt(
            (current_x - target_x)**2 + 
            (current_y - target_y)**2
        )
    
        # Calculate azimuth (relative angle to target)
        target_heading = math.atan2(target_y - current_y, target_x - current_x)
        relative_angle = abs(math.atan2(math.sin(target_heading - self.current_yaw), 
                               math.cos(target_heading - self.current_yaw)
                                        )
                             )
    
        return np.array([distance, relative_angle], dtype=np.float32)


    def is_robot_flipped(self):
        """Detect if robot has flipped in any direction past 85 degrees"""
        FLIP_THRESHOLD = 1.48  # ~85 degrees in radians
        
        # Check both roll and pitch angles
        if abs(self.current_roll) > FLIP_THRESHOLD:
            print('flipped')
            return 'roll_left' if self.current_roll > 0 else 'roll_right'
        elif abs(self.current_pitch) > FLIP_THRESHOLD:
            print('flipped')
            return 'pitch_forward' if self.current_pitch < 0 else 'pitch_backward'
        
        return False
        
    def reset(self, seed=None, options=None):
        print('################ Environment Reset')
        print('')
        """Reset the environment to its initial state"""
        super().reset(seed=seed)
        x_insert = np.random.uniform(*self.rand_x_range)
        y_insert = np.random.uniform(*self.rand_y_range)
        z_insert = 5.5
        if x_insert < -24.5 and y_insert < -24.5:
            z_insert = 6.5
        
        # Generate random yaw angle (in radians) between -π and π
        random_yaw = np.random.uniform(-np.pi, np.pi)
        
        # Convert yaw to quaternion (keeping roll and pitch as 0)
        # For yaw only, the conversion is:
        # w = cos(yaw/2)
        # z = sin(yaw/2)
        # x and y remain 0
        quat_w = np.cos(random_yaw / 2)
        quat_z = np.sin(random_yaw / 2)
        
        # Reset robot pose using ign service
        try:
            reset_cmd = [
                'ign', 'service', '-s', self.world_pose_path,
                '--reqtype', 'ignition.msgs.Pose',
                '--reptype', 'ignition.msgs.Boolean',
                '--timeout', '2000',
                '--req', 'name: "rover_zero4wd", position: {x: ' + str(x_insert) +
                ',y: '+ str(y_insert) +
                ', z: '+ str(z_insert) + '}, orientation: {x: 0, y: 0, z: ' +
                str(quat_z) + ', w: ' + str(quat_w) + '}'
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
        self.target_positions_x = np.random.uniform(*self.rand_x_range)  # Random x between -3.5 and 2.5
        self.target_positions_y = np.random.uniform(*self.rand_y_range)   # Random y between -2.5 and 5.0
        self.previous_distance = None
        
        # Add a small delay to ensure the robot has time to reset
        for _ in range(100):  # Increased from 3 to 5 to allow more time for pose reset
            rclpy.spin_once(self.node, timeout_sec=0.1)
            
        observation = self.get_observation()
        twist = Twist()
        # Normal operation
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        
        self.publisher.publish(twist)
        return observation, {}
    
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
            print(f"Observation lidar size: {self.lidar_points}")
            self.angle_min = msg.angle_min
            self.angle_max = msg.angle_max
            self.angle_increment = msg.angle_increment
            self.first = True

        # Convert to numpy array and clip values
        lidar_data = np.array(msg.ranges, dtype=np.float32)
    
        # Add Gaussian noise
        gaussian_noise = np.random.normal(0, 0.05, size=lidar_data.shape)  # 0.1m standard deviation
        lidar_data = lidar_data + gaussian_noise
    
        # Add random dropouts (set some measurements to max_range)
        dropout_mask = np.random.random(lidar_data.shape) < 0.05  # 5% chance of dropout
        lidar_data[dropout_mask] = self.max_lidar_range
        
        # Add distance-dependent noise (noise increases with distance)
        #distance_noise = np.random.normal(0, 0.02 * lidar_data, size=lidar_data.shape)
        #lidar_data = lidar_data + distance_noise
    
        # Add some spurious short readings
        #short_readings_mask = np.random.random(lidar_data.shape) < 0.02  # 2% chance of spurious short reading
        #lidar_data[short_readings_mask] = lidar_data[short_readings_mask] * np.random.un
     
        # Clip values to valid range
        lidar_data = np.clip(lidar_data, 0, self.max_lidar_range)
    
        # Reshape into segments and take mean of each segment
        segment_size = len(lidar_data) // self.lidar_points
        reshaped_data = lidar_data[:segment_size *
                                   self.lidar_points].reshape(self.lidar_points, segment_size)
        self.lidar_data = np.mean(reshaped_data, axis=1)
    
        self._received_scan = True
            

    def lidar_callback_origin(self, msg):
        if not self.first:
            print("First scan received:")
            print(f"Number of points: {len(msg.ranges)}")
            print(f"Angle min: {msg.angle_min}, Angle max: {msg.angle_max}")
            print(f"Angle increment: {msg.angle_increment}")
            print(f"Range min: {msg.range_min}, Range max: {msg.range_max}")
            print(f"First 20 ranges: {msg.ranges[:20]}")
            print(f"Last 20 ranges: {msg.ranges[-20:]}")
            print(f"Observation lidar size: {self.lidar_points}")
            self.angle_min = msg.angle_min
            self.angle_max = msg.angle_max
            self.angle_increment = msg.angle_increment
            self.first = True

        # Convert to numpy array and clip values
        lidar_data = np.array(msg.ranges, dtype=np.float32)
        lidar_data = np.clip(lidar_data, 0, self.max_lidar_range)
        
        # Reshape into 16 segments and take mean of each segment
        segment_size = len(lidar_data) // self.lidar_points
        reshaped_data = lidar_data[:segment_size * self.lidar_points].reshape(self.lidar_points,
                                                                              segment_size)
        self.lidar_data = np.mean(reshaped_data, axis=1)
        
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


