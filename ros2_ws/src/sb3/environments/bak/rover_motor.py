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
from sensor_msgs.msg import Image
import cv2
from collections import deque


class RoverEnv(gym.Env):
    """Custom Environment that follows gymnasium interface"""
    metadata = {'render_modes': ['human']}
    def __init__(self, size=(64, 64), length=3000, scan_topic='/scan', imu_topic='/imu/data',
                 cmd_vel_topic='/cmd_vel', camera_topic='/camera/image_raw',
                 connection_check_timeout=30, lidar_points=32, max_lidar_range=12.0):

        super().__init__()
        
        # Initialize ROS2 node and publishers/subscribers
        rclpy.init()
        self.bridge = CvBridge()
        self.node = rclpy.create_node('turtlebot_controller')


        self.the_world = 'maze' #'default' #default for inspection

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

        self.bridge = CvBridge()
        self.camera_subscriber = self.node.create_subscription(
            Image,
            camera_topic,
            self.camera_callback,
            10)

        # Add this in __init__ with your other subscribers
        self.odom_subscriber = self.node.create_subscription(
            Odometry,
            '/odometry/wheels',  
            self.odom_callback,
            10)

        self.current_image = np.zeros((64, 64), dtype=np.float32)  # grayscale image buffer

        # Initialize these in __init__
        self.current_linear_velocity = 0.0
        self.current_angular_velocity = 0.0

        
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
        self.min_raw_lidar = 100
        
        # Stuck detection parameters
        self.position_history = []
        self.stuck_threshold = 0.02  # Minimum distance the robot should move
        self.stuck_window = 600    # Number of steps to check for being stuck
        self.stuck_penilty = -25.0

        # Collision detection parameters
        self.collision_history = []
        self.collision_window = 15  # Number of steps to check for persistent collision
        self.collision_count = 0
        
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

        self.yaw_history = deque(maxlen=200)
        
        # Add these as class variables in your environment's __init__
        self.down_facing_training_steps = 200000  # Duration of temporary training
        self.heading_steps = 0  # Add this to track when training began
        
        # Navigation parameters previous
        #self.rand_goal_x_range = (-26, -21) #x(-5.4, -1) # moon y(-9.3, -0.5) # moon,  x(-3.5, 2.5) 
        #self.rand_goal_y_range = (-25, -20) # -27,-19 for inspection
        #self.rand_x_range = (-21, -14) #x(-5.4, -1) # moon y(-9.3, -0.5) # moon,  x(-3.5, 2.5) 
        #self.rand_y_range = (-28.5, -19.5) # -27,-19 for inspection

        self.rand_goal_x_range = (-7, 7) #(-23, -14) 
        self.rand_goal_y_range = (-7, 7)# (-28, -19) 
        self.rand_x_range =  (-7, 7) #(-23, -14) #x(-5.4, -1) # moon y(-9.3, -0.5) moon, x(-3.5, 2.5) 
        self.rand_y_range =  (-7, 7) #(-28, -19) 
        self.target_positions_x = 0
        self.target_positions_y = 0
        self.previous_distance = None

        self.world_pose_path = '/world/' + self.the_world + '/set_pose'
        self.too_far_away_low_x = -30 # 17 for inspection
        self.too_far_away_high_x = 30#-13  # 29 for inspection
        self.too_far_away_low_y = -30 # for inspection
        self.too_far_away_high_y = -17.5  # 29 for inspection
        self.too_far_away_penilty = -50#-25.0

        self.goal_reward = 100
        
        self.last_time = time.time()
        # Add at the end of your existing __init__ 
        self.heading_log = []  # To store headings
        self.heading_log_file = "initial_headings.csv"
        self.heading_log_created = False
        
        # Define action space
        # [speed, desired_heading]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -3.0]),  # [min_speed, min_angular]
            high=np.array([1.0, 3]),   # [max_speed, max_angular]
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
                low=np.array([-30.0, -30.0, -10.0]),
                high=np.array([30.0, 30.0, 10.0]),
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
            ),
            #'image': spaces.Box(
            #    low=0,
            #    high=255,
            #    shape=(64, 64),
            #    dtype=np.float32
            #),
            'velocities': spaces.Box(
                low=np.array([-10.0, -10.0]),
                high=np.array([10.0, 10.0]),
                shape=(2,),
                dtype=np.float32
            ),
        })

        # Check robot connection
        self._robot_connected = self._check_robot_connection(timeout=connection_check_timeout)
        if not self._robot_connected:
            self.node.get_logger().warn("No actual robot detected. Running in simulation mode.")



    def too_far_away(self):
        if ( self.current_pose.position.x < self.too_far_away_low_x or
             self.current_pose.position.x > self.too_far_away_high_x or
             self.current_pose.position.y < self.too_far_away_low_y or
             self.current_pose.position.y > self.too_far_away_high_y):
            print('too far, x, y is', self.current_pose.position.x,
                  self.current_pose.position.y, ', episode done. ************** reward',
                  self.too_far_away_penilty)
            return True
        else:
            return False


    def step(self, action):
        """Execute one time step within the environment"""
        self.total_steps += 1

        if self.collision_count > self.stuck_window:
            self.collision_count = 0
            print('stuck in collision, ending episode')
            return self.get_observation(), -1 * self.goal_reward, True, False, {}  

        flip_status = self.is_robot_flipped()
        if flip_status:
            print('Robot flipped', flip_status, ', episode done')
            if self._step > 500:
                print('Robot flipped on its own')
                return self.get_observation(), (-1 * self.goal_reward), True, False, {}
            else:
                return self.get_observation(), 0, True, False, {} 
            

        # Update position history
        self.position_history.append((self.current_pose.position.x, self.current_pose.position.y))
        if len(self.position_history) > self.stuck_window:
            self.position_history.pop(0)

        # Check if robot is stuck - do this every step once we have enough history
        if len(self.position_history) >= self.stuck_window:
            start_pos = self.position_history[0]
            end_pos = self.position_history[-1]
            distance_moved = math.sqrt((end_pos[0] - start_pos[0])**2 +
                                       (end_pos[1] - start_pos[1])**2)

            if distance_moved < self.stuck_threshold:
                print('Robot is stuck, has moved only', distance_moved,
                      'meters in', self.stuck_window, 'steps, resetting')
                return self.get_observation(), self.stuck_penilty, True, False, {}
            
                
        # Wait for new sensor data
        while not self._received_scan:
            rclpy.spin_once(self.node, timeout_sec=0.01)

        
        self._received_scan = False
        
        twist = Twist()
        speed = float(action[0])
        self.last_speed = speed
        # Normal operation
        twist.linear.x = speed
        twist.angular.z = float(action[1])
        
        self.publisher.publish(twist)
        

        # Calculate reward and components
        reward = self.task_reward()

        # Check if episode is done
        self._step += 1
        done = (self._step >= self._length)
        
        # Get observation
        observation = self.get_observation()


        if self.total_steps % 1000 == 0:
            temp_obs_target = observation["target"]
            print(
                f"current target x,y: ({self.target_positions_x:.2f}, {self.target_positions_y:.2f}), "
                f"distance and angle to target: ({temp_obs_target[0]:.3f}, {temp_obs_target[1]:.3f}), "
                f"Speed: {speed:.2f}, ", #Heading: {math.degrees(desired_heading):.1f}°"
                f"Final Reward: {reward:.3f}"
            )
        
        if self.total_steps % 1000 == 0:
            print('Observation: Pose:', observation['pose'],
                  ', IMU:', observation['imu'],
                  ', target:', observation['target'],
                  )
        
        info = {
            'steps': self._step,
            'total_steps': self.total_steps,
            'reward': reward
        }
        return observation, reward, done, False, info


    def get_observation(self):
        return {
            'lidar': self.lidar_data,
            'pose': self.rover_position,
            'imu': np.array([self.current_pitch, self.current_roll, self.current_yaw],
                            dtype=np.float32),
            'target': self.get_target_info(),
            #'image': self.current_image,
            'velocities': np.array([self.current_linear_velocity, self.current_angular_velocity],
                                   dtype=np.float32)
        }    


    def update_target_pos(self):
        print('###################################################### GOAL ACHIVED!')
        self.target_positions_x = np.random.uniform(*self.rand_goal_x_range)
        self.target_positions_y = np.random.uniform(*self.rand_goal_y_range)
        self.previous_distance = None
        return
    
    
    def task_reward(self):
        """Reward function with balanced rewards and detailed logging"""
        # Constants
        collision_threshold = 0.2
        collision_reward = -1.0        # Increased due to recovery time
        distance_scaling_factor = 5.0  # Reduced from 20
        step_penalty = -0.01        # Further reduced
        heading_bonus = 0.02         # Further reduced
        reverse_penalty = -0.02       # Reduced but still discouraged
        success_distance = 0.3
        max_possible_distance = 30.0
        
        # Initialize reward components dictionary for logging
        reward_components = {
            'collision': 0.0,
            'heading': 0.0,
            'progress': 0.0,
            'step': step_penalty,
            'motion': 0.0,
            'goal': 0.0,
            'down_facing': 0.0,  # New component
            'yaw_alignment': 0.0
        }
        
        if self.current_pose is None:
            return 0.0

        # Calculate distance to current target
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y

        distance_heading_angle = self.get_target_info()
        current_distance = distance_heading_angle[0]
        heading_diff = distance_heading_angle[1]
        # Initialize previous_distance if not set
        if self.previous_distance is None:
            self.previous_distance = current_distance
            return 0.0
        
        # Success reward
        if current_distance < success_distance:
            self.update_target_pos()
            return self.goal_reward


        # Collision detection with persistence check
        min_distance = np.min(self.lidar_data[np.isfinite(self.lidar_data)])
        self.collision_history.append(min_distance < collision_threshold)

        # Keep only the last N steps
        if len(self.collision_history) > self.collision_window:
            self.collision_history.pop(0)

        # Check if we've been in collision state for the entire window
        if len(self.collision_history) == self.collision_window and all(self.collision_history):
            print(f'Persistent collision detected, min distance is {min_distance:.3f}')
            reward_components['collision'] = collision_reward
            self.collision_count +=1
        else:
            self.collision_count = 0
        
        # Movement rewards/penalties
        if self.last_speed > 0.0:
            motion_reward = 0 #self.last_speed / 40  # Reduced from /5
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

        """
        # Then in task_reward(), add this before calculating total_reward:
        # Convert current yaw from radians to degrees
        current_heading_deg = math.degrees(self.current_yaw)
        if self.heading_steps < self.down_facing_training_steps:  # Only apply for period
            self.too_far_away_penilty = -50.0
            reward_components['yaw_alignment'] = self.yaw_to_reward(self.current_yaw)
            if abs(current_heading_deg) > 170:
                reward_components['down_facing'] = -0.5
        else:
            self.too_far_away_penilty = -25.0
        self.heading_steps += 1
        """
        
        # Calculate total reward
        total_reward = sum(reward_components.values())

        # Debug logging (every 100 steps)
        if self.total_steps % 1000 == 0:
            self.debug_logging(heading_diff, current_distance, reward_components, total_reward)
            print(f"Current yaw (rad): {self.current_yaw:.2f}, (deg): {math.degrees(self.current_yaw):.2f}")
                
        return total_reward


    def yaw_to_reward(self, yaw_rad):
        """
        Maps absolute yaw to reward value:
        |yaw| = 180° (π rad) -> 0.0
        |yaw| = 0° (0 rad) -> 0.5
        """
        abs_yaw_deg = abs(math.degrees(yaw_rad))
        # Linear interpolation: (180 - abs_yaw) * (0.5/180)
        reward = (180 - abs_yaw_deg) * (0.5/180)
        # Clamp between 0 and 0.5 to handle angles > 180
        return max(0.0, min(0.5, reward))

    
    def get_yaw_delta(self):
        """
        Calculate absolute difference between current yaw and yaw from 200 steps ago.
        Returns 0 if there isn't 200 steps of history yet.
        """
        # Add current yaw to history
        self.yaw_history.append(self.current_yaw)
    
        # If we don't have enough history yet, return 0
        if len(self.yaw_history) < 200:
            return 0.0
        
        # Get yaw from 200 steps ago
        old_yaw = self.yaw_history[0]
    
        # Calculate absolute difference
        # Using atan2 to handle the circular nature of angles
        yaw_diff = abs(math.atan2(math.sin(self.current_yaw - old_yaw), 
                                  math.cos(self.current_yaw - old_yaw)))
                             
        return yaw_diff

    
    def check_sample_rate_performance(self):
        """
        Monitors the sample rate and returns the duration (in seconds) that it has been below threshold.
        Returns 0.0 if sample rate is currently above threshold.
        """
        current_time = time.time()
        
        # Initialize tracking attributes if they don't exist
        if not hasattr(self, '_last_sample_time'):
            self._last_sample_time = current_time
            self._below_threshold_start = None
            self._frame_times = []
            return 0.0
        
        # Calculate current sample rate
        sample_interval = current_time - self._last_sample_time
        current_rate = 1.0 / sample_interval if sample_interval > 0 else float('inf')
        
        # Update frame times list (keep last 10 frames for smoothing)
        self._frame_times.append(current_rate)
        if len(self._frame_times) > 10:
            self._frame_times.pop(0)
        
        # Calculate average frame rate over the last 10 frames
        avg_rate = sum(self._frame_times) / len(self._frame_times)
        
        # Check if we're below threshold
        if avg_rate < self.min_sample_rate:
            if self._below_threshold_start is None:
                self._below_threshold_start = current_time
            duration_below_threshold = current_time - self._below_threshold_start
        else:
            self._below_threshold_start = None
            duration_below_threshold = 0.0
        
        # Update last sample time
        self._last_sample_time = current_time
        
        return duration_below_threshold    

    def debug_logging(self, heading_diff, current_distance, reward_components, total_reward):
        self.node.get_logger().info(
            f'\nTarget x,y: {self.target_positions_x:.2f}, {self.target_positions_y:.2f}'
            f'\nCurrent x,y: {self.current_pose.position.x:.2f}, {self.current_pose.position.y:.2f}'
            f'\nHeading difference: {math.degrees(heading_diff):.1f}° '
            f'\nDistance to target: {current_distance:.2f}m'
            f'\n- Collision: {reward_components["collision"]:.3f}'
            f'\n- Heading: {reward_components["heading"]:.3f}'
            f'\n- Progress: {reward_components["progress"]:.3f}'
            f'\n- Step: {reward_components["step"]:.3f}'
            f'\n- Motion: {reward_components["motion"]:.3f}'
            f'\n- Goal: {reward_components["goal"]:.3f}'
            f'\n- down facing: {reward_components["down_facing"]:.3f}'
            f'\n- yaw_alignment: {reward_components["yaw_alignment"]:.3f}'
            f'\nTotal Reward: {total_reward:.3f}'
        )


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
        twist = Twist()
        # Normal operation
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.publisher.publish(twist)
        """Reset the environment to its initial state"""
        super().reset(seed=seed)
        self.collision_history = []  # Clear collision history on reset
        x_insert = np.random.uniform(*self.rand_x_range)
        y_insert = np.random.uniform(*self.rand_y_range)
        z_insert = 2 # for maze
        #z_insert = 5.5 # for inspection
        #if x_insert < -24.5 and y_insert < -24.5: inspection
        #    z_insert = 6.5 

        ##  Random Yaw
        final_yaw = np.random.uniform(-np.pi, np.pi)
        print(f"Generated heading: {math.degrees(final_yaw)}°")
        # Normalize to [-pi, pi] range
        final_yaw = np.arctan2(np.sin(final_yaw), np.cos(final_yaw))

        # make final yaw point down cause normal action distribution pointing up.
        #final_yaw = np.pi + np.random.uniform(-0.2, 0.2)
        
        quat_w = np.cos(final_yaw / 2)
        quat_z = np.sin(final_yaw / 2)

        # Print the full reset command
        reset_cmd_str = ('name: "rover_zero4wd", ' +
                        f'position: {{x: {x_insert}, y: {y_insert}, z: {z_insert}}}, ' +
                        f'orientation: {{x: 0, y: 0, z: {quat_z}, w: {quat_w}}}')
        #print("\nFull reset command:")
        #print(reset_cmd_str)
        
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
        self.target_positions_x = np.random.uniform(*self.rand_goal_x_range)
        self.target_positions_y = np.random.uniform(*self.rand_goal_y_range)
        self.previous_distance = None
        
        # Add a small delay to ensure the robot has time to reset
        for _ in range(100):  # Increased from 3 to 5 to allow more time for pose reset
            rclpy.spin_once(self.node, timeout_sec=0.1)
        time.sleep(1.0)        
        observation = self.get_observation()

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
        

    def camera_callback(self, msg):
        try:
            # Convert ROS Image to CV2, then to grayscale
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            # Resize to 64x64
            self.current_image = cv2.resize(cv_image, (64, 64))
        except Exception as e:
            self.node.get_logger().warn(f"Failed to process image: {e}")

            
    def lidar_callbackNoise(self, msg):# function needs to be double checked!!!!
        """Process LIDAR data with error checking and downsampling."""
        # Convert to numpy array
        try:
            lidar_data = np.array(msg.ranges, dtype=np.float32)
        except Exception as e:
            print(f"Error converting LIDAR data to numpy array: {e}")
            return

        # First handle all invalid values
        # Replace inf values with max_lidar_range
        inf_mask = np.isinf(lidar_data)
        if np.any(inf_mask):
            lidar_data[inf_mask] = self.max_lidar_range

        # Replace any remaining invalid values (NaN, negative)
        invalid_mask = np.logical_or(np.isnan(lidar_data), lidar_data < 0)
        if np.any(invalid_mask):
            print(f"INFO: Replaced {np.sum(invalid_mask)} invalid values with max_lidar_range")
            lidar_data[invalid_mask] = self.max_lidar_range

        # Clip values to valid range
        lidar_data = np.clip(lidar_data, 0, self.max_lidar_range)

        # Now add noise to valid data
        #gaussian_noise = np.random.normal(0, 0.05, size=lidar_data.shape)
        #lidar_data = lidar_data + gaussian_noise

        # Clip again after adding noise to ensure no invalid values
        lidar_data = np.clip(lidar_data, 0, self.max_lidar_range)

        # Add random dropouts last (after noise)
        #dropout_mask = np.random.random(lidar_data.shape) < 0.05  # 5% chance of dropout
        #lidar_data[dropout_mask] = self.max_lidar_range

        # Verify we have enough data points for downsampling
        expected_points = self.lidar_points * (len(lidar_data) // self.lidar_points)
        if expected_points == 0:
            print(f"ERROR: Not enough LIDAR points for downsampling. Got {len(lidar_data)} points")
            return

        # Downsample by taking minimum value in each segment
        try:
            segment_size = len(lidar_data) // self.lidar_points
            reshaped_data = lidar_data[:segment_size * self.lidar_points].reshape(self.lidar_points,
                                                                                  segment_size)
            self.lidar_data = np.min(reshaped_data, axis=1)
            
            # Verify downsampled data
            if len(self.lidar_data) != self.lidar_points:
                print(f"ERROR: Downsampled wrong size. Expected {self.lidar_points}, got {len(self.lidar_data)}")
                return
                
            if np.any(np.isnan(self.lidar_data)) or np.any(np.isinf(self.lidar_data)):
                print("ERROR: Downsampled data contains invalid values")
                print("NaN count:", np.sum(np.isnan(self.lidar_data)))
                print("Inf count:", np.sum(np.isinf(self.lidar_data)))
                return
                
        except Exception as e:
            print(f"Error during downsampling: {e}")
            return

        self._received_scan = True
            
    def lidar_callback(self, msg):
        """Process LIDAR data with error checking and downsampling."""

        # Convert to numpy array
        try:
            lidar_data = np.array(msg.ranges, dtype=np.float32)
        except Exception as e:
            print(f"Error converting LIDAR data to numpy array: {e}")
            return

        # Check for invalid values before processing
        #if np.any(np.isneginf(lidar_data)):
        #    print(f"WARNING: Found {np.sum(np.isneginf(lidar_data))} negative infinity values")
            
        if np.any(np.isnan(lidar_data)):
            print(f"WARNING: Found {np.sum(np.isnan(lidar_data))} NaN values")
            
        #if np.any(lidar_data < 0):
        #    print(f"WARNING: Found {np.sum(lidar_data < 0)} negative values")
            #print("Negative values:", lidar_data[lidar_data < 0])

        # Add Gaussian noise
        #gaussian_noise = np.random.normal(0, 0.05, size=lidar_data.shape)  # 0.1m standard deviation
        #lidar_data = lidar_data + gaussian_noise
    
        # Add random dropouts (set some measurements to max_range)
        #dropout_mask = np.random.random(lidar_data.shape) < 0.05  # 5% chance of dropout
        #lidar_data[dropout_mask] = self.max_lidar_range

        # Replace inf values with max_lidar_range
        inf_mask = np.isinf(lidar_data)
        if np.any(inf_mask):
            #print(f"INFO: Replaced {np.sum(inf_mask)} infinity values with max_lidar_range")
            lidar_data[inf_mask] = self.max_lidar_range

        # Replace any remaining invalid values (NaN, negative) with max_range
        invalid_mask = np.logical_or(np.isnan(lidar_data), lidar_data < 0)
        if np.any(invalid_mask):
            print(f"INFO: Replaced {np.sum(invalid_mask)} invalid values with max_lidar_range")
            lidar_data[invalid_mask] = self.max_lidar_range

        # Clip values to valid range
        lidar_data = np.clip(lidar_data, 0, self.max_lidar_range)

        # Verify we have enough data points for downsampling
        expected_points = self.lidar_points * (len(lidar_data) // self.lidar_points)
        if expected_points == 0:
            print(f"ERROR: Not enough LIDAR points for downsampling. Got {len(lidar_data)} points")
            return

        # Downsample by taking minimum value in each segment
        try:
            segment_size = len(lidar_data) // self.lidar_points
            reshaped_data = lidar_data[:segment_size * self.lidar_points].reshape(self.lidar_points,
                                                                                  segment_size)
            self.lidar_data = np.min(reshaped_data, axis=1)
            
            # Verify downsampled data
            if len(self.lidar_data) != self.lidar_points:
                print(f"ERROR: Downsampled has wrong size. Expected {self.lidar_points}, got {len(self.lidar_data)}")
                return
                
            if np.any(np.isnan(self.lidar_data)) or np.any(np.isinf(self.lidar_data)):
                print("ERROR: Downsampled data contains invalid values")
                print("NaN count:", np.sum(np.isnan(self.lidar_data)))
                print("Inf count:", np.sum(np.isinf(self.lidar_data)))
                return
                
        except Exception as e:
            print(f"Error during downsampling: {e}")
            return

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


    # Add this callback
    def odom_callback(self, msg):
        """Process odometry data for velocities"""
        self.current_linear_velocity = msg.twist.twist.linear.x
        self.current_angular_velocity = msg.twist.twist.angular.z

    def _check_robot_connection(self, timeout):
        start_time = time.time()
        while not self._received_scan:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if time.time() - start_time > timeout:
                return False
            if self._received_scan:
                return True
        return False


    
