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
        self.reset_simulation_client = self.node.create_client(Empty, '/world/maze/reset')

        # Initialize publishers and subscribers
        self.publisher = self.node.create_publisher(Twist, cmd_vel_topic, 10)
        self.lidar_subscriber = self.node.create_subscription(LaserScan, scan_topic,
                                                            self.lidar_callback, 10)
        self.odom_subscription = self.node.create_subscription(Odometry, odom_topic,
                                                               self.odom_callback, 10)
        self.imu_subscriber = self.node.create_subscription(Imu, imu_topic,
                                                            self.imu_callback, 10)
        #self.set_entity_state_client = self.node.create_client(SetEntityState, '/set_entity_state')
        
        # Initialize environment parameters
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

            success = self.reset_simulation()
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
        reward = self.calc_wall_following_reward()
        
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
                f"climbing_status: {climbing_status},  climbing_severity: {climbing_severity},  "
                f"Pitch: {round(self.current_pitch,3)},  Roll: {round(self.current_roll,3)},  "
                f"min lidar: {round(np.nanmin(self.lidar_data),3)}   Yaw: {round(self.current_yaw,3)},  "
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

    def reset_simulation(self):
        """Reset the Gazebo simulation"""
        if not self.reset_simulation_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().error('Reset service not available')
            return False

        #request = Empty.Request()
        #future = self.reset_simulation_client.call_async(request)
    
        # Wait for result
        #rclpy.spin_until_future_complete(self.node, future)
        twist.linear.x = -0.1
        twist.angular.z = 0.0 #-self.current_roll * 1.0
        self.publisher.publish(twist)
        
        if future.result() is not None:
            self.node.get_logger().info('Simulation reset successful')
            self.is_flipped = False
            return True
        else:
            self.node.get_logger().error('Failed to reset simulation')
            return False

    def reset(self, seed=None, options=None):
        """Reset the environment to its initial state"""
        super().reset(seed=seed)
    
        # Reset internal state
        self._step = 0
        self.last_linear_velocity = 0.0
        self.steps_since_correction = self.cooldown_steps
        self.is_flipped = False
        
        # Reset the robot's position
        #request = SetEntityState.Request()
        #state = EntityState()
        #state.name = 'rover'  # Replace with your robot's actual model name in Gazebo
        #state.pose.position.x = 0.0  # Desired x position
        #state.pose.position.y = 0.0  # Desired y position
        #state.pose.position.z = 0.0  # Desired z position
        
        # Set orientation (if needed)
        #state.pose.orientation.x = 0.0
        #state.pose.orientation.y = 0.0
        #state.pose.orientation.z = 0.0
        #state.pose.orientation.w = 1.0
        #request.state = state

        # Wait for the service to be available
        """
        if not self.set_entity_state_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().error('SetEntityState service not available')
        else:
            future = self.set_entity_state_client.call_async(request)
            rclpy.spin_until_future_complete(self.node, future)
            if future.result() is not None:
                self.node.get_logger().info('SetEntityState request successful')
            else:
                self.node.get_logger().error('SetEntityState request failed')

        """
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


    def calc_wall_following_reward(self):
        # Constants
        desired_distance = 0.5
        collision_threshold = 0.25
        min_forward_velocity = 0.05
        
        # Get right-side distances (keeping your existing code)
        lidar_ranges = self.lidar_data
        num_readings = len(lidar_ranges)
        
        right_start_angle_deg = 250
        right_end_angle_deg = 290
        
        degrees_per_index = 360 / num_readings
        right_start_idx = int(right_start_angle_deg / degrees_per_index) % num_readings
        right_end_idx = int(right_end_angle_deg / degrees_per_index) % num_readings
        
        if right_start_idx <= right_end_idx:
            right_side_indices = np.arange(right_start_idx, right_end_idx + 1)
        else:
            right_side_indices = np.concatenate((
                np.arange(right_start_idx, num_readings),
                np.arange(0, right_end_idx + 1)
            ))
        
        right_distances = lidar_ranges[right_side_indices]
        right_distances = right_distances[np.isfinite(right_distances)]
    
        if len(right_distances) == 0:
            return 0.0
    
        # 1. Distance maintenance reward
        average_distance = np.mean(right_distances)
        distance_error = abs(average_distance - desired_distance)
        distance_reward = np.exp(-2.0 * distance_error)  # Exponential decay
    
        # 2. Collision penalty
        min_distance = np.min(lidar_ranges[np.isfinite(lidar_ranges)])
        if min_distance < collision_threshold:
            collision_penalty = -2.0
        else:
            collision_penalty = 0.0
    
        # 3. Forward motion reward
        if self.last_linear_velocity > min_forward_velocity:
            forward_reward = 0.5 * (self.last_linear_velocity / 0.3)  # Normalized by max velocity
        else:
            forward_reward = 0.0
    
        # 4. Stability reward (penalize oscillations)
        angular_velocity = ( float(self.last_angular_velocity)
                             if hasattr(self, 'last_angular_velocity') else 0.0 )
        stability_reward = -0.1 * abs(angular_velocity)
    
        # Combine rewards
        total_reward = (
            0.4 * distance_reward +    # Distance maintenance is primary objective
            0.4 * forward_reward +     # Encourage forward motion
            0.2 * stability_reward +   # Discourage oscillations
            collision_penalty          # Safety critical, so added separately
        )
    
        # Store angular velocity for next step
        self.last_angular_velocity = angular_velocity
    
        return total_reward

    
    def render(self):
        """Render the environment (optional)"""
        pass


    def close(self):
        """Clean up resources"""
        self.node.destroy_node()
        rclpy.shutdown()



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


