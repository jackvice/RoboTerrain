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

class RoverEnv(gym.Env):
    """Custom Environment that follows gymnasium interface with flip detection"""
    metadata = {'render_modes': ['human']}

    def __init__(self, size=(64, 64), length=200, scan_topic='/scan', imu_topic='/imu/data',
                 cmd_vel_topic='/cmd_vel', odom_topic='/odometry/wheels', camera_topic='/camera/image_raw',
                 connection_check_timeout=30, lidar_points=640, max_lidar_range=12.0):
        super().__init__()
        
        # Initialize ROS2 node and publishers/subscribers
        rclpy.init()
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

        # Flip detection parameters
        self.flip_threshold = math.pi / 3  # 60 degrees in radians
        self.is_flipped = False
        self.initial_position = None
        self.initial_orientation = None
        
        # Rest of your initialization code remains the same
        [...]  # (keeping existing initialization code)

    def check_flip_status(self):
        """Check if the robot has flipped based on IMU data"""
        # Use absolute roll and pitch to detect if robot is tilted too much
        if abs(self.current_roll) > self.flip_threshold or abs(self.current_pitch) > self.flip_threshold:
            self.is_flipped = True
            return True
        return False

    async def reset_simulation(self):
        """Reset the Gazebo simulation"""
        # Wait for service to be available
        while not self.reset_simulation_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().warn('Reset service not available, waiting...')

        # Create request
        request = Empty.Request()
        
        try:
            # Call reset service
            future = self.reset_simulation_client.call_async(request)
            await future
            self.node.get_logger().info('Simulation reset successful')
            self.is_flipped = False
            return True
        except Exception as e:
            self.node.get_logger().error(f'Failed to reset simulation: {str(e)}')
            return False

    def step(self, action):
        """Execute one time step within the environment"""
        self.total_steps += 1
        
        # Check if robot has flipped
        if self.check_flip_status():
            self.node.get_logger().warn("Robot has flipped! Initiating reset...")
            # Create and publish zero velocity command
            stop_cmd = Twist()
            self.publisher.publish(stop_cmd)
            
            # Reset the simulation
            rclpy.spin_once(self.node)  # Process any pending callbacks
            future = self.reset_simulation()
            
            # Wait for reset to complete
            while rclpy.ok():
                rclpy.spin_once(self.node)
                if future.done():
                    break
            
            # Return observation with large negative reward
            observation = {
                'lidar': self.lidar_data,
                'odom': np.array(self.rover_position, dtype=np.float32),
                'imu': np.array([self.current_pitch, self.current_roll, self.current_yaw], dtype=np.float32)
            }
            return observation, -100.0, True, False, {'reset_reason': 'flip'}

        # Rest of your step implementation remains the same
        [...]  # (keeping existing step code)

    def reset(self, seed=None, options=None):
        """Reset the environment to its initial state"""
        super().reset(seed=seed)
        
        # Reset internal state
        self._step = 0
        self.total_steps = 0
        self.last_linear_velocity = 0.0
        self.steps_since_correction = self.cooldown_steps
        self.is_flipped = False
        
        # Reset simulation if robot is in bad state
        if self.check_flip_status():
            future = self.reset_simulation()
            while rclpy.ok():
                rclpy.spin_once(self.node)
                if future.done():
                    break
        
        # Ensure we get fresh sensor data after reset
        for _ in range(10):  # Spin a few times to get fresh data
            rclpy.spin_once(self.node, timeout_sec=0.1)
        
        observation = {
            'lidar': self.lidar_data,
            'odom': np.array(self.rover_position, dtype=np.float32),
            'imu': np.array([self.current_pitch, self.current_roll, self.current_yaw], dtype=np.float32)
        }
        
        return observation, {}

    # Rest of your class implementation remains the same
    [...]
