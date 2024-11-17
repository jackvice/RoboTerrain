import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import math
import csv
import os
from datetime import datetime

class MetricsNode(Node):
    def __init__(self):
        super().__init__('metrics_node')
        
        # Debug mode flag and settings
        self.debug_mode = self.declare_parameter('debug_mode', False).value
        self.debug_interval = self.declare_parameter('debug_interval', 100).value
        self.update_count = 0
        
        # Initialize metrics with thresholds
        self.metrics = {
            'total_collisions': 0,
            'collision_threshold': 0.2,  # 20cm collision threshold
            'smoothness_metric': 0.0,
            'smoothness_threshold': 10.0,  # Threshold for "rough" movement
            'obstacle_clearance': float('inf'),
            'min_safe_clearance': 0.5,  # 50cm minimum safe clearance
            'total_distance': 0.0,
            'distance_threshold': 0.5,  # Maximum reasonable distance per update
            'velocity_over_rough': 0.0,
            'rough_terrain_threshold': 15.0  # m/s^2 acceleration threshold for rough terrain
        }
        
        # Buffers for metric calculation
        self.buffer = {
            'imu_readings': [],
            'clearance_readings': [],
            'velocity_readings': [],
            'buffer_size': 100  # Keep last 100 readings
        }
        
        # State tracking
        self.previous_position = None
        self.start_time = self.get_clock().now()
        self.last_log_time = self.start_time
        
        # Set up subscribers
        self.setup_subscribers()
        
        # Set up logging
        self.setup_logging()
        
        # Create timer for periodic metric logging
        self.create_timer(1.0, self.periodic_logging)  # Log every second

    def setup_subscribers(self):
        """Set up all subscribers with appropriate QoS profiles"""
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )
        
        self.imu_subscription = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.pose_array_subscriber = self.create_subscription(
            PoseArray,
            '/rover/pose_array',
            self.pose_array_callback,
            qos_profile
        )
        
        self.odometry_subscription = self.create_subscription(
            Odometry,
            '/odometry/wheels',
            self.odometry_callback,
            10
        )

    def setup_logging(self):
        """Set up CSV logging with timestamp and create directory if needed"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #log_dir = os.path.join(os.getcwd(), 'metric_logs')
        log_dir = os.path.join('/home/jack/src/RoboTerrain/metrics_analyzer/data', 'metric_logs')
        os.makedirs(log_dir, exist_ok=True)
        
        self.file_path = os.path.join(log_dir, f'metrics_log_{timestamp}.csv')
        self.create_csv_header()

    def create_csv_header(self):
        """Create CSV file with detailed headers"""
        with open(self.file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'Timestamp',
                'Total Collisions',
                'Current Collision Status',
                'Smoothness Metric',
                'Current Smoothness',
                'Obstacle Clearance',
                'Distance Traveled',
                'Current Velocity',
                'IMU Acceleration Magnitude',
                'Is Rough Terrain',
                'Notes'
            ])

    def lidar_callback(self, msg):
        """Process LiDAR data for collision detection and obstacle clearance"""
        # Convert ranges to numpy array and handle inf/nan
        ranges = np.array(msg.ranges)
        finite_ranges = ranges[np.isfinite(ranges)]
        
        if len(finite_ranges) > 0:
            # Update obstacle clearance
            current_clearance = np.min(finite_ranges)
            self.metrics['obstacle_clearance'] = current_clearance
            self.buffer['clearance_readings'].append(current_clearance)
            
            # Check for collisions
            collision_detected = current_clearance < self.metrics['collision_threshold']
            if collision_detected:
                self.metrics['total_collisions'] += 1
            
            # Buffer management
            if len(self.buffer['clearance_readings']) > self.buffer['buffer_size']:
                self.buffer['clearance_readings'].pop(0)
            
            if self.debug_mode and self.update_count % self.debug_interval == 0:
                self.get_logger().info(
                    f"LiDAR Update - Clearance: {current_clearance:.2f}m, "
                    f"Collisions: {self.metrics['total_collisions']}"
                )

    def imu_callback(self, msg):
        """Process IMU data for smoothness and terrain roughness detection"""
        # Calculate acceleration magnitude
        accel = msg.linear_acceleration
        accel_magnitude = math.sqrt(accel.x**2 + accel.y**2 + accel.z**2)
        
        # Update smoothness metric
        angular_vel = msg.angular_velocity
        angular_magnitude = math.sqrt(angular_vel.x**2 + angular_vel.y**2 + angular_vel.z**2)
        
        current_smoothness = accel_magnitude + angular_magnitude
        self.metrics['smoothness_metric'] += current_smoothness
        
        # Check for rough terrain
        is_rough = accel_magnitude > self.metrics['rough_terrain_threshold']
        
        # Store in buffer
        self.buffer['imu_readings'].append({
            'accel_magnitude': accel_magnitude,
            'angular_magnitude': angular_magnitude,
            'is_rough': is_rough
        })
        
        # Buffer management
        if len(self.buffer['imu_readings']) > self.buffer['buffer_size']:
            self.buffer['imu_readings'].pop(0)
            
        if self.debug_mode and self.update_count % self.debug_interval == 0:
            self.get_logger().info(
                f"IMU Update - Accel Mag: {accel_magnitude:.2f}, "
                f"Angular Mag: {angular_magnitude:.2f}, "
                f"Is Rough: {is_rough}"
            )

    def pose_array_callback(self, msg):
        """Process pose data for distance calculation"""
        if not msg.poses:
            return
            
        current_pose = msg.poses[0]
        current_position = np.array([
            current_pose.position.x,
            current_pose.position.y,
            current_pose.position.z
        ])
        
        if self.previous_position is not None:
            # Calculate distance
            distance = np.linalg.norm(current_position - self.previous_position)
            
            # Only update if the distance is reasonable
            if distance < self.metrics['distance_threshold']:
                self.metrics['total_distance'] += distance
                
                if self.debug_mode and self.update_count % self.debug_interval == 0:
                    self.get_logger().info(
                        f"Position Update - Distance: {distance:.2f}m, "
                        f"Total: {self.metrics['total_distance']:.2f}m"
                    )
        
        self.previous_position = current_position

    def odometry_callback(self, msg):
        """Process odometry data for velocity tracking"""
        velocity = msg.twist.twist.linear
        velocity_magnitude = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        self.buffer['velocity_readings'].append(velocity_magnitude)
        
        # Buffer management
        if len(self.buffer['velocity_readings']) > self.buffer['buffer_size']:
            self.buffer['velocity_readings'].pop(0)
        
        # Update velocity over rough terrain metric
        if self.buffer['imu_readings'] and self.buffer['imu_readings'][-1]['is_rough']:
            self.metrics['velocity_over_rough'] = velocity_magnitude

    def periodic_logging(self):
        """Log metrics periodically to CSV file"""
        current_time = self.get_clock().now()
        
        # Calculate current metrics
        current_smoothness = np.mean([reading['accel_magnitude'] + reading['angular_magnitude'] 
                                    for reading in self.buffer['imu_readings']]) if self.buffer['imu_readings'] else 0
        current_clearance = np.mean(self.buffer['clearance_readings']) if self.buffer['clearance_readings'] else float('inf')
        current_velocity = np.mean(self.buffer['velocity_readings']) if self.buffer['velocity_readings'] else 0
        
        # Get latest IMU reading
        latest_imu = self.buffer['imu_readings'][-1] if self.buffer['imu_readings'] else {'accel_magnitude': 0, 'is_rough': False}
        
        # Prepare notes
        notes = []
        """
        if current_clearance < self.metrics['min_safe_clearance']:
            notes.append("Low clearance")
        if latest_imu['is_rough']:
            notes.append("Rough terrain")
        if current_smoothness > self.metrics['smoothness_threshold']:
            notes.append("Rough movement")
        """ 
        # Log to CSV
        with open(self.file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                current_time.to_msg().sec,
                round(self.metrics['total_collisions'], 4),
                1 if current_clearance < self.metrics['collision_threshold'] else 0,
                round(self.metrics['smoothness_metric'], 4),
                round(current_smoothness, 4),
                round(current_clearance, 4),
                round(self.metrics['total_distance'], 4),
                round(current_velocity, 4),
                round(latest_imu['accel_magnitude'], 4),
                1 if latest_imu['is_rough'] else 0,
                "; ".join(notes) if notes else "Normal operation"
            ])
        
        self.update_count += 1

def main(args=None):
    rclpy.init(args=args)
    metrics_node = MetricsNode()
    
    try:
        rclpy.spin(metrics_node)
    except KeyboardInterrupt:
        pass
    finally:
        metrics_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
