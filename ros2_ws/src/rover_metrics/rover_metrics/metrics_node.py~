import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu
import numpy as np
import math
import csv
import os


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import numpy as np
import math
import csv
import os


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose
from rclpy.action import ActionClient
import numpy as np
import math
import csv
import os

class MetricsNode(Node):
    """
    MetricsNode is responsible for monitoring and logging various performance metrics during robot
               navigation trials.
    
    The node subscribes to sensor topics such as LiDAR, IMU, and odometry to gather data about the
    robot's environment and movement. It also interacts with an experiment controller to start and
    stop trials, and computes the following metrics:
    
    - Total Collisions: Number of collisions based on LiDAR data.
    - Smoothness of Route: Calculated from IMU data to estimate how smooth the robot's movement is.
    - Obstacle Clearance: Minimum distance from obstacles based on LiDAR readings.
    - Mean Time to Traverse (MTT): Average time taken to traverse the environment.
    - Traverse Rate (TR): Percentage of the environment classified as navigable terrain.
    - Velocity Over Rough Terrain (VOR): Average speed when navigating rough terrain.
    - Optimal Path Length vs Actual Path Length: Comparison between the optimal path calculated
             using Nav2 and the path taken by the robot.
    
    Metrics are logged to a CSV file for each trial, providing detailed insights into the robot's
    performance.
    """
    
    def __init__(self):
        super().__init__('metrics_node')
        
        # Initialize metrics
        self.total_collisions = 0
        self.smoothness_metric = 0
        self.obstacle_clearance = 0
        self.collision_threshold = 0.2  # Collision if an obstacle is within 20 cm
        self.imu_data = None
        self.start_time = None
        self.stop_time = None
        self.trial_number = None
        self.start_position = None
        self.goal_position = None
        self.traverse_times = []
        self.traverse_rate = 0
        self.velocity_over_rough_terrain = 0
        self.total_distance = 0
        self.rough_terrain_velocity_sum = 0
        self.rough_terrain_samples = 0
        self.optimal_path_length = 0

        # Subscribe to topics
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
        self.odometry_subscription = self.create_subscription(
            Odometry,
            '/odometry/wheels',
            self.odometry_callback,
            10
        )
        self.experiment_controller_subscription = self.create_subscription(
            String,
            '/experiment_controller',
            self.experiment_controller_callback,
            10
        )

        # Set up CSV file for logging
        self.file_path = os.path.join(os.getcwd(), 'metrics_log.csv')
        with open(self.file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Trial Number', 'Timestamp', 'Total Collisions', 'Smoothness of Route',
                             'Obstacle Clearance','Mean Time to Traverse (s)', 'Traverse Rate (%)',
                             'Velocity Over Rough Terrain (m/s)', 'Optimal Path Length (m)',
                             'Actual Path Length (m)'])

        # Action client for path planning
        self.path_planner_client = ActionClient(self, ComputePathToPose, 'compute_path_to_pose')

    def lidar_callback(self, msg):
        # Total Collisions
        num_collisions = np.sum(np.array(msg.ranges) < self.collision_threshold)
        self.total_collisions += num_collisions

        # Obstacle Clearance (minimum distance to obstacle)
        self.obstacle_clearance = np.nanmin(msg.ranges)

        # Save metrics to file
        self.log_metrics()
        

    def imu_callback(self, msg):
        # Calculate Smoothness of Route based on linear acceleration and angular velocity
        linear_accel = msg.linear_acceleration
        angular_vel = msg.angular_velocity

        # Use acceleration and angular velocity magnitude for smoothness measure
        accel_magnitude = math.sqrt(linear_accel.x**2 + linear_accel.y**2 + linear_accel.z**2)
        angular_vel_magnitude = math.sqrt(angular_vel.x**2 + angular_vel.y**2 + angular_vel.z**2)

        # Update smoothness metric (lower is smoother)
        self.smoothness_metric += abs(accel_magnitude) + abs(angular_vel_magnitude)

        
    def odometry_callback(self, msg):
        # Calculate distance traveled and velocity over rough terrain
        linear_velocity = msg.twist.twist.linear
        velocity_magnitude = math.sqrt(linear_velocity.x**2 + linear_velocity.y**2 + linear_velocity.z**2)
        self.total_distance += velocity_magnitude * (1 / 10)  # Assuming 10 Hz callback rate

        # Check if the terrain is rough (using IMU data or some predefined criteria)
        if self.imu_data:
            accel_magnitude = math.sqrt(self.imu_data.linear_acceleration.x**2 +
                                        self.imu_data.linear_acceleration.y**2 +
                                        self.imu_data.linear_acceleration.z**2)
            if accel_magnitude > 1.5:  # Example threshold for rough terrain
                self.rough_terrain_velocity_sum += velocity_magnitude
                self.rough_terrain_samples += 1

                
    def experiment_controller_callback(self, msg):
        """
        Callback function for experiment controller commands. Starts and stops trials based on incoming
        messages.
        
        - Parses commands for 'START' and 'STOP'.
        - Sets start time, trial number, start, and goal positions at trial start.
        - Calculates metrics and logs them upon receiving a stop command.
        """
        data = msg.data.split(',')
        command = data[0].strip()

        if command == 'START':
            self.start_time = self.get_clock().now()
            self.trial_number = int(data[1].strip())
            self.start_position = (float(data[2].strip()), float(data[3].strip()))
            self.goal_position = (float(data[4].strip()), float(data[5].strip()))
            self.calculate_optimal_path()
        elif command == 'STOP':
            self.stop_time = self.get_clock().now()
            self.calculate_metrics()
            

    def calculate_optimal_path(self):
        """
        Calculates the optimal path from the start to the goal position using a path planning action.
        
        - Sets up start and goal poses based on controller input.
        - Requests path planning from Nav2's ComputePathToPose action and handles the result asynchronously.
        """
        start_pose = PoseStamped()
        start_pose.header.frame_id = 'map'
        start_pose.pose.position.x = self.start_position[0]
        start_pose.pose.position.y = self.start_position[1]

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.pose.position.x = self.goal_position[0]
        goal_pose.pose.position.y = self.goal_position[1]

        # Send request to ComputePathToPose action in Nav2
        if not self.path_planner_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Path planner action server not available!')
            return

        goal_msg = ComputePathToPose.Goal()
        goal_msg.start = start_pose
        goal_msg.goal = goal_pose

        self.path_planner_client.send_goal_async(goal_msg).add_done_callback(self.handle_path_response)

        
    def handle_path_response(self, future):
        """
        Handles the response from the path planning action.
        
        - Checks if the path planning goal was accepted.
        - Initiates the retrieval of the planning result if accepted.
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Path planning goal rejected!')
            return

        goal_handle.get_result_async().add_done_callback(self.handle_path_result)

        
    def handle_path_result(self, future):
        """
        Handles the path planning result and calculates the optimal path length.
        
        - Retrieves the planned path and calculates its length using a helper function.
        - Stores the calculated path length for metric logging.
        """
        result = future.result().result
        if result:
            self.optimal_path_length = self.calculate_path_length(result.path)



    def calculate_3d_path_length(path):
        """
        Calculates the 3D path length for a given set of waypoints, taking elevation into account.
        
        - Iterates through waypoints and calculates the Euclidean distance in 3D space between consecutive points.
        - Returns the total path length.
        """
        total_length = 0.0
        for i in range(1, len(path.poses)):
            x1, y1, z1 = (path.poses[i-1].pose.position.x, 
                          path.poses[i-1].pose.position.y, 
                          path.poses[i-1].pose.position.z)
            x2, y2, z2 = (path.poses[i].pose.position.x, 
                          path.poses[i].pose.position.y, 
                          path.poses[i].pose.position.z)
            total_length += math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        return total_length

            
    def calculate_path_length(self, path):
        """
        Calculates the 2D path length (ignoring elevation) for a given set of waypoints.
        
        - Computes the Euclidean distance between consecutive points in the X-Y plane.
        - Returns the total path length.
        """
        total_length = 0.0
        for i in range(1, len(path.poses)):
            x1, y1 = path.poses[i-1].pose.position.x, path.poses[i-1].pose.position.y
            x2, y2 = path.poses[i].pose.position.x, path.poses[i].pose.position.y
            total_length += math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return total_length

    
    def calculate_metrics(self):
        """
        Calculates key metrics for the trial, such as Mean Time to Traverse, Traverse Rate, and
                        Velocity over Rough Terrain.
        
        - Computes the mean time to traverse based on recorded times.
        - Calculates the traverse rate as a percentage of environment size covered.
        - Determines the average velocity over rough terrain samples.
        - Logs all calculated metrics.
        """
        time_diff = (self.stop_time - self.start_time).nanoseconds * 1e-9
        self.traverse_times.append(time_diff)
        mean_time_to_traverse = sum(self.traverse_times) / len(self.traverse_times)

        # Calculate Traverse Rate (TR)
        self.traverse_rate = (self.total_distance / self.calculate_total_environment_size()) * 100  # Example

        # Calculate Velocity Over Rough Terrain (VOR)
        if self.rough_terrain_samples > 0:
            self.velocity_over_rough_terrain = self.rough_terrain_velocity_sum / self.rough_terrain_samples

        # Log metrics
        self.log_metrics(mean_time_to_traverse, self.traverse_rate, self.velocity_over_rough_terrain,
                         self.optimal_path_length, self.total_distance)

    def calculate_total_environment_size(self):
        # Placeholder for environment size calculation
        return 100.0  # Example value

    
    def log_metrics(self, mtt=None, tr=None, vor=None, optimal_path_length=None, actual_path_length=None):
        """
        Logs key performance metrics to a CSV file for analysis.
        
        - Records trial number, timestamp, collision count, smoothness metric, obstacle clearance,
                                      and other metrics.
        - Logs to a file defined by `self.file_path`.
        """
        timestamp = self.get_clock().now().to_msg()
        with open(self.file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                self.trial_number if self.trial_number else 'N/A',
                f"{timestamp.sec}.{timestamp.nanosec}",
                self.total_collisions,
                self.smoothness_metric,
                self.obstacle_clearance,
                mtt if mtt else 'N/A',
                tr if tr else 'N/A',
                vor if vor else 'N/A',
                optimal_path_length if optimal_path_length else 'N/A',
                actual_path_length if actual_path_length else 'N/A'
            ])

def main(args=None):
    rclpy.init(args=args)
    metrics_node = MetricsNode()
    rclpy.spin(metrics_node)

    # Destroy the node explicitly (optional)
    metrics_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

exit()

class MetricsNodeNewOld(Node):
    def __init__(self):
        super().__init__('metrics_node')
        
        # Initialize metrics
        self.total_collisions = 0
        self.smoothness_metric = 0
        self.obstacle_clearance = 0
        self.collision_threshold = 0.2  # Collision if an obstacle is within 20 cm
        self.imu_data = None
        self.start_time = None
        self.stop_time = None
        self.trial_number = None
        self.start_position = None
        self.goal_position = None
        self.traverse_times = []
        self.traverse_rate = 0
        self.velocity_over_rough_terrain = 0
        self.total_distance = 0
        self.rough_terrain_velocity_sum = 0
        self.rough_terrain_samples = 0

        # Subscribe to topics
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
        self.odometry_subscription = self.create_subscription(
            Odometry,
            '/odometry/wheels',
            self.odometry_callback,
            10
        )
        self.experiment_controller_subscription = self.create_subscription(
            String,
            '/experiment_controller',
            self.experiment_controller_callback,
            10
        )

        # Set up CSV file for logging
        self.file_path = os.path.join(os.getcwd(), 'metrics_log.csv')
        with open(self.file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Trial Number', 'Timestamp', 'Total Collisions', 'Smoothness of Route', 'Obstacle Clearance',
                             'Mean Time to Traverse (s)', 'Traverse Rate (%)', 'Velocity Over Rough Terrain (m/s)'])

    def lidar_callback(self, msg):
        # Total Collisions
        num_collisions = np.sum(np.array(msg.ranges) < self.collision_threshold)
        self.total_collisions += num_collisions

        # Obstacle Clearance (minimum distance to obstacle)
        self.obstacle_clearance = np.nanmin(msg.ranges)

        # Save metrics to file
        self.log_metrics()

    def imu_callback(self, msg):
        # Calculate Smoothness of Route based on linear acceleration and angular velocity
        linear_accel = msg.linear_acceleration
        angular_vel = msg.angular_velocity

        # Use acceleration and angular velocity magnitude for smoothness measure
        accel_magnitude = math.sqrt(linear_accel.x**2 + linear_accel.y**2 + linear_accel.z**2)
        angular_vel_magnitude = math.sqrt(angular_vel.x**2 + angular_vel.y**2 + angular_vel.z**2)

        # Update smoothness metric (lower is smoother)
        self.smoothness_metric += abs(accel_magnitude) + abs(angular_vel_magnitude)

    def odometry_callback(self, msg):
        # Calculate distance traveled and velocity over rough terrain
        linear_velocity = msg.twist.twist.linear
        velocity_magnitude = math.sqrt(linear_velocity.x**2 + linear_velocity.y**2 + linear_velocity.z**2)
        self.total_distance += velocity_magnitude * (1 / 10)  # Assuming 10 Hz callback rate

        # Check if the terrain is rough (using IMU data or some predefined criteria)
        if self.imu_data:
            accel_magnitude = math.sqrt(self.imu_data.linear_acceleration.x**2 +
                                        self.imu_data.linear_acceleration.y**2 +
                                        self.imu_data.linear_acceleration.z**2)
            if accel_magnitude > 1.5:  # Example threshold for rough terrain
                self.rough_terrain_velocity_sum += velocity_magnitude
                self.rough_terrain_samples += 1

    def experiment_controller_callback(self, msg):
        # Parse experiment controller message
        data = msg.data.split(',')
        command = data[0].strip()

        if command == 'START':
            self.start_time = self.get_clock().now()
            self.trial_number = int(data[1].strip())
            self.start_position = (float(data[2].strip()), float(data[3].strip()))
            self.goal_position = (float(data[4].strip()), float(data[5].strip()))
        elif command == 'STOP':
            self.stop_time = self.get_clock().now()
            self.calculate_metrics()

    def calculate_metrics(self):
        # Calculate Mean Time to Traverse (MTT)
        time_diff = (self.stop_time - self.start_time).nanoseconds * 1e-9
        self.traverse_times.append(time_diff)
        mean_time_to_traverse = sum(self.traverse_times) / len(self.traverse_times)

        # Calculate Traverse Rate (TR)
        self.traverse_rate = (self.total_distance / self.calculate_total_environment_size()) * 100  # Example

        # Calculate Velocity Over Rough Terrain (VOR)
        if self.rough_terrain_samples > 0:
            self.velocity_over_rough_terrain = self.rough_terrain_velocity_sum / self.rough_terrain_samples

        # Log metrics
        self.log_metrics(mean_time_to_traverse, self.traverse_rate, self.velocity_over_rough_terrain)

    def calculate_total_environment_size(self):
        # Placeholder for environment size calculation
        return 100.0  # Example value

    def log_metrics(self, mtt=None, tr=None, vor=None):
        # Log the current metrics to a CSV file
        timestamp = self.get_clock().now().to_msg()
        with open(self.file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                self.trial_number if self.trial_number else 'N/A',
                f"{timestamp.sec}.{timestamp.nanosec}",
                self.total_collisions,
                self.smoothness_metric,
                self.obstacle_clearance,
                mtt if mtt else 'N/A',
                tr if tr else 'N/A',
                vor if vor else 'N/A'
            ])

def main(args=None):
    rclpy.init(args=args)
    metrics_node = MetricsNode()
    rclpy.spin(metrics_node)

    # Destroy the node explicitly (optional)
    metrics_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




class MetricsNodeOld(Node):
    def __init__(self):
        super().__init__('metrics_node')
        
        # Initialize metrics
        self.total_collisions = 0
        self.smoothness_metric = 0
        self.obstacle_clearance = 0
        self.collision_threshold = 0.2  # Collision if an obstacle is within 20 cm
        self.imu_data = None

        # Subscribe to topics
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

        # Set up CSV file for logging
        self.file_path = os.path.join(os.getcwd(), 'metrics_log.csv')
        with open(self.file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Total Collisions', 'Smoothness of Route', 'Obstacle Clearance'])

    def lidar_callback(self, msg):
        # Total Collisions
        num_collisions = np.sum(np.array(msg.ranges) < self.collision_threshold)
        self.total_collisions += num_collisions

        # Obstacle Clearance (minimum distance to obstacle)
        self.obstacle_clearance = np.nanmin(msg.ranges)

        # Save metrics to file
        self.log_metrics()

    def imu_callback(self, msg):
        # Calculate Smoothness of Route based on linear acceleration and angular velocity
        linear_accel = msg.linear_acceleration
        angular_vel = msg.angular_velocity

        # Use acceleration and angular velocity magnitude for smoothness measure
        accel_magnitude = math.sqrt(linear_accel.x**2 + linear_accel.y**2 + linear_accel.z**2)
        angular_vel_magnitude = math.sqrt(angular_vel.x**2 + angular_vel.y**2 + angular_vel.z**2)

        # Update smoothness metric (lower is smoother)
        self.smoothness_metric += abs(accel_magnitude) + abs(angular_vel_magnitude)

    def log_metrics(self):
        # Log the current metrics to a CSV file
        timestamp = self.get_clock().now().to_msg()
        with open(self.file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                f"{timestamp.sec}.{timestamp.nanosec}",
                self.total_collisions,
                self.smoothness_metric,
                self.obstacle_clearance
            ])

def main(args=None):
    rclpy.init(args=args)
    metrics_node = MetricsNode()
    rclpy.spin(metrics_node)

    # Destroy the node explicitly (optional)
    metrics_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
