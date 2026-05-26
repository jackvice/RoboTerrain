
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav2_msgs.action import ComputePathToPose
from rclpy.action import ActionClient
import numpy as np
import math
import csv
import os


class MetricsNode(Node):
    """ ros2 run rover_metrics metrics_node
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

        # Metrics Data
        self.current_position = None

        self.previous_position = None
        self.total_distance = 0.0
        self.distance_threshold = 0.5  # Maximum reasonable distance per update

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

        self.experiment_controller_subscription = self.create_subscription(
            String,
            '/experiment_controller',
            self.experiment_controller_callback,
            10
        )

        self.odometry_subscription = self.create_subscription(
            Odometry,
            '/odometry/wheels',
            self.odometry_callback,
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

        self.trial_configs = {
            'navigation': {
                'targets': [(-2,6), (-4,3), (-2,-3)],  # Your existing targets
                'success_distance': 0.5,
                'max_time': 300,  # 5 minutes per trial
                'collision_threshold': 0.2
            },
            'wall_following': {
                'desired_distance': 0.5,
                'max_time': 180,  # 3 minutes per trial
                'path': [(0,0), (-5,0), (-5,5), (0,5), (0,0)],  # Example wall-following path
                'success_criteria': 'complete_loop'
            },
            'rough_terrain': {
                'zones': [(-3,2,1), (-1,-1,2)],  # (x,y,difficulty) for rough areas
                'max_time': 240,  # 4 minutes per trial
                'success_criteria': 'traverse_all_zones'
            }
        }
        
        self.current_trial = None
        self.trial_results = []
        self.current_config = None
        self.trial_timer = None
        
        # Add timer for periodic trial status checking
        self.status_timer = self.create_timer(1.0, self.check_trial_status)

    def start_trial(self, trial_type, config_name):
        """Start a new trial with specified configuration."""
        if self.current_trial is not None:
            self.get_logger().warn('Trial already in progress!')
            return False
            
        if trial_type not in self.trial_configs:
            self.get_logger().error(f'Unknown trial type: {trial_type}')
            return False
            
        self.current_config = self.trial_configs[trial_type]
        self.current_trial = {
            'type': trial_type,
            'config_name': config_name,
            'start_time': self.get_clock().now(),
            'start_position': self.current_position,
            'collisions': 0,
            'distance_traveled': 0.0,
            'goals_reached': 0,
            'status': 'in_progress'
        }
        
        # Start trial timer
        self.trial_timer = self.create_timer(
            self.current_config['max_time'],
            self.timeout_trial
        )
        
        self.get_logger().info(f'Started {trial_type} trial: {config_name}')
        return True

    def stop_trial(self, status='completed'):
        """Stop current trial and record results."""
        if self.current_trial is None:
            return
            
        # Cancel timer if it exists
        if self.trial_timer:
            self.trial_timer.cancel()
            self.trial_timer = None
            
        # Record end time and calculate duration
        end_time = self.get_clock().now()
        duration = (end_time - self.current_trial['start_time']).nanoseconds / 1e9
        
        # Compile trial results
        trial_result = {
            **self.current_trial,
            'end_time': end_time,
            'duration': duration,
            'final_position': self.current_position,
            'status': status,
            'metrics': {
                'total_distance': self.total_distance,
                'smoothness': self.smoothness_metric,
                'collisions': self.total_collisions,
                'min_clearance': self.obstacle_clearance
            }
        }
        
        self.trial_results.append(trial_result)
        self.save_trial_results(trial_result)
        
        # Reset current trial
        self.current_trial = None
        self.reset_trial_metrics()
        
        self.get_logger().info(f'Trial completed with status: {status}')

    def timeout_trial(self):
        """Handle trial timeout."""
        self.get_logger().warn('Trial timed out!')
        self.stop_trial(status='timeout')

    def check_trial_status(self):
        """Periodically check trial status and success conditions."""
        if not self.current_trial or not self.current_position:
            return
            
        trial_type = self.current_trial['type']
        
        if trial_type == 'navigation':
            self.check_navigation_success()
        elif trial_type == 'wall_following':
            self.check_wall_following_success()
        elif trial_type == 'rough_terrain':
            self.check_rough_terrain_success()

    def check_navigation_success(self):
        """Check if navigation goals have been reached."""
        if not self.current_trial:
            return
            
        current_target = self.trial_configs['navigation']['targets'][self.current_target_idx]
        distance_to_target = math.sqrt(
            (self.current_position[0] - current_target[0])**2 +
            (self.current_position[1] - current_target[1])**2
        )
        
        if distance_to_target < self.trial_configs['navigation']['success_distance']:
            self.current_trial['goals_reached'] += 1
            self.current_target_idx = (self.current_target_idx + 1) % len(self.trial_configs['navigation']['targets'])
            
            if self.current_target_idx == 0:  # Completed all targets
                self.stop_trial(status='success')

    def save_trial_results(self, result):
        """Save trial results to CSV file."""
        timestamp = result['end_time'].to_msg()
        filename = f'trial_results_{result["type"]}_{result["config_name"]}.csv'
        
        # Ensure directory exists
        os.makedirs('trial_results', exist_ok=True)
        filepath = os.path.join('trial_results', filename)
        
        # Write results
        with open(filepath, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            # Write header if file is new
            if file.tell() == 0:
                writer.writerow([
                    'Timestamp', 'Trial Type', 'Config', 'Status', 'Duration',
                    'Distance', 'Collisions', 'Goals Reached', 'Min Clearance',
                    'Smoothness'
                ])
            
            writer.writerow([
                f"{timestamp.sec}.{timestamp.nanosec}",
                result['type'],
                result['config_name'],
                result['status'],
                result['duration'],
                result['metrics']['total_distance'],
                result['metrics']['collisions'],
                result['goals_reached'],
                result['metrics']['min_clearance'],
                result['metrics']['smoothness']
            ])

    def load_trial_config(self, config_file):
        """Load trial configurations from file."""
        try:
            with open(config_file, 'r') as f:
                new_configs = yaml.safe_load(f)
                self.trial_configs.update(new_configs)
                self.get_logger().info(f'Loaded configurations from {config_file}')
        except Exception as e:
            self.get_logger().error(f'Error loading config file: {e}')

    def reset_trial_metrics(self):
        """Reset metrics at the start of each trial"""
        self.total_distance = 0.0
        self.previous_position = None
        # Reset other metrics as needed
        
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

    def pose_array_callback(self, msg):
        """Callback for processing pose array messages and calculating distance traveled"""
        if msg.poses:  # Check if we have any poses
            current_pose = msg.poses[0]  # Take the first pose
        
            # Store current position
            current_position = np.array([
                current_pose.position.x,
                current_pose.position.y,
                current_pose.position.z
            ], dtype=np.float32)
        
            # If we have a previous position, calculate distance
            if hasattr(self, 'previous_position'):
                # Calculate Euclidean distance between positions
                distance = np.linalg.norm(current_position - self.previous_position)
                # Add to total distance only if it's a reasonable value (filter out jumps/errors)
                if distance < 0.5:  # Max reasonable distance per update (you may need to tune this)
                    self.total_distance += distance
                
                if self.start_time and not self.stop_time:  # Only log during active trials
                    print(f"Distance traveled: {self.total_distance:.2f}m")
        
            # Update previous position
            self.previous_position = current_position
        

    def odometry_callback(self, msg):
        # Calculate distance traveled and velocity over rough terrain
        linear_velocity = msg.twist.twist.linear
        velocity_magnitude = math.sqrt(linear_velocity.x**2 + linear_velocity.y**2 + linear_velocity.z**2)
        #self.total_distance += velocity_magnitude * (1 / 10)  # Assuming 10 Hz callback rate

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
            self.reset_trial_metrics()  # Reset metrics for new trial
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


    """    
    def pose_callback(self, msg):
        # Process ground truth position from Gazebo
        for pose in msg.pose:
            if pose.name == "rover_zero4wd":
                self.previous_position = self.current_position
                self.current_position = (pose.position.x, pose.position.y, pose.position.z)
                
                if self.previous_position is not None:
                    distance = math.sqrt(
                        (self.current_position[0] - self.previous_position[0])**2 +
                        (self.current_position[1] - self.previous_position[1])**2 +
                        (self.current_position[2] - self.previous_position[2])**2
                    )
                    self.total_distance_traveled += distance
                break
    """


