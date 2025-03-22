import rclpy
import sys
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose, TransformStamped
import tf2_ros
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import threading
import subprocess

class PoseConverterNode(Node):
    def __init__(self, world_name, robot_name):
        super().__init__('pose_converter')
        
        self.world_name = world_name
        self.robot_name = robot_name
        
        # Initialize the pose lock
        self.pose_lock = threading.Lock()
        self.latest_poses = None
        
        # Configure QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Create publishers
        self.pose_array_pub = self.create_publisher(
            PoseArray,
            '/rover/pose_array',
            qos)
            
        # TF2 broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Start the pose processing with dynamic world name
        self.process = subprocess.Popen(
            ['ign', 'topic', '-e', '-t', f'/world/{self.world_name}/dynamic_pose/info'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Create timer to check subprocess output
        self.create_timer(0.1, self.process_ign_output)
        
        self.get_logger().info('Pose converter node initialized')
    
    def get_latest_poses(self):
        """Thread-safe method to get the latest poses"""
        with self.pose_lock:
            return self.latest_poses.copy() if self.latest_poses else None
            
    def process_ign_output(self):
        """Process Ignition output and publish only rover_zero4wd pose"""
        if self.process.poll() is not None:
            self.get_logger().error('Ignition topic process ended unexpectedly')
            return

        # Read line by line from subprocess output
        while True:
            line = self.process.stdout.readline()
            if not line:
                break
                
            # Parse the Ignition message
            if 'pose {' in line:
                try:
                    # Parse pose data
                    current_pose = Pose()
                    is_rover = False
                    
                    while True:
                        line = self.process.stdout.readline()
                        if not line or '}' in line:
                            break
                        
                        if f'name: "{self.robot_name}"' in line:
                            is_rover = True
                        elif 'position {' in line and is_rover:

                            # Read position
                            while True:
                                line = self.process.stdout.readline()
                                if '}' in line:
                                    break
                                if 'x:' in line:
                                    current_pose.position.x = float(line.split(':')[1])
                                elif 'y:' in line:
                                    current_pose.position.y = float(line.split(':')[1])
                                elif 'z:' in line:
                                    current_pose.position.z = float(line.split(':')[1])
                                    
                        elif 'orientation {' in line and is_rover:
                            # Read orientation
                            while True:
                                line = self.process.stdout.readline()
                                if '}' in line:
                                    break
                                if 'x:' in line:
                                    current_pose.orientation.x = float(line.split(':')[1])
                                elif 'y:' in line:
                                    current_pose.orientation.y = float(line.split(':')[1])
                                elif 'z:' in line:
                                    current_pose.orientation.z = float(line.split(':')[1])
                                elif 'w:' in line:
                                    current_pose.orientation.w = float(line.split(':')[1])

                    
                    # Only publish if we found rover_zero4wd
                    if is_rover:
                        pose_array = PoseArray()
                        pose_array.header.stamp = self.get_clock().now().to_msg()
                        pose_array.header.frame_id = 'world'
                        pose_array.poses.append(current_pose)
                        
                        # Update latest poses thread-safely
                        with self.pose_lock:
                            self.latest_poses = [current_pose]
                        
                        # Publish pose array
                        self.pose_array_pub.publish(pose_array)
                        
                        # Broadcast transform
                        t = TransformStamped()
                        t.header.stamp = self.get_clock().now().to_msg()
                        t.header.frame_id = 'world'
                        t.child_frame_id = self.robot_name
                        #t.child_frame_id = 'rover_zero4wd'
                        t.transform.translation.x = current_pose.position.x
                        t.transform.translation.y = current_pose.position.y
                        t.transform.translation.z = current_pose.position.z
                        t.transform.rotation = current_pose.orientation
                        self.tf_broadcaster.sendTransform(t)
                    
                except Exception as e:
                    self.get_logger().error(f'Error processing pose data: {str(e)}')
    
    def __del__(self):
        if hasattr(self, 'process'):
            self.process.terminate()

def main(args=None):
    if len(sys.argv) != 3:
        print("Usage: python3 ign_ros2_pose_topic.py <world_name> <robot_name>")
        sys.exit(1)
        
    world_name = sys.argv[1]
    robot_name = sys.argv[2]
    
    rclpy.init(args=args)
    node = PoseConverterNode(world_name, robot_name)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


def mainOld(args=None):
    rclpy.init(args=args)
    node = PoseConverterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
