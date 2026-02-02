import rclpy
import sys
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import PoseArray, Pose, TransformStamped
import tf2_ros
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import threading
import subprocess
from nav_msgs.msg import Odometry
import queue

class PoseConverterNode(Node):
    def __init__(self, world_name, robot_name):
        # Force use_sim_time to True so our TF timestamps match Gazebo
        super().__init__('pose_converter', 
                         parameter_overrides=[
                             Parameter('use_sim_time', Parameter.Type.BOOL, True)
                         ])
        
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

        self.odom_pub = self.create_publisher(
            Odometry,
            '/odom_ground_truth',
            10)
        
        # TF2 broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Queue for thread-safe communication
        self.pose_queue = queue.Queue(maxsize=100)
        
        # Start subprocess reader in separate thread
        self.process = subprocess.Popen(
            ['ign', 'topic', '-e', '-t', f'/world/{self.world_name}/dynamic_pose/info'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
        )
        self.reader_thread = threading.Thread(target=self._read_subprocess, daemon=True)
        self.reader_thread.start()
        
        # Timer just publishes from queue - never blocks
        self.create_timer(0.02, self._publish_from_queue)

        
        self.get_logger().info(f'Pose converter initialized for {robot_name} using Sim Time')

    def _read_subprocess(self) -> None:
        """Background thread: blocking reads from subprocess, parses, queues poses."""
        current_pose = None
        is_rover = False
        parse_state = None  # None, 'position', 'orientation'
        
        while True:
            line = self.process.stdout.readline()
            if not line:
                if self.process.poll() is not None:
                    self.get_logger().error('Subprocess exited')
                    break
                continue
            
            line = line.strip()
            
            if f'name: "{self.robot_name}"' in line:
                is_rover = True
                current_pose = Pose()
            elif 'name:' in line and 'name: "' in line:
                # Different entity - reset
                if is_rover and current_pose is not None:
                    try:
                        self.pose_queue.put_nowait(current_pose)
                    except queue.Full:
                        pass  # Drop if queue full
                is_rover = False
                current_pose = None
                parse_state = None
            elif is_rover and current_pose is not None:
                if 'position {' in line:
                    parse_state = 'position'
                elif 'orientation {' in line:
                    parse_state = 'orientation'
                elif line == '}':
                    parse_state = None
                elif parse_state == 'position':
                    if 'x:' in line:
                        current_pose.position.x = float(line.split(':')[1])
                    elif 'y:' in line:
                        current_pose.position.y = float(line.split(':')[1])
                    elif 'z:' in line:
                        current_pose.position.z = float(line.split(':')[1])
                elif parse_state == 'orientation':
                    if 'x:' in line:
                        current_pose.orientation.x = float(line.split(':')[1])
                    elif 'y:' in line:
                        current_pose.orientation.y = float(line.split(':')[1])
                    elif 'z:' in line:
                        current_pose.orientation.z = float(line.split(':')[1])
                    elif 'w:' in line:
                        current_pose.orientation.w = float(line.split(':')[1])


    def _publish_from_queue(self) -> None:
        """Timer callback: non-blocking publish newest pose (drain queue)."""
        current_pose = None
        while True:
            try:
                current_pose = self.pose_queue.get_nowait()
            except queue.Empty:
                break

        if current_pose is None:
            return

        now = self.get_clock().now().to_msg()

        # 1. Publish Pose Array
        pose_array = PoseArray()
        pose_array.header.stamp = now
        pose_array.header.frame_id = 'world'
        pose_array.poses.append(current_pose)
        self.pose_array_pub.publish(pose_array)
        
        # 2. Publish Odometry
        odom_msg = Odometry()
        odom_msg.header.stamp = now
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_footprint'
        odom_msg.pose.pose = current_pose
        odom_msg.pose.covariance = [0.01 if i in [0, 7, 14, 21, 28, 35] else 0.0 for i in range(36)]
        self.odom_pub.publish(odom_msg)
        
        # 3. Broadcast TF
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_footprint'
        t.transform.translation.x = current_pose.position.x
        t.transform.translation.y = current_pose.position.y
        t.transform.translation.z = current_pose.position.z
        t.transform.rotation = current_pose.orientation
        self.tf_broadcaster.sendTransform(t)


                        
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
