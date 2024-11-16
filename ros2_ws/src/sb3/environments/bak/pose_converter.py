#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose, TransformStamped
import tf2_ros
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import threading
import subprocess

class PoseConverterNode(Node):
    def __init__(self):
        super().__init__('pose_converter')
        
        # Add a thread-safe way to store the latest poses
        self.latest_poses = None
        self.pose_lock = threading.Lock()
        
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
        
        # Start the pose processing in a separate thread
        self.process = subprocess.Popen(
            ['ign', 'topic', '-e', '-t', '/world/maze/dynamic_pose/info'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Create timer to check subprocess output
        self.create_timer(0.1, self.process_ign_output)
        
    def get_latest_poses(self):
        """Thread-safe method to get the latest poses"""
        with self.pose_lock:
            return self.latest_poses.copy() if self.latest_poses else None
            
    def process_ign_output(self):
        """Process the output from Ignition"""
        if self.process.poll() is not None:
            self.get_logger().error('Ignition topic process ended unexpectedly')
            return
            
        while True:
            line = self.process.stdout.readline()
            if not line:
                break
                
            if 'pose {' in line:
                try:
                    poses = []
                    current_pose = Pose()
                    name = ''
                    
                    while True:
                        line = self.process.stdout.readline()
                        if not line or '}' in line:
                            break
                            
                        # Parse pose data (position and orientation)
                        # ... (rest of your existing pose parsing code)
                            
                    # Update latest poses thread-safely
                    with self.pose_lock:
                        self.latest_poses = poses
                        
                    # Publish pose array
                    pose_array = PoseArray()
                    pose_array.header.stamp = self.get_clock().now().to_msg()
                    pose_array.header.frame_id = 'world'
                    pose_array.poses = poses
                    self.pose_array_pub.publish(pose_array)
                    
                except Exception as e:
                    self.get_logger().error(f'Error processing pose data: {str(e)}')

    def __del__(self):
        if hasattr(self, 'process'):
            self.process.terminate()

