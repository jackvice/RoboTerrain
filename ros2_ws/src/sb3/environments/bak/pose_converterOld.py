#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose, TransformStamped
import tf2_ros
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import sys
import subprocess

class PoseConverterNode(Node):
    def __init__(self):
        super().__init__('pose_converter')
        
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
            
        # TF2 broadcaster for coordinate frames
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Start ign topic echo in a subprocess and process its output
        self.process = subprocess.Popen(
            ['ign', 'topic', '-e', '-t', '/world/maze/dynamic_pose/info'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Create timer to check subprocess output
        self.create_timer(0.1, self.process_ign_output)  # 10Hz
        
        self.get_logger().info('Pose converter node initialized')
        
    def process_ign_output(self):
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
                            
                        
                        if 'rover_zero4wd' in line:
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
                    
                    
                    
                    # Broadcast transform
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


                        t = TransformStamped()
                        t.header.stamp = self.get_clock().now().to_msg()
                        t.header.frame_id = 'world'
                        t.child_frame_id = 'rover_zero4wd'
                        t.transform.translation.x = current_pose.position.x
                        t.transform.translation.y = current_pose.position.y
                        t.transform.translation.z = current_pose.position.z
                        t.transform.rotation = current_pose.orientation
                        self.tf_broadcaster.sendTransform(t)
                    
                    # Publish pose array
                    self.pose_array_pub.publish(pose_array)
                    
                except Exception as e:
                    self.get_logger().error(f'Error processing pose data: {str(e)}')
        
    def __del__(self):
        if hasattr(self, 'process'):
            self.process.terminate()

def main(args=None):
    rclpy.init(args=args)
    node = PoseConverterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
