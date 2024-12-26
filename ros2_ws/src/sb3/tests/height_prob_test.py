#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from ros_gz_interfaces.srv import SpawnEntity, DeleteEntity
from geometry_msgs.msg import PoseStamped
import time

class HeightProbeNode(Node):
    def __init__(self):
        super().__init__('height_probe')
        
        # Create service clients for spawning and deleting entities
        self.spawn_client = self.create_client(SpawnEntity, '/world/default/create')
        self.delete_client = self.create_client(DeleteEntity, '/world/default/remove')
        
        # Wait for services
        while not self.spawn_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for spawn service...')
        while not self.delete_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for delete service...')

        # Create pose subscriber
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/model/height_probe/pose',
            self.pose_callback,
            10)
        
        self.current_probe_id = None
        self.last_z = None
        self.stable_count = 0

    def spawn_probe(self, x: float, y: float) -> str:
        """Spawn a probe at the given x,y coordinates"""
        probe_sdf = f"""
        <?xml version="1.0" ?>
        <sdf version="1.6">
          <model name="height_probe">
            <pose>{x} {y} 20 0 0 0</pose>
            <link name="link">
              <collision name="collision">
                <geometry>
                  <sphere><radius>0.1</radius></sphere>
                </geometry>
              </collision>
              <visual name="visual">
                <geometry>
                  <sphere><radius>0.1</radius></sphere>
                </geometry>
              </visual>
              <gravity>1</gravity>
            </link>
          </model>
        </sdf>
        """
        
        req = SpawnEntity.Request()
        req.xml = probe_sdf
        req.name = "height_probe"
        
        future = self.spawn_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def delete_probe(self, name: str):
        """Delete the probe with the given name"""
        req = DeleteEntity.Request()
        req.name = name
        
        future = self.delete_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def pose_callback(self, msg: PoseStamped):
        """Monitor probe position for stability"""
        current_z = msg.pose.position.z
        
        if self.last_z is not None:
            if abs(current_z - self.last_z) < 0.001:  # 1mm threshold
                self.stable_count += 1
            else:
                self.stable_count = 0
                
        self.last_z = current_z
        
        # If position is stable for 10 callbacks
        if self.stable_count >= 10:
            self.get_logger().info(f'Height at point: {current_z}')
            self.delete_probe('height_probe')

def main(args=None):
    rclpy.init(args=args)
    node = HeightProbeNode()
    
    # Test with a single point
    node.spawn_probe(-15.0, -23.0)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        if node.current_probe_id:
            node.delete_probe('height_probe')
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
