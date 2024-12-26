import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import RayQuery  # Ensure this message type is available
import math

class TrajectoryGeneratorNode(Node):
    def __init__(self):
        super().__init__('trajectory_generator')

        # Create client for ray query service
        self.ray_client = self.create_client(RayQuery, '/ray_query')
        while not self.ray_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for Ray Query Service...')
        self.get_logger().info('Ray Query Service is available.')
        self.ray_req = RayQuery.Request()

    def get_height_at_position(self, x, y):
        """Query the height of terrain at (x, y) using a ray cast."""
        self.ray_req.start.x = x
        self.ray_req.start.y = y
        self.ray_req.start.z = 100.0  # High above the ground
        self.ray_req.end.x = x
        self.ray_req.end.y = y
        self.ray_req.end.z = -100.0  # Below the ground

        future = self.ray_client.call_async(self.ray_req)
        rclpy.spin_until_future_complete(self, future)

        if future.result():
            intersection = future.result().intersection
            self.get_logger().info(f"Height at ({x}, {y}): {intersection.z}")
            return intersection.z  # Return the z-value (height)
        else:
            self.get_logger().warn(f"Failed to get height at ({x}, {y})")
            return 0.0

    def calculate_3d_waypoints(self, waypoints_2d, velocity):
        """Generate 3D waypoints with height data and timing."""
        waypoints_3d = []
        time = 0.0

        for i, (x, y) in enumerate(waypoints_2d):
            z = self.get_height_at_position(x, y)
            if i > 0:
                # Calculate 3D distance from the previous waypoint
                prev_x, prev_y, prev_z = waypoints_3d[-1][:3]
                distance = math.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2 + (z - prev_z) ** 2)
                time += distance / velocity
            else:
                time = 0.0  # Start time at the first waypoint
            
            waypoints_3d.append((x, y, z, time))

        return waypoints_3d

    def generate_sdf_trajectory_block(self, waypoints_3d):
        """Generate an SDF trajectory block based on 3D waypoints."""
        trajectory_block = "<trajectory id=\"0\" type=\"walk\">\n"
        for wp in waypoints_3d:
            x, y, z, time = wp
            yaw = 0  # Default yaw for simplicity
            trajectory_block += (
                f"  <waypoint>\n"
                f"    <time>{time:.2f}</time>\n"
                f"    <pose>{x} {y} {z} 0 0 {yaw}</pose>\n"
                f"  </waypoint>\n"
            )
        trajectory_block += "</trajectory>"
        return trajectory_block

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryGeneratorNode()

    # Provide 2D waypoints and velocity
    waypoints_2d = [(-15, -23), (-28, -28), (-28, -17), (-23, -23)]  # Example coordinates
    velocity = 0.5  # meters per second

    # Calculate 3D waypoints with terrain-following height
    waypoints_3d = node.calculate_3d_waypoints(waypoints_2d, velocity)

    # Generate the SDF trajectory block
    sdf_trajectory_block = node.generate_sdf_trajectory_block(waypoints_3d)
    print("\nGenerated SDF Trajectory Block:")
    print(sdf_trajectory_block)

    # Clean up and shutdown
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
