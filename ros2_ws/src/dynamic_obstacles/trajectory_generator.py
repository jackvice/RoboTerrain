import math
import json
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import GetEntityState



class TrajectoryGeneratorNode(Node):
    def __init__(self):
        super().__init__('trajectory_generator')
        # Create client to get the height from Gazebo
        ...
        # Create client to get the height from Gazebo
        self.client = self.create_client(GetEntityState, '/get_entity_state')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for Gazebo GetEntityState service...')
        self.req = GetEntityState.Request()

    def get_height_at_position(self, x, y):
        # Request entity state to get height at a given (x, y) location
        self.req.name = "ground_plane"  # Or any other suitable entity
        response = self.client.call(self.req)
        if response.success:
            return response.pose.position.z
        return 0.0

    def calculate_3d_waypoints(self, waypoints_2d, velocity):
        waypoints_3d = []
        time = 0
        for i, (x, y) in enumerate(waypoints_2d):
            z = self.get_height_at_position(x, y)
            if i > 0:
                # Calculate the 3D distance from the previous waypoint
                prev_x, prev_y, prev_z = waypoints_3d[-1]
                distance = math.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2 + (z - prev_z) ** 2)
                # Calculate the time needed to travel this distance at the desired velocity
                time += distance / velocity
            else:
                time = 0  # Start point, no time accumulation

            waypoints_3d.append((x, y, z, time))

        return waypoints_3d

    def generate_sdf_trajectory_block(self, waypoints_3d):
        trajectory_block = "<trajectory id=\"0\" type=\"walk\">\n"
        for wp in waypoints_3d:
            x, y, z, time = wp
            yaw = 0  # Default yaw for this example, can be calculated as needed
            trajectory_block += f"  <waypoint>\n    <time>{time:.2f}</time>\n    <pose>{x} {y} {z} 0 0 {yaw}</pose>\n  </waypoint>\n"
        trajectory_block += "</trajectory>"
        return trajectory_block

def main(args=None):
    rclpy.init(args=args)
    metrics_node = MetricsNode()

    # Provide the waypoints and velocity
    waypoints_2d = [(-2, -1), (-3, -1), (-3, 2), (-2, 2)]
    velocity = 0.5  # meters per second

    # Calculate the 3D waypoints with timing
    waypoints_3d = metrics_node.calculate_3d_waypoints(waypoints_2d, velocity)

    # Generate SDF trajectory block
    sdf_trajectory_block = metrics_node.generate_sdf_trajectory_block(waypoints_3d)
    print(sdf_trajectory_block)

    # Clean up and shutdown
    metrics_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
