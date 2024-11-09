import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState

class HumanMover(Node):
    def __init__(self):
        super().__init__('human_mover')
        self.client = self.create_client(SetEntityState, '/world/maze/set_entity_state')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = SetEntityState.Request()
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.t = 0.0

    def timer_callback(self):
        x = 5.0 * math.sin(self.t)
        self.t += 0.05

        pose = Pose()
        pose.position.x = x
        pose.position.y = 0.0
        pose.position.z = 1.0
        pose.orientation.w = 1.0

        entity_state = EntityState()
        entity_state.name = 'static_human'
        entity_state.pose = pose

        self.req.state = entity_state
        self.future = self.client.call_async(self.req)

def main(args=None):
    rclpy.init(args=args)
    human_mover = HumanMover()
    rclpy.spin(human_mover)
    human_mover.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
