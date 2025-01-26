import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json

class MinimalStateTester(Node):
    def __init__(self):
        super().__init__('minimal_state_tester')
        
        # Subscribe to world state
        self.subscription = self.create_subscription(
            String,  # We'll use String first to see the raw message
            '/world/default/state',
            self.state_callback,
            10)
        
        self.got_data = False
        
    def state_callback(self, msg):
        if not self.got_data:
            self.get_logger().info('Received world state message:')
            self.get_logger().info(str(msg.data)[:200] + '...')  # Print first 200 chars
            self.got_data = True

def main(args=None):
    rclpy.init(args=args)
    node = MinimalStateTester()
    rclpy.spin_once(node, timeout_sec=5.0)
    node.get_logger().info('Test complete')
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
