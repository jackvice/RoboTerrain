#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, DurabilityPolicy, ReliabilityPolicy

# from tf_transformations import euler_from_quaternion
from nav_msgs.msg import OccupancyGrid
import sys


class MapRepublisher(Node):
    def __init__(self):
        super().__init__('people_distance_measure')

        self.subscription = self.create_subscription(
            OccupancyGrid,
            "/map",
            self.mapCallback,
            10)
        qos_profile = QoSProfile(
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_ALL,
            depth=1
        )
        self.done = False
        self.publisher_ = self.create_publisher(
            OccupancyGrid, '/map',  qos_profile=qos_profile)
        # self.create_timer(0.05, self.timer_callback)

    # def timer_callback(self):
    #     if self.map != None:
    #         self.publisher_.publish(self.map)

    def mapCallback(self, msg):
        if self.done == False:
            self.publisher_.publish(msg)
            self.done = True


if __name__ == '__main__':

    rclpy.init(args=sys.argv)

    map = MapRepublisher()
    rclpy.spin(map)

    map.destroy_node()
    rclpy.shutdown()
