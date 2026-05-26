#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import sys
import csv

# from tf_transformations import euler_from_quaternion
from nav_msgs.msg import Odometry


def overwrite_csv(file_path, data_list):
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data_list)
       
class Odometer(Node):
    def __init__(self, test_name = "test"):
        super().__init__('odometer')
        self.sub = self.create_subscription(
            Odometry,
            "/base_pose_ground_truth",
            self.callback,
            10)

        self.test_name = f"{test_name}_traveled.csv"
        self.total_dist = 0
        self.prev_x = 0
        self.prev_y = 0
        self.first_run = True


    def callback(self, msg):

        if self.first_run:
            self.prev_x = float(msg.pose.pose.position.x)
            self.prev_y = float(msg.pose.pose.position.y)
            self.first_run = False

        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)

        movement = ((x - self.prev_x)**2 + (y - self.prev_y)**2)**0.5

        self.total_dist += movement
        print(self.total_dist)
        
        self.prev_x = float(msg.pose.pose.position.x)
        self.prev_y = float(msg.pose.pose.position.y)
        
        overwrite_csv(self.test_name, [self.total_dist])


if __name__ == '__main__':

    rclpy.init(args=sys.argv)
    if len(sys.argv) == 2:
        test_name = sys.argv[1]
    else:
        test_name = "test"
    odometer = Odometer(test_name=test_name)
    rclpy.spin(odometer)

    odometer.destroy_node()
    rclpy.shutdown()