#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from multi_person_tracker_interfaces.msg import People, Person

import sys
import re
import csv

# from tf_transformations import euler_from_quaternion
from nav_msgs.msg import Odometry

def append_to_csv(file_path, data_list):
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data_list)

class StagePeople(Node):
    def __init__(self, num_hum, test_name = "test"):
        super().__init__('people_distance_measure')
        self.num_hum = num_hum
        self.people = [0 for i in range(self.num_hum)]
        self.test_name = f"{test_name}_people_dist.csv"

        self.people_marker_sub = []
        for human_id in range(1, self.num_hum + 1):
            name = 'human'+str(human_id)
            subscription = self.create_subscription(
            Odometry,
            f"/{name}/base_pose_ground_truth",
            self.people_callback,
            10)

            subscription
            self.people_marker_sub.append(subscription)

        
        self.rob_sub = self.create_subscription(
            Odometry,
            f"/base_pose_ground_truth",
            self.robot_callback,
            10)

    def people_callback(self, msg):

        index = [int(num) for num in re.findall(r"\d+", msg.header.frame_id)]
        index = index[0] - 1

        person = Person()
        person.position.x = float(msg.pose.pose.position.x)
        person.position.y = float(msg.pose.pose.position.y)


        self.people[index] = person

    def robot_callback(self, msg):

        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)

        dists = []
        try:
            for person in self.people:
                dist = ((x - person.position.x)**2 + (y - person.position.y)**2)**0.5
                dists.append(dist)
            # print(dists)
            append_to_csv(self.test_name, dists)
        except Exception as e:
            # print(f"Exception: {e}")
            pass


if __name__ == '__main__':

    rclpy.init(args=sys.argv)
    nh = sys.argv[1]
    if len(sys.argv) == 3:
        test_name = sys.argv[2]
    else:
        test_name = "test"

    humans = StagePeople(num_hum=int(nh), test_name=test_name)
    rclpy.spin(humans)

    humans.destroy_node()
    rclpy.shutdown()