#!/usr/bin/env python3
import rclpy
import sys
from rclpy.node import Node

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs
from tf_transformations import quaternion_about_axis, euler_from_quaternion

import numpy as np
from scipy.ndimage import rotate
from multi_person_tracker_interfaces.msg import People
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from context_aware_navigation.asymetricGausian import *
from std_msgs.msg import Header

import cv2


class SocialMapGenerator(Node):

    def __init__(self, height, width, density, maxcost):
        super().__init__('social_map_generator')
        self.width = width
        self.height = height
        self.density = density  # px/m
        # Proxemic template half-size [m]; controls the canvas the asymmetric
        # Gaussian is rasterised onto.  3 m was the original outdoor value and
        # produced 6 m × 6 m blobs that easily overlapped the 15 × 15 m social
        # window's centre (= the robot itself).  1.5 m keeps the high-cost
        # core (~1 m forward / ~0.7 m side / ~0.5 m back) but trims the tail
        # so blobs no longer reach the robot from neighbouring tracklets.
        self.socialCostSize = 1.5
        self.maxcost = maxcost
        # %standard diviations %adjust to get different shapes
        self.sigmaFront = 2
        self.sigmaSide = 4/3
        self.sigmaBack = 1
        self.velocities = np.array([0])
        self.socialZones = initSocialZones(
            self.density, 2, self.velocities, self.maxcost, self.socialCostSize)  # 4/3, 1,

        self.center = ((self.width*self.density)/2,
                       (self.height*self.density)/2)
        self.socialMap = None

        self.publisher_ = self.create_publisher(Image, 'social_map', 10)
        self.cvBridge = CvBridge()
        self.people_sub = self.create_subscription(
            People,
            'people',
            self.people_callback,
            10)
        # tf listener stuff so we can transform people into there
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=2))
        self.tf_listener = TransformListener(self.tf_buffer, self,spin_thread=True)
        self.people_sub  # prevent unused variable warning

    def people_callback(self, msg: People):
        # Top-level guard: a single bad message must never crash the node
        # (ROS2's default executor will tear down rclpy.spin on uncaught
        # exceptions, taking /social_map permanently dark).
        try:
            self._people_callback_impl(msg)
        except Exception as ex:                           # noqa: BLE001
            self.get_logger().error(
                f'people_callback raised {type(ex).__name__}: {ex} '
                '(message dropped, node staying alive).',
                throttle_duration_sec=5.0)

    def _people_callback_impl(self, msg: People):
        # Build an empty social map up-front so that we can always publish
        # *something*, even when the /people message is empty or the TF lookup
        # fails.  The previous behaviour was to ``return`` on TF failure,
        # which silently took the /social_map topic dark and left Nav2's
        # SocialLayer with no input.
        self.socialMap = np.zeros(
            (round(self.height/self.density), round(self.width/self.density)),
            np.float32)
        self.center = (np.shape(self.socialMap)[0] / 2,
                       np.shape(self.socialMap)[1] / 2)

        # Look up the latest available transform from msg.header.frame_id
        # (typically ``odom``) to ``base_link``.  ``rclpy.time.Time()`` asks
        # for the latest available transform; the 200 ms timeout gives TF a
        # chance to catch up.
        map_to_base = None
        if msg.people:
            try:
                map_to_base = self.tf_buffer.lookup_transform(
                    "base_link",
                    msg.header.frame_id,
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=0, nanoseconds=200_000_000))
            except TransformException as ex:
                self.get_logger().warn(
                    f'Could not transform {msg.header.frame_id} -> base_link '
                    f'({ex}); publishing empty social map this cycle.',
                    throttle_duration_sec=2.0)
        else:
            self.get_logger().info(
                '/people message has no entries — publishing empty social map.',
                throttle_duration_sec=10.0)
        painted = 0
        people_iter = msg.people if map_to_base is not None else []
        for person in people_iter:
            # Reject person poses with NaN/Inf before they reach numpy so we
            # don't propagate invalid values into the TF matmul.  This was
            # observed when the upstream tracker produced a degenerate Kalman
            # estimate.
            if not (np.isfinite(person.position.x)
                    and np.isfinite(person.position.y)
                    and np.isfinite(person.position.z)):
                self.get_logger().warn(
                    f'Skipping person with non-finite position '
                    f'({person.position.x}, {person.position.y}, '
                    f'{person.position.z}).',
                    throttle_duration_sec=5.0)
                continue

            person_pose = PoseStamped()
            person_pose.header = msg.header
            person_pose.pose.position.x = person.position.x
            person_pose.pose.position.y = person.position.y
            person_pose.pose.position.z = 0.0
            person_pose.pose.orientation.w = 1.0

            try:
                person_base = tf2_geometry_msgs.do_transform_pose_stamped(
                    person_pose, map_to_base)
            except Exception as ex:                       # noqa: BLE001
                # Defensive: any tf2 internal failure must not bring down
                # the node.  Skip the person and keep going.
                self.get_logger().warn(
                    f'do_transform_pose_stamped failed for person at '
                    f'({person.position.x:.2f}, {person.position.y:.2f}): {ex}',
                    throttle_duration_sec=5.0)
                continue

            # The TF chain (e.g. odom->base_footprint from pose_topic) can
            # transiently publish NaN during recovery behaviours / Gazebo
            # state hiccups.  matmul silently propagates the NaN, which
            # then crashed the original code on int(np.floor(NaN)).  Guard
            # both inputs and outputs.
            px = float(person_base.pose.position.x)
            py = float(person_base.pose.position.y)
            if not (np.isfinite(px) and np.isfinite(py)):
                self.get_logger().warn(
                    f'TF returned NaN/Inf for person at '
                    f'({person.position.x:.2f}, {person.position.y:.2f}); '
                    'skipping (likely transient odom NaN).',
                    throttle_duration_sec=5.0)
                continue

            # Offset from robot (base_link origin) in metres -> pixels
            X = int(np.floor(px / self.density))
            Y = -int(np.floor(py / self.density))
            if abs(X) < self.center[0] and abs(Y) < self.center[1]:
                # transform relative to the top left corner of the map
                X = int(np.floor((X + self.center[0])))
                Y = int(np.floor((Y + self.center[1])))

                social_zone = rotate(
                    self.socialZones[0], np.rad2deg(person.position.z), reshape=True)

                (width, height) = np.shape(social_zone)
                width = int(np.floor(width/2))
                height = int(np.floor(height/2))

                minx = max(0, X-width)
                maxx = min(np.shape(self.socialMap)[0], X+width)
                miny = max(0, Y-height)
                maxy = min(np.shape(self.socialMap)[1], Y+width)
                roi = self.socialMap[miny:maxy, minx:maxx]

                sminx = width - min(width, X)
                sminy = height - min(height, Y)
                smaxx = width + min(width, np.shape(self.socialMap)[0]-X)
                smaxy = height + min(height, np.shape(self.socialMap)[1]-Y)

                social_zone = social_zone[sminy:smaxy, sminx:smaxx]
                self.socialMap[miny:maxy, minx:maxx] = np.maximum(
                    roi, social_zone)
                painted += 1
        if msg.people and painted == 0:
            self.get_logger().warn(
                f'{len(msg.people)} people received but none fell inside the '
                f'{self.width}×{self.height} m social window around the robot '
                '(they may be too far away or TF/map frame is wrong).',
                throttle_duration_sec=5.0)
        elif painted > 0 and float(np.max(self.socialMap)) <= 0.0:
            self.get_logger().warn(
                'People were placed on the social map but all costs are zero — '
                'check asymetricGausian / maxcost.',
                throttle_duration_sec=10.0)
        social_mapHeader = Header()
        social_mapHeader.frame_id = "base_link"
        social_mapHeader.stamp = msg.header.stamp
        self.publisher_.publish(self.cvBridge.cv2_to_imgmsg(
            self.socialMap, encoding="passthrough", header=social_mapHeader))


def main(args=sys.argv):
    rclpy.init(args=args)

    social_map_generator = SocialMapGenerator(15, 15, 0.05, int(args[1]))
    rclpy.spin(social_map_generator)
    social_map_generator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main(args=sys.argv)
