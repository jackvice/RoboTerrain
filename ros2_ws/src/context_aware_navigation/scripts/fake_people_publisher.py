#!/usr/bin/env python3
"""Publish a static set of people on ``/people`` for Nav2CAN sim testing.

The real pipeline gets ``People`` messages from ``multi_person_tracker`` which
in turn consumes an RGB-D camera.  In Gazebo we usually don't have one, so
this node simply broadcasts a hard-coded set of people (positions + facing
direction) at a fixed rate, allowing the social-map generator and Nav2's
SocialLayer / InteractionLayer to react as if real detections were arriving.

Usage:
    ros2 run context_aware_navigation fake_people_publisher.py \\
        --ros-args -p people:='[1.0, 0.5, 0.0, -1.0, 0.5, 3.14]'

The ``people`` parameter is a flat ``[x1, y1, theta1, x2, y2, theta2, ...]``
list in the ``frame_id`` frame (default ``map``).  If unset, defaults to two
people facing each other near the origin.
"""

from __future__ import annotations

import ast
import math
import sys
import traceback

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

from multi_person_tracker_interfaces.msg import People, Person
from std_msgs.msg import ColorRGBA

# Two people facing each other ahead of the TB3 spawn (~-2, -0.5) in map frame.
DEFAULT_PEOPLE = [2.0, 1.2, math.pi, 2.0, -1.2, 0.0]
# Launch files pass this as a string; must match PARAMETER_STRING on declare.
DEFAULT_PEOPLE_STR = "[2.0, 1.2, 3.141592653589793, 2.0, -1.2, 0.0]"

try:
    from visualization_msgs.msg import Marker, MarkerArray
    _HAS_MARKERS = True
except ImportError:
    Marker = None  # type: ignore[misc, assignment]
    MarkerArray = None  # type: ignore[misc, assignment]
    _HAS_MARKERS = False


class FakePeoplePublisher(Node):
    def __init__(self):
        super().__init__("fake_people_publisher")

        self.declare_parameter(
            "frame_id", "map",
            ParameterDescriptor(description="TF frame the people poses are in"))
        self.declare_parameter(
            "rate_hz", 10.0,
            ParameterDescriptor(description="Publish rate"))
        # Launch always injects a YAML list *string*; declaring as DOUBLE_ARRAY
        # causes InvalidParameterTypeException on startup.
        self.declare_parameter(
            "people",
            DEFAULT_PEOPLE_STR,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Flat [x,y,theta,...] as a YAML list string"))

        self.frame_id = self.get_parameter("frame_id").value
        rate_hz = float(self.get_parameter("rate_hz").value)
        raw = _parse_people_parameter(self.get_parameter("people").value)
        if not raw:
            self.get_logger().warn(
                "Empty `people` parameter — using built-in defaults.")
            raw = list(DEFAULT_PEOPLE)
        if len(raw) % 3 != 0:
            self.get_logger().warn(
                f"`people` parameter has {len(raw)} entries, expected a multiple "
                f"of 3 (x,y,theta).  Dropping trailing values.")
            raw = raw[: len(raw) - (len(raw) % 3)]
        self.people_poses = [(raw[i], raw[i + 1], raw[i + 2])
                             for i in range(0, len(raw), 3)]

        self.pub = self.create_publisher(People, "/people", 10)
        self.marker_pub = None
        if _HAS_MARKERS:
            self.marker_pub = self.create_publisher(
                MarkerArray, "/people_markers", 10)
        else:
            self.get_logger().warn(
                "visualization_msgs not available — skipping /people_markers")

        self.timer = self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().info(
            f"Publishing {len(self.people_poses)} fake people in frame "
            f"'{self.frame_id}' at {rate_hz:.1f} Hz "
            f"(poses: {self.people_poses})")

    def _tick(self):
        msg = People()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        for (x, y, theta) in self.people_poses:
            p = Person()
            p.position.x = float(x)
            p.position.y = float(y)
            p.position.z = float(theta)
            p.velocity.x = 0.0
            p.velocity.y = 0.0
            p.velocity.z = 0.0
            msg.people.append(p)
        self.pub.publish(msg)
        if self.marker_pub is not None:
            self._publish_markers(msg.header)

    def _publish_markers(self, header):
        markers = MarkerArray()
        for idx, (x, y, theta) in enumerate(self.people_poses):
            body = Marker()
            body.header = header
            body.ns = "fake_people"
            body.id = idx * 2
            body.type = Marker.CYLINDER
            body.action = Marker.ADD
            body.pose.position.x = float(x)
            body.pose.position.y = float(y)
            body.pose.position.z = 0.35
            body.pose.orientation.w = 1.0
            body.scale.x = 0.4
            body.scale.y = 0.4
            body.scale.z = 0.7
            body.color = ColorRGBA(r=0.1, g=0.6, b=1.0, a=0.85)
            body.lifetime.sec = 0
            markers.markers.append(body)

            arrow = Marker()
            arrow.header = header
            arrow.ns = "fake_people"
            arrow.id = idx * 2 + 1
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.pose.position.x = float(x)
            arrow.pose.position.y = float(y)
            arrow.pose.position.z = 0.75
            arrow.pose.orientation.z = math.sin(theta / 2.0)
            arrow.pose.orientation.w = math.cos(theta / 2.0)
            arrow.scale.x = 0.6
            arrow.scale.y = 0.12
            arrow.scale.z = 0.12
            arrow.color = ColorRGBA(r=1.0, g=0.4, b=0.1, a=0.9)
            arrow.lifetime.sec = 0
            markers.markers.append(arrow)

        self.marker_pub.publish(markers)


def _parse_people_parameter(value) -> list[float]:
    """Accept double[], YAML list strings from launch, or empty []."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    if isinstance(value, str):
        text = value.strip()
        if not text or text == "[]":
            return []
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Could not parse `people` parameter: {text!r}") from exc
        if not isinstance(parsed, (list, tuple)):
            raise ValueError(f"`people` must be a list, got {type(parsed)}")
        return [float(v) for v in parsed]
    return [float(value)]


def main(args=None):
    rclpy.init(args=args)
    try:
        node = FakePeoplePublisher()
        rclpy.spin(node)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
