import math
import rclpy
import os
from rclpy.node import Node
import threading
import numpy as np
import cv2
from cv_bridge import CvBridge
from tf_transformations import euler_from_quaternion, quaternion_about_axis
import tf2_ros
import tf2_geometry_msgs
import csv  # DC remove later

from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, PointStamped

from .person_keypoints import *
from .pose_backend import build_pose_backend
from multi_person_tracker_interfaces.msg import People, Person
from .tracking import PeopleTracker, Detection
from rclpy.qos import QoSProfile, HistoryPolicy, DurabilityPolicy, ReliabilityPolicy



class MultiPersonTracker(Node):
    def __init__(self, publishPoseMsg: bool = True, publishKeypoints: bool = None,
                 dt: float = None, n_cameras: int = None, newTrack: float = None,
                 keeptime: float = None, target_frame: str = None,
                 debug: bool = None):
        '''
        Class for pose estimation of a person using Nvidia jetson Orin implementation
        of PoseNet and passing messages using ROS2.
        The Class uses Intel Realsense messages on the ROS2 network as input for rgb and depth images

        Parameters
        ----------
        publishPoseMsg: publish filtered marker arrows to the ROS2 network 
        publishKeypoints: publish non filtered keypoints as markers
        dt: rate of prediction for the trackers
        n_cameras: number of publishing cameras on the ROS2 network
        newTrack: meters distance at which detection is not assigned to tracklets and new ones are generated 
        keeptime: seconds to keep tracklets after last detection
        target_frame ouput tf_frame of the poses
        debug: display debug messages in the console
        '''

        super().__init__('multi_person_tracker')

        # Declare params on the main node itself so use_sim_time and other
        # overrides from the launch file propagate correctly to this node's
        # clock / TF buffer.  (Reading them via a separate bootstrap node
        # broke use_sim_time inheritance, leading to wall-clock timestamps
        # on /people and "Transform data too old" errors in Nav2.)
        self.declare_parameter('n_cameras', 2)
        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('dt', 0.1)
        self.declare_parameter('new_track', 3.0)
        self.declare_parameter('keeptime', 5.0)
        self.declare_parameter('debug', False)

        if n_cameras is None:
            n_cameras = int(self.get_parameter('n_cameras').value)
        if target_frame is None:
            target_frame = str(self.get_parameter('target_frame').value)
        if dt is None:
            dt = float(self.get_parameter('dt').value)
        if newTrack is None:
            newTrack = float(self.get_parameter('new_track').value)
        if keeptime is None:
            keeptime = float(self.get_parameter('keeptime').value)
        if debug is None:
            debug = bool(self.get_parameter('debug').value)
        if publishKeypoints is None:
            publishKeypoints = False

        self.create_timer(dt, self.timer_callback)
        self.people_tracker = PeopleTracker(
            newTrack=newTrack, keeptime=keeptime, dt=dt, debug=debug)
        self.people_publisher = self.create_publisher(People, 'people', 10)
        self.people_arrow_publisher = self.create_publisher(
            MarkerArray, 'people_arrows', 10)
        self.people_keypoint_publisher = self.create_publisher(
            MarkerArray, 'people_keypoints', 10)
        self.publishPoseMsg = publishPoseMsg
        self.publishKeypointsMsg = publishKeypoints
        self.debug = debug
        self.target_frame = target_frame
        ### Variables for pose detection###
        self.peopleCount = 0
        self.imageCount = -1
        self.written = False
        self.cameras = []
        self.Orientations = []
        # Pose detection backend (jetson_inference on Jetson, Ultralytics
        # YOLOv8-pose on x86 GPU). Configure via POSE_BACKEND / POSE_MODEL env
        # vars; defaults to auto-detection.
        self.threshold = float(os.environ.get("POSE_THRESHOLD", "0.3"))
        self.network = os.environ.get("POSE_BACKEND", "auto")
        self.output_location = os.environ.get(
            "POSE_OUTPUT_DIR", "/docker-volume/images")
        self.pose_backend = build_pose_backend(threshold=self.threshold)
        self.save_debug_images = bool(
            int(os.environ.get("POSE_SAVE_DEBUG_IMAGES", "0")))

        self.detectionMergingThreshold = 0.5
        # Initialize camera objects with propper namespacing
        if n_cameras > 1:
            self.cameras = [self.Camera(self, namespace="camera"+str(i+1))
                            for i in range(n_cameras)]
        else:
            self.cameras = [self.Camera(self)]

    def timer_callback(self):
        # Publishes Tracker Ouput and predicts next state
        people = People()
        people.header.stamp = self.get_clock().now().to_msg()
        # TODO change when we have tf goodness
        people.header.frame_id = self.target_frame
        # TODO implement index and reliabílity
        dropped = 0
        for p in self.people_tracker.tracklets:
            # Degenerate Kalman estimates have been observed in the wild (see
            # /people echo showing values like (nan, inf, 2.49)).  Bad tracklets
            # propagate into social_map_generator and the SocialLayer, where
            # they paint LETHAL/INSCRIBED costmap cells at undefined locations
            # — causing "Starting point in lethal space" planner aborts and
            # 0.5 m proxemic violations against ghost pedestrians.  Drop them
            # at the source so no downstream consumer sees them.
            if not (math.isfinite(p.personX)
                    and math.isfinite(p.personY)
                    and math.isfinite(p.personTheta)
                    and math.isfinite(p.personXdot)
                    and math.isfinite(p.personYdot)
                    and math.isfinite(p.personThetadot)):
                dropped += 1
                continue
            person = Person()
            person.position.x = float(p.personX)
            person.position.y = float(p.personY)
            person.position.z = float(p.personTheta)
            person.velocity.x = float(p.personXdot)
            person.velocity.y = float(p.personYdot)
            person.velocity.z = float(p.personThetadot)
            people.people.append(person)

        if dropped:
            self.get_logger().warn(
                f'Dropped {dropped} non-finite tracklet(s) before publishing /people',
                throttle_duration_sec=5.0)

        self.people_publisher.publish(people)
        if self.publishPoseMsg:
            self.publishPoseArrows(self.people_tracker.tracklets)
        if self.publishKeypointsMsg:
            self.publishKeypoints(self.people_tracker.tracklets)
        self.people_tracker.predict(self.get_clock().now().nanoseconds)

    def publishPoseArrows(self, people):
        # Set the scale of the marker
        marker_array_msg = MarkerArray()

        for i, person in enumerate(people):
            # Set the pose of the marker
            if (person.personX and person.personY and person.personTheta):
                quad = quaternion_about_axis(person.personTheta, (0, 0, 1))
                marker = Marker()
                marker.header.frame_id = self.target_frame
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.type = 0
                marker.id = i
                marker.pose.position.x = float(person.personX)
                marker.pose.position.y = float(person.personY)
                marker.pose.position.z = float(0)
                marker.pose.orientation.x = quad[0]
                marker.pose.orientation.y = quad[1]
                marker.pose.orientation.z = quad[2]
                marker.pose.orientation.w = quad[3]
                marker.scale.x = 1.0
                marker.scale.y = 0.1
                marker.scale.z = 0.1

                # Set the color
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0
                marker.frame_locked = False
                marker_array_msg.markers.append(marker)
        self.people_arrow_publisher.publish(marker_array_msg)

    def publishKeypoints(self, people):
        # Set the scale of the marker
        marker_array_msg = MarkerArray()
        for i, person in enumerate(people):
            # Set the pose of the marker
            if (person.personX and person.personY and person.personTheta and len(person.keypoints)):
                # Set the pose of the marker
                marker = Marker()
                marker.header.frame_id = self.target_frame
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.type = 8
                marker.id = i
                marker.scale.x = .05
                marker.scale.y = .05
                marker.scale.z = .05
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0
                for kp in person.keypoints:
                    marker.points.append(kp.point)
                marker_array_msg.markers.append(marker)
        self.people_keypoint_publisher.publish(marker_array_msg)

    def detect(self, image, depthImage):
        '''
        Perform pose estimation on a numpy image.  ``image`` may be BGR or BGRA.

        Diagnostic counters help confirm the pipeline is alive when
        ``/people`` stays empty.  Set the ``debug`` ROS param or
        ``POSE_SAVE_DEBUG_IMAGES=1`` env var to enable per-frame dumps in
        ``saveImage``.
        '''
        self._detect_calls = getattr(self, '_detect_calls', 0) + 1
        if image is None:
            self._detect_no_image = getattr(self, '_detect_no_image', 0) + 1
            return None
        if not isinstance(depthImage, np.ndarray):
            self._detect_no_depth = getattr(self, '_detect_no_depth', 0) + 1
            if self._detect_no_depth in (1, 30, 100) or self._detect_no_depth % 200 == 0:
                self.get_logger().warn(
                    f'detect() skipped — no depth image yet '
                    f'({self._detect_no_depth} consecutive frames).  Check '
                    f'`ros2 topic hz </camera>/aligned_depth_to_color/image_raw`.')
            return None
        poses = self.pose_backend.process(image)
        self._detect_with_depth = getattr(self, '_detect_with_depth', 0) + 1
        npose = len(poses) if poses is not None else 0
        if self._detect_with_depth in (1, 30, 100) or self._detect_with_depth % 200 == 0:
            self.get_logger().info(
                f'YOLO ran on frame {self._detect_with_depth}: '
                f'{npose} pose(s) returned')
        return poses

    def saveImage(self, image):
        '''Optionally write a debug frame to disk.  No GPU-side overlay is
        produced here; callers can extend this for richer visualisation.'''
        if not self.save_debug_images:
            return
        self.imageCount += 1
        try:
            os.makedirs(self.output_location, exist_ok=True)
            out_path = os.path.join(
                self.output_location, f"frame_{self.imageCount:06d}.png")
            if image.ndim == 3 and image.shape[2] == 4:
                bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            else:
                bgr = image
            if bgr.dtype != np.uint8:
                bgr = np.clip(bgr, 0, 255).astype(np.uint8)
            cv2.imwrite(out_path, bgr)
            if self.imageCount in (1, 10, 30, 100) or self.imageCount % 100 == 0:
                self.get_logger().info(
                    f'Wrote debug frame {self.imageCount} → {out_path}')
        except Exception as exc:
            self.get_logger().warn(f'Failed to write debug image: {exc}')

    class Camera(object):
        def __init__(self, tracker_self, namespace: str = "camera"):

            self.rgb = None
            self.image = None  # numpy image used for pose detection
            self.depth = None
            self.bridge = CvBridge()
            self.timestamp = None
            self.tracker = tracker_self
            self.debug = self.tracker.debug
            if self.debug:
                print("init camera")
            self.qos_profile = QoSProfile(
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                depth=5)
            self.namespace = namespace
            self.tfFrame = self.namespace+"_color_frame"#TODO check if this is supposed to be "aligned_depth_to_color_frame"
            self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.time.Duration(seconds=5.0))
            self.tf_listener = tf2_ros.TransformListener(
                self.tf_buffer, self.tracker, spin_thread = True)

            # Initialize subscribers in tracker object for this camera
            self.rgb_subscription = self.tracker.create_subscription(
                Image,
                '/' + namespace+'/color/image_raw',
                self.rgb_callback,
                10)

            self.depth_subscription = self.tracker.create_subscription(
                Image,
                '/' + namespace+'/aligned_depth_to_color/image_raw',
                self.depth_callback,
                10)

        def rgb_callback(self, msg):
            try:
                self.rgb = self.bridge.imgmsg_to_cv2(
                    msg, desired_encoding='passthrough')
                # Keep a BGR/BGRA numpy image around for the pose backend.
                self.image = self.rgb
                self.timestamp = self.tracker.get_clock().now().nanoseconds

                # Diagnostic: dump every RGB frame the tracker receives so we
                # can confirm (a) the camera is reaching YOLO and (b) what
                # YOLO is being asked to detect.  Cheap when off (env var
                # POSE_SAVE_DEBUG_IMAGES=0).
                self.tracker.saveImage(self.image)

                # detect poses when new rgb immage is available
                poses = self.tracker.detect(
                    self.image, self.depth)

                # generate 3D coordinates for all keypoints and calculate x,y,theta
                if poses:
                    kpPersons = self.generatePeople(poses)
                    self.tracker._kp_persons_total = getattr(
                        self.tracker, '_kp_persons_total', 0) + len(kpPersons)
                    self.tracker._poses_total = getattr(
                        self.tracker, '_poses_total', 0) + len(poses)
                    if self.tracker._poses_total in (1, 5, 30, 100) or \
                            self.tracker._poses_total % 200 == 0:
                        self.tracker.get_logger().info(
                            f'generatePeople kept {self.tracker._kp_persons_total} '
                            f'kpPersons out of {self.tracker._poses_total} YOLO poses '
                            f'(0 means depth keypoint lookups all failed)')
                    z = np.array([[complex(p.x, p.y) for p in kpPersons]])
                    popCounter=0
                    distanceMatrix=abs(z.T-z)
                    distanceMatrix = np.where(np.logical_and(0 < distanceMatrix, distanceMatrix < self.tracker.detectionMergingThreshold))
                    for i,detection in enumerate(distanceMatrix[0]):
                        kpPersons[detection-popCounter].x = (kpPersons[detection-popCounter].x + kpPersons[distanceMatrix[1][i]-popCounter].x)/2
                        kpPersons[detection-popCounter].y = (kpPersons[detection-popCounter].y + kpPersons[distanceMatrix[1][i]-popCounter].y)/2
                        kpPersons[detection-popCounter].orientation = (kpPersons[detection-popCounter].orientation + kpPersons[distanceMatrix[1][i]-popCounter].orientation)/2
                        kpPersons.pop(distanceMatrix[1][i]-popCounter)
                        popCounter+=1
                        if self.debug: print("removed double detection")
                    # make detection objects
                    detections = []
                    trans = None
                    try:
                        trans = self.tf_buffer.lookup_transform(
                            self.tracker.target_frame, self.tfFrame,
                            rclpy.time.Time(),
                            timeout=rclpy.time.Duration(seconds=0.5))
                    except Exception as e:
                        self.tracker._tf_fails = getattr(
                            self.tracker, '_tf_fails', 0) + 1
                        if self.tracker._tf_fails in (1, 5, 30) or \
                                self.tracker._tf_fails % 50 == 0:
                            self.tracker.get_logger().warn(
                                f'TF lookup {self.tfFrame} → '
                                f'{self.tracker.target_frame} failed '
                                f'({self.tracker._tf_fails} times): {e}')
                    if trans:
                        pose = Pose()
                        for person in kpPersons:
                            try:
                                if self.tracker.publishKeypoints:
                                    keypoints = []
                                    for kp in person.keypoints:
                                        if kp.x and kp.y and kp.z:
                                            point = PointStamped()
                                            point.point.x = float(kp.x)
                                            point.point.y = float(kp.y)
                                            point.point.z = float(kp.z)
                                            point = tf2_geometry_msgs.do_transform_point(
                                                point, trans)
                                            keypoints.append(point)
                                # transformation to target_frame
                                pose.position.x = float(person.x)
                                pose.position.y = float(person.y)
                                pose.position.z = float(0.0)
                                angle = person.orientation if person.orientation < np.pi else person.orientation-2*np.pi
                                quad = quaternion_about_axis(
                                    person.orientation, (0, 0, 1))
                                pose.orientation.x = quad[0]
                                pose.orientation.y = quad[1]
                                pose.orientation.z = quad[2]
                                pose.orientation.w = quad[3]
                                pose = tf2_geometry_msgs.do_transform_pose(
                                    pose, trans)
                                quad = [
                                    pose.orientation.x,
                                    pose.orientation.y,
                                    pose.orientation.z,
                                    pose.orientation.w,
                                ]
                                angle = euler_from_quaternion(quad)[2]
                                angle = angle if angle > 0 else angle+2*np.pi


                                if self.tracker.publishKeypoints:
                                    detections.append(
                                        Detection(pose.position.x, pose.position.y, angle, person.withTheta, keypoints))
                                else:
                                    detections.append(
                                        Detection(pose.position.x, pose.position.y, angle, person.withTheta))
                            except np.linalg.LinAlgError:
                                pass
                        # Update tracker with new detections
                        if len(detections):
                            self.tracker.people_tracker.update(
                                detections, self.timestamp)

                            # save image and make csv if required
                            if self.debug:
                                # self.writing(kpPersons)
                                self.tracker.peopleCount += len(
                                    kpPersons)
                                self.tracker.saveImage(self.image)
            except Exception as e:
                if self.debug:
                    print(f"Exception on rgb_callback")
                    print(e)

        def depth_callback(self, msg):
            # get and update depth image
            try:
                depth = self.bridge.imgmsg_to_cv2(
                    msg, desired_encoding='passthrough')
                # Hardware Intel RealSense publishes depth as uint16 in mm.
                # Gazebo's simulated RealSense plugin publishes float32 already
                # in metres.  Detect by encoding/dtype so both work.
                arr = np.asarray(depth)
                if arr.dtype == np.uint16 or msg.encoding == '16UC1':
                    self.depth = arr.astype(np.float32) * 0.001  # mm → m
                else:
                    self.depth = arr.astype(np.float32)           # already m
                if not getattr(self, '_depth_logged', False):
                    self.tracker.get_logger().info(
                        f'depth_callback: encoding={msg.encoding} dtype={arr.dtype} '
                        f'shape={arr.shape} min={float(np.nanmin(arr)):.3f} '
                        f'max={float(np.nanmax(arr)):.3f}')
                    self._depth_logged = True
                # TODO Check do we actually want to update the timestamp
                self.timestamp = self.tracker.get_clock().now().nanoseconds
            except Exception as e:
                self.tracker.get_logger().warn(f'Exception on depth_callback: {e}')

        def generatePeople(self, poses):
            '''
            Calculates the location of the person as X and Y coordinates along with the orientation of the person
            '''
            persons = []
            for pose in poses:
                kpPerson = person_keypoint(pose.Keypoints, self.depth)
                if kpPerson.x != None and kpPerson.y != None:
                    persons.append(kpPerson)
            return persons

        def writing(self, orientation):
            '''
            Data collection function for writing csv file with person variables for captured images
            '''
            with open('Measurement.csv', mode='a') as csvfile:
                writer = csv.writer(csvfile, delimiter=',',
                                    quotechar='"', quoting=csv.QUOTE_MINIMAL)

                if not self.tracker.written:
                    writer.writerow(['Orientation'])

                self.tracker.written = True
                writer.writerow([str(round(orientation, 3))])


def main(args=None):
    """Entry point.

    Launch parameters (all declared on the node itself; defaults shown):

    * ``use_sim_time`` (bool, default False) — set True under Gazebo.
    * ``n_cameras`` (int, default 2) — number of RealSense-style cameras.
      With ``n_cameras=1`` the tracker subscribes to ``/camera/...`` topics;
      with N>1 it uses ``/camera1/...``, ``/camera2/...``, etc.
    * ``target_frame`` (str, default ``"map"``) — TF frame the published
      ``/people`` poses are expressed in.
    * ``dt`` (double, default 0.1) — tracker update rate (seconds).
    * ``new_track`` (double, default 3.0) — distance threshold for starting
      a new tracklet.
    * ``keeptime`` (double, default 5.0) — seconds to retain a lost tracklet.
    * ``debug`` (bool, default False) — verbose logging.
    """
    rclpy.init(args=args)
    node = MultiPersonTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
