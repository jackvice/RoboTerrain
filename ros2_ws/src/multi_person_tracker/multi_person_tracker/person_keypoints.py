
import numpy as np
from typing import List


class keypoint():

    def __init__(self, ID: int, xImage: float, yImage: float):
        self.ID: str = ID
        self.xImage: float = xImage
        self.yImage: float = yImage
        self.x: float = None
        self.y: float = None
        self.z: float = None

    def calculate3DKeypoint(self, depth, depthRadiusX: int = 2, depthRadiusY: int = 2, resolutionX=640, resolutionY=480, HFOV=np.radians(54.732), VFOV=np.radians(42.4115)):
        """
        Parameters
        ----------
        depth : numpy array 
            numpy array of depth aligned with the image for detections
        Return
        ----------
        distance : float
            distance to the detected person in meters
        angle : float
            angle to the centre of the bounding box of the person
        """

        # relative to the image center such that the distances are postive going left and upwards according to REP
        centreX = (resolutionX/2) - self.xImage
        centreY = (resolutionY/2) - self.yImage

        # Length of the distance to the virtual image plane
        Id = (resolutionX/2)/np.tan(HFOV/2)
        # Length of the hypothenuse going towards the bb on the y=centrey plane
        Idx = np.sqrt((Id**2) + (centreX**2))
        # TODO check if arctan is better than arctan2
        # angle between Idx and the line going from the camera to the centre of the bb
        delta = np.arctan2(centreY, Idx)

        # Angle between idx and ID
        gamma = np.arctan2(centreX, Id)

        # get distances of depth image assuming same resolution and allignment relative to bounding box coordinates
        distBox = depth[int(max(self.yImage - depthRadiusY, 0)):
                        int(min(self.yImage + depthRadiusY, resolutionY)),
                        int(max(self.xImage - depthRadiusX, 0)):
                        int(min(self.xImage + depthRadiusX, resolutionX))
                        ]
        distBox = distBox.flatten()

        distBox = np.delete(distBox, np.argwhere(distBox == 0))

        distance = np.nanmedian(distBox)

        self.z = np.sin(delta) * distance
        # Projection to horizontal plane happening here
        distance = distance * np.cos(delta)
        # Output in polar coordinates such that angles to the left are positive and angles to the right are negative
        # Distance Forward is positiv backwards not possible
        # return x andy according to REP
        self.y = np.sin(gamma) * distance
        self.x = np.cos(gamma) * distance


class person_keypoint:
    '''
    Keypoints for one detection are passed to this object for calculation of 3D location and orientation
    '''

    def __init__(self, keypoints, depth):
        # self.IDnames = np.array(["nose", "left_eye", "right_eye", "left_ear",
        #                    "right_ear", "left_shoulder", "right_shoulder",
        #                    "left_elbow", "right_elbow", "left_wrist",
        #                    "right_wrist", "left_hip", "right_hip", "left_knee",
        #                    "right_knee", "left_ankle", "right_ankle", "neck"])

        self.keypoints: List[keypoint] = []
        self.depth = depth
        self.orientation: float = 0.0
        self.x: float = None
        self.y: float = None
        self.withTheta: bool = True
        for kp in keypoints:
            self.keypoints.append(keypoint(kp.ID, kp.x, kp.y))

        self.left_shoulder = next(
            (point for point in self.keypoints if point.ID == 5), None)
        self.right_shoulder = next(
            (point for point in self.keypoints if point.ID == 6), None)
        self.left_hip = next(
            (point for point in self.keypoints if point.ID == 11), None)
        self.right_hip = next(
            (point for point in self.keypoints if point.ID == 12), None)
        self.left_ear = next(
            (point for point in self.keypoints if point.ID == 3), None)
        self.right_ear = next(
            (point for point in self.keypoints if point.ID == 4), None)
        self.neck = next(
            (point for point in self.keypoints if point.ID == 17), None)

        self.getPersonOrientation()
        self.getPersonPosition()

    def getPersonOrientation(self):
        '''Brief: do the check which keypoints are present, then calculate orientation'''
        if (self.left_shoulder and self.right_shoulder):
            self.getOrientationFromPoints(
                self.left_shoulder, self.right_shoulder)
            return
        if (self.left_hip and self.right_hip):
            self.getOrientationFromPoints(self.left_hip, self.right_hip)
            return
        # if (self.left_hip or self.left_shoulder):
        #     self.orientation = np.pi/2
        #     return
        # if (self.right_hip or self.right_shoulder):
        #     self.orientation = 3*np.pi/2
        #     return
        self.withTheta = False

    def getOrientationFromPoints(self, left: keypoint, right: keypoint):
        left.calculate3DKeypoint(self.depth)
        right.calculate3DKeypoint(self.depth)
        if (left.x == np.nan or left.y == np.nan or right.x == np.nan or right.y == np.nan):
            self.withTheta = False
        else:
            # arctan returns angle of shoulders therefore normal vector is offset by 90deg
            beta = np.arctan2(left.y-right.y, right.x-left.x)
            # make angle 0->2pi
            beta = beta if beta > 0 else beta+2*np.pi

            beta = np.pi/2 - beta

            if beta != np.nan:
                # make sure its between 0->2 pi
                self.orientation = np.mod(beta, 2 * np.pi)
            else:
                self.orientation = 0.0
            # get angle back into the spectrum of -pi->pi

    def getPersonPosition(self):
        importantKeypoints = [self.neck, self.left_shoulder,
                              self.right_shoulder, self.left_hip, self.right_hip]
        kpx = []
        kpy = []
        kp: keypoint
        for kp in importantKeypoints:
            if kp:
                kp.calculate3DKeypoint(
                    depth=self.depth, depthRadiusX=1, depthRadiusY=1)
                kpx.append(kp.x)
                kpy.append(kp.y)
        if len(kpx) != 0 and len(kpy) != 0:
            self.x = np.nanmean(np.array(kpx))
            self.y = np.nanmean(np.array(kpy))
