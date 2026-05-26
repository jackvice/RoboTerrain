import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
class Detection:
    def __init__(self, x: float, y: float, orientation: float, withTheta: bool = True, keypoints: list = []):
        self.x = x
        self.y = y
        self.orientation = orientation
        self.withTheta = withTheta
        self.keypoints = keypoints


class KalmanFilter(object):
    """
    Kalman Filter with states x,y,theta,xdot,ydot,thetadot
    Constant velocity model but with natural decay of velocity
    Can be updated with and without theta by setting measWithTheta flag accordingly

    Parameters
    ----------
    x: initial x measurement
    y: initial y measurement
    theta: initial orientation measurement
    dt: sampling time (time for 1 cycle)
    u_x: acceleration in x-direction
    u_y: acceleration in y-direction
    u_theta: acceleration in orientation
    std_acc: process noise magnitude
    std_theta_acc: theta process noise magnitude
    x_std_meas: standard deviation of the measurement in x-direction
    y_std_meas: standard deviation of the measurement in y-direction
    theta_std_meas: standard deviation of the measurement in orientation (theta)
    decay: amount of decay applied to velocities at each prediction
    """

    def __init__(
            self,
            x,
            y,
            theta=0,
            withTheta=True,
            timestamp=0,
            dt=0.1,
            u_x=0,
            u_y=0,
            u_theta=0,
            std_acc=0.0001,
            std_theta_acc=0.00001,
            x_std_meas=0.000001,
            y_std_meas=0.000001,
            theta_std_meas=0.000001,
            decay=0.90,
            keypoints=[],
            debug=False):

        self.debug = debug

        # Define variables for sotring current position
        self.personX = x
        self.personY = y
        self.personTheta = theta
        self.personXdot = 0
        self.personYdot = 0
        self.personThetadot = 0
        self.timestamp = timestamp
        self.decay = decay*dt  # so the decay is consistent over different dt's
        self.initialized = False
        self.measX = x
        self.measY = y
        self.measTheta = theta
        self.measTimestamp = timestamp
        self.measWithTheta = withTheta
        self.residual = 0
        self.keypoints = keypoints

        # Define sampling time
        self.dt = dt
        self.std_acc = std_acc
        # Define the  control input variables
        self.u = np.matrix([[u_x], [u_y], [u_theta]])

        # Intial State
        self.x = np.matrix([[self.personX], [self.personY], [self.personTheta], [
                           self.personXdot], [self.personYdot], [self.personThetadot]])

        # Define the State Transition Matrix A
        self.A = np.matrix([[1, 0, 0, self.dt, 0, 0],
                            [0, 1, 0, 0, self.dt, 0],
                            [0, 0, 1, 0, 0, self.dt],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1]])

        self.Q = np.matrix([[(self.dt**4) / 4, 0, 0, (self.dt**3) / 2, 0, 0],
                            [0, (self.dt**4) / 4, 0, 0, (self.dt**3) / 2, 0],
                            [0, 0, (self.dt**4) / 4, 0, 0, (self.dt**3) / 2],
                            [(self.dt**3) / 2, 0, 0, self.dt**2, 0, 0],
                            [0, (self.dt**3) / 2, 0, 0, self.dt**2, 0],
                            [0, 0, (self.dt**3) / 2, 0, 0, self.dt**2]]) * self.std_acc**2
        self.Q[2, 2] = ((self.dt**4)/4)*std_theta_acc**2
        self.Q[2, 5] = ((self.dt**3) / 2)*std_theta_acc**2
        self.Q[5, 2] = ((self.dt**3) / 2)*std_theta_acc**2
        self.Q[5, 5] = (self.dt**2)*std_theta_acc**2
        # Define the Control Input Matrix B
        self.B = np.matrix([[(self.dt**2) / 2, 0, 0],
                            [0, (self.dt**2) / 2, 0],
                            [0, 0, (self.dt**2) / 2],
                            [self.dt, 0, 0],
                            [0, self.dt, 0],
                            [0, 0, self.dt]])

        # Define Measurement Mapping Matrix
        self.H = np.matrix([[1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0]])
        # Define Measurement Mapping Matrix without Theta
        self.Halternative = np.matrix([[1, 0, 0, 0, 0, 0],
                                       [0, 1, 0, 0, 0, 0]])

        # Initial Measurement Noise Covariance
        self.R = np.matrix([[x_std_meas**2, 0, 0],
                           [0, y_std_meas**2, 0],
                           [0, 0, theta_std_meas**2]])
        # Initial Measurement Noise Covariance
        self.Ralternative = np.matrix([[x_std_meas**2, 0],
                                       [0, y_std_meas**2]])

        # Initial Covariance Matrix
        self.P = np.eye(self.A.shape[1])

        # matrix for decay the influence of prediction on movement over time when no detection
        self.DecayMatrix = np.matrix([[1, 0, 0, 0.0, 0, 0],
                                      [0, 1, 0, 0, 0.0, 0],
                                      [0, 0, 1, 0, 0, 0.0],
                                      [0, 0, 0, self.decay, 0, 0],
                                      [0, 0, 0, 0, self.decay, 0],
                                      [0, 0, 0, 0, 0, self.decay]])

    def predict(self):
        # Update time state
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)

        # TODO determine use of Decay velocity measurement
        self.x = np.dot(self.DecayMatrix, self.x)

        # Calculate error covariance
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

        # Update position
        self.personX = self.x[0]
        self.personY = self.x[1]
        self.personTheta = self.x[2]
        self.personXdot = self.x[3]
        self.personYdot = self.x[4]
        self.personThetadot = self.x[5]

    def update(self):
        if self.measWithTheta:
            H = self.H
            R = self.R
            self.measTheta = np.unwrap(
                np.array([float(self.x[2]), float(self.measTheta)]))[1]
            z = [[self.measX], [self.measY], [self.measTheta]]
        else:
            H = self.Halternative
            R = self.Ralternative
            z = [[self.measX], [self.measY]]

        S = np.dot(H, np.dot(self.P, H.T)) + R

        # Calculate the Kalman Gain
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))
        # print(f"Kalman gain:\n {K}")

        # self.x = np.round(self.x + np.dot(K, (z - np.dot(H, self.x))))
        self.x = self.x + np.dot(K, (z - np.dot(H, self.x)))

        I = np.eye(H.shape[1])

        # Update error covariance matrix
        self.P = (I - (K * H)) * self.P

        # Update position
        self.personX = self.x[0]
        self.personY = self.x[1]
        self.personTheta = self.x[2]
        self.personXdot = self.x[3]
        self.personYdot = self.x[4]
        self.personThetadot = self.x[5]
        self.timestamp = self.measTimestamp


class PeopleTracker(object):
    """
    Multi object tracker class used for tracking 2D pose of humans

    Tracklets can be updated and initialised by supplying a List[Detection] and a timestamp in ns as int of when the detection occurred
    Tracklets can be updated with and withoud Theta by setting withTheta of a Detection object to false

    Each Tracklet is represented by a Kalman Filter with a constant velocity model and a natural velocity decay that tracks X,Y,Theta,Xdot,Ydot,Thetadot
    Detections are automatically assigned to the tracklets using munkres algorithm

    Tracklets are added when the amount of tracklets is smaller than the amount of detections or when a detection is further away than newTrack
    Tracklets are removed when they havent been updated in keeptime

    Parameters
    ----------
    newTrack: distance at which new tracklets are initialised instead of being asigned to existing ones
    keeptime: amount of time tracklets are held without being updated [s]
    dt: dt at which kalman filter predictions are run
    """

    def __init__(self, newTrack=3, keeptime=5, dt=0.02, debug=False):
        # initialise the tracker with an empty list of people
        self.newTrack = newTrack
        self.keeptime = keeptime
        self.dt = dt
        self.debug = debug
        self.tracklets = []

    def predict(self,timestamp):
        popCounter=0
            # remove tracklet if it hasn't been updated in too long
        for i in range(len(self.tracklets)):
            if abs(timestamp-self.tracklets[i-popCounter].timestamp)*1e-9 > self.keeptime:
                if self.debug:
                    print("popped segment for being too old")
                self.tracklets.pop(i-popCounter)
                popCounter += 1
                
        # perform prediction with the Kalman filter
        for person in self.tracklets:
            person.predict()

    def update(self, detections, timestamp):
        # update the tracklets with new detections
        updates = self.MunkresTrack(
            detections, self.tracklets, timestamp)
        for update in updates:
            self.tracklets[update].update()

    def MunkresDistances(self, detections, tracklets, timestamp):
        # Calculate distances between objects and detections and save shortest
        popCounter = 0
        tracklet_pos=[]
        detection_pos=[]
        for person in tracklets:
            tracklet_pos.append((float(person.personX),float(person.personY)))
        for detection in detections:
            detection_pos.append((float(detection.x),float(detection.y)))
        distMat = cdist(tracklet_pos,detection_pos,metric="euclidean")
        # if detection is too far away pop it and create new KF for it
        mins = distMat.min(axis=0)
        if np.any(mins > self.newTrack):
            for i, min in enumerate(mins):
                if min > self.newTrack:
                    tracklets.append(
                        KalmanFilter(
                            detections[i - popCounter].x,
                            detections[i - popCounter].y,
                            detections[i - popCounter].orientation,
                            withTheta=detections[i - popCounter].withTheta,
                            timestamp=timestamp,
                            dt=self.dt,
                            keypoints=detections[i - popCounter].keypoints))
                    detections.pop(i - popCounter)
                    distMat = np.delete(distMat, i-popCounter, 0)
                    popCounter += 1
        # Find shortest distance
        self.indexes = linear_sum_assignment(distMat)
        return self.indexes

    def MunkresTrack(self, detections, tracklets, timestamp):
        updates = []
        popCounter=0
        if len(tracklets):
            indexes = self.MunkresDistances(
                detections, tracklets, timestamp)
            if len(detections):  # check again since we might have popped one
                if self.debug:
                    if len(tracklets) < len(detections):
                        print("More detections than tracklets")
                    if len(tracklets) > len(detections):
                        print("More tracklets than detections")
                    if len(tracklets) == len(detections):
                        print("Same amount of detections and tracklets")

                # assign all found assignments
                for index in zip(indexes[0], indexes[1]):
                    tracklets[index[0]].measX = detections[index[1]-popCounter].x
                    tracklets[index[0]].measY = detections[index[1]-popCounter].y
                    tracklets[index[0]
                              ].measTheta = detections[index[1]-popCounter].orientation
                    tracklets[index[0]].measTimestamp = timestamp
                    tracklets[index[0]
                              ].measWithTheta = detections[index[1]-popCounter].withTheta
                    tracklets[index[0]
                              ].keypoints = detections[index[1]-popCounter].keypoints
                    updates.append(index[0])
                    detections.pop(index[1]-popCounter)  # pop every assigned detection
                    popCounter+=1
        # append the remaining detections as new KF's
        for detection in detections:
            tracklets.append(
                KalmanFilter(
                    detection.x,
                    detection.y,
                    detection.orientation,
                    withTheta=detection.withTheta,
                    timestamp=timestamp,
                    dt=self.dt,
                    keypoints=detection.keypoints))

        return updates
