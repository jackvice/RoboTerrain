# This message represents the pose and velocity of a differential wheeled robot, estimated from the wheel encoders.
#
# The velocity_* fields represent the linear and angular velocity of the robot.
# The pose_* fields represent the x, y and yaw pose of the robot w.r.t. the starting pose.
#
# The coordinate frame that represents the robot is located at the center of rotation.

builtin_interfaces/Time stamp
float32 velocity_lin
float32 velocity_ang
float32 pose_x
float32 pose_y
float32 pose_yaw
