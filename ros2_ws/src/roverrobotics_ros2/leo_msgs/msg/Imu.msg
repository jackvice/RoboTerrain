# This message holds the data retrieved from an Accel/Gyro+Temp IMU sensor
#
# The temperature field represents the temperature reported by the sensor in Degrees Celcius
# The gyro_* fields represent the rotational velocity in rad/s
# The accel_* fields represent the linear acceleration in m/s^2

builtin_interfaces/Time stamp
float32 temperature
float32 gyro_x
float32 gyro_y
float32 gyro_z
float32 accel_x
float32 accel_y
float32 accel_z
