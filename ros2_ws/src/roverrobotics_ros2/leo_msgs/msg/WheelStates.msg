# This message describes the states of the wheels in Leo Rover
#
# The state of each wheel is defined by:
#  * the position of the wheel (rad),
#  * the velocity of the wheel (rad/s)
#  * the torque that is applied in the wheel (Nm)
#  * the PWM Duty cycle (%)
#
# The stamp specifies the time at which the wheel states were recorded.
#
# This message consists of a multiple arrays, one for each part of the wheel state.
# The order of the wheels in each array is: FL, RL, FR, RR

builtin_interfaces/Time stamp

float32[4] position
float32[4] velocity
float32[4] torque
float32[4] pwm_duty_cycle
