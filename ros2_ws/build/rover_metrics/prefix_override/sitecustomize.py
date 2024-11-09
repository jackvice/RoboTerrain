import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/jack/src/RoboTerrain/ros2_ws/install/rover_metrics'
