# __________________________________________________________________________________
# MIT License                                                                       |
#                                                                                   |
# Copyright (c) 2024 W.M. Nipun Dhananjaya Weerakkodi                               |
#                                                                                   | 
# Permission is hereby granted, free of charge, to any person obtaining a copy      |
# of this software and associated documentation files (the "Software"), to deal     |
# in the Software without restriction, including without limitation the rights      |
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell         |
# copies of the Software, and to permit persons to whom the Software is             |
# furnished to do so, subject to the following conditions:                          |
#                                                                                   |
# The above copyright notice and this permission notice shall be included in all    |
# copies or substantial portions of the Software.                                   |
#                                                                                   |
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR        |
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,          |
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE       |
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER            |
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,     |
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE     |
# SOFTWARE.                                                                         |
# __________________________________________________________________________________|

from cmd_manager.cmd_manager_node import CmdManager_ROS 
from cmd_manager.hyperdog_variables import Body, Leg, Cmds
from body_motion_planner.body_motion_planner import BodyMotionPlanner
from gait_generator.gait_planner import GaitPlanner
from multiprocessing import Process
import threading
import time
import rclpy

leg = Leg()
body = Body()
cmd = Cmds()


# lilnk  objects
# leg.foot_zero_pnt[:,2] = body.height  = cmd.body.height
cmd.gait.cycle_time = 0.8
cmd.gait.swing_time = 0.2
cmd.leg.foot_zero_pnt[0,0] = -10
cmd.leg.foot_zero_pnt[1,0] = -10
cmd.leg.foot_zero_pnt[2,0] = -70
cmd.leg.foot_zero_pnt[3,0] = -70
cmd.gait.stance_step_h = 0

cmd_manager = CmdManager_ROS(set_msgs=cmd, send_msgs=[leg, body])
gait_planner = GaitPlanner(cmd, leg, body)
bmp = BodyMotionPlanner(cmd, leg, body, gait_planner)

gait_planner.len_zmp_wavegait = 50

def main(args=None):
    print('starting')

    thread_cmd_manager = threading.Thread(target=cmd_manager.start)
    thread_gait_planner = threading.Thread(target=gait_planner.run)
    thread_bmp = threading.Thread(target=bmp.run)
    
    # thread_cmd_manager = Process(target=cmd_manager.start)
    # thread_gait_planner = Process(target=gait_planner.run)


    bmp.set_init_pose()

    try:
        thread_cmd_manager.start()
        thread_bmp.start()
        thread_gait_planner.start()
        
        while 1:
            print("tnumber of threads in background: {}".format(threading.active_count()))
            # print("current thread: {}\n".format(threading.current_thread().name))
            # print(cmd.body.height, '----', leg.FR.pose.cur_coord)
            time.sleep(1)
            # print(cmd.mode.start)
            

    except  KeyboardInterrupt:
        cmd_manager.node.destroy_node()
        rclpy.shutdown()
        


if __name__ == '__main__':
    main()
    