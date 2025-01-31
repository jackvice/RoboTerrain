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

from matplotlib.cbook import file_requires_unicode
import numpy as np




MIN_LEG_HEIGHT = 80
MAX_LEG_HEIGHT = 240
MAX_ROLL       = 45
MAX_PITCH      = 45
MAX_YAW        = 50 



class RobotState():
    start = False
    walk  = False
    side_move_mode = 0
    height = MIN_LEG_HEIGHT
    eular_ang = [0, 0, 0]
    zeroPnts = [[0,0], [0,0], [0,0], [0,0]]
    speed = 0

class GaitParameters():
    step_length = [0, 0]
    step_height = 0
    step_depth = 0
    freq       = 0
    swing_time = 0
    start_pnt  = [0, 0]
    end_pnt    = [0, 0]

class LegParameters():
    zeroPnt = [0,0]
    cur_pose = [0, 0, 0]
    target_pose = [0,0, 0]
    z_err    = 0

class LegsParameters():
    FR = LegParameters()
    FL = LegParameters()
    BR = LegParameters()
    BL = LegParameters()

class JointAngles():
    cur_angle    = [0, 0, 0]
    target_angle = [0, 0, 0]




        
        



