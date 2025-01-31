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

import numpy as np

class Cmds():
    class _mode():
        start = False
        walk = False
        side_walk_mode = 0
        gait_type = 0
    # -----------------
    class _body():
        height = 80
        roll = 0
        pitch = 0
        yaw = 0
        slant = np.zeros([3])
   
    # -----------------
    class _leg():
        foot_zero_pnt = np.zeros([4,3]) # [FR,FL,BR,BL][x,y]
    # ------------------
    class _gait():
        step_len = np.zeros([2]) # [len_x,len_y]
        swing_step_h = 0
        stance_step_h = 0
        cycle_time = 0
        swing_time = 0

    mode = _mode()
    body = _body()
    leg = _leg()
    gait = _gait()


#============================================================================ 
        
class Body():
    def __init__(self):
        self.height = None
        self.centerOfMass = np.zeros([3])
        self.physical = self._physiacal_params()
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.ZMP_handler = np.zeros([4,3])
    class _physiacal_params():
        _length = 300
        _width = 172
        _min_height = 80
        _max_height = 240
     


#============================================================================

class _LegParams:
    def __init__(self):
        self.pose = self.leg_pose()
        self.gait = self.gait_params()    
    class gait_params:
        def __init__(self):
            self.cycle_time = None
            self.step_len = np.zeros([3])
            self.stance = self.params()
            self.swing = self.params()
            self.traj_pnt = np.zeros([3]) # [x,y,z]
        class params:
            def __init__(self):
                self.start = False
                self.time = None
                self.start_pnt = np.zeros([3])
                self.end_pnt = np.zeros([3])


    class leg_pose:
        def __init__(self):
            self.zero_pnt = np.zeros([3])
            self.cur_coord = np.zeros([3])
            self.origin_2_endEfector_dist = np.sqrt(self.cur_coord[0]**2 + self.cur_coord[1]**2 + self.cur_coord[2]**2)

 # public
class Leg:
    def __init__(self):
        self.FR = _LegParams()
        self.FL = _LegParams()
        self.BR = _LegParams()
        self.BL = _LegParams()
        self.physical = self._physical_params()
    class _physical_params():
        _L1 = 104 # mm
        _L2 = 150 # mm
        _L3 = 150 # mm



#============================================================================