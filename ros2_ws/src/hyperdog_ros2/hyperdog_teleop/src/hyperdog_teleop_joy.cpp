// __________________________________________________________________________________
// MIT License                                                                       |
//                                                                                   |
// Copyright (c) 2024 W.M. Nipun Dhananjaya Weerakkodi                               |
//                                                                                   | 
// Permission is hereby granted, free of charge, to any person obtaining a copy      |
// of this software and associated documentation files (the "Software"), to deal     |
// in the Software without restriction, including without limitation the rights      |
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell         |
// copies of the Software, and to permit persons to whom the Software is             |
// furnished to do so, subject to the following conditions:                          |
//                                                                                   |
// The above copyright notice and this permission notice shall be included in all    |
// copies or substantial portions of the Software.                                   |
//                                                                                   |
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR        |
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,          |
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE       |
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER            |
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,     |
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE     |
// SOFTWARE.                                                                         |
// __________________________________________________________________________________|
#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>


#include <array>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>

#include "std_msgs/msg/string.hpp"
#include "chrono"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"

#include "hyperdog_msgs/msg/joy_ctrl_cmds.hpp"
// #include "hyperdog_msgs/msg/ctrl_variables.hpp"

#include "hyperdog_definitions.hpp"
// #include "hyperdog_ctrl_variables.hpp"


using std::placeholders::_1; 
// ;
 
int t1, t2, t3 = clock();
int btn_tgl_delay = 3000;

auto cmd = hyperdog_msgs::msg::JoyCtrlCmds();

bool LJ_btn_sw = 0;

class PublishingSubscriber : public rclcpp::Node
{
  public: 
    PublishingSubscriber() 
    : Node("hyperdog_teleop_gamepad_node")
    {
      subscription_ = this->create_subscription<sensor_msgs::msg::Joy>(
        "joy", 12, std::bind(&PublishingSubscriber::topic_callback, this, _1));

      publisher_ = this->create_publisher<hyperdog_msgs::msg::JoyCtrlCmds>("hyperdog_joy_ctrl_cmd",40);
    }

  
    void joy_state_to_joy_cmd(sensor_msgs::msg::Joy::SharedPtr msg_joy)
    { 
      // set start <- btn: start
      if (!cmd.states[0] && msg_joy->buttons[7] && (clock() - t1 > btn_tgl_delay)){
        cmd.states[0] = true;
        t1 = clock();
      }
      else if (cmd.states[0] && msg_joy->buttons[7] && (clock() - t1 > btn_tgl_delay )){
        cmd.states[0] = false;
        t1 = clock();
      }

      if (cmd.states[0]){
        // set walk <- btn: back
        if (!cmd.states[1] && msg_joy->buttons[6] && (clock() - t2 > btn_tgl_delay)){
          cmd.states[1] = true;
          t2 = clock();
        }
        else if (cmd.states[1] && msg_joy->buttons[6] && (clock() - t2 > btn_tgl_delay )){
          cmd.states[1] = false;
          t2 = clock();
        }

        // change side_walk_mode <- btn: RJ btn
        if (!cmd.states[2] && msg_joy->buttons[10] && (clock() - t3 > btn_tgl_delay)){
          cmd.states[2] = true;
          t3 = clock();
        }
        else if (cmd.states[2] && msg_joy->buttons[10] && (clock() - t3 > btn_tgl_delay )){
          cmd.states[2] = false;
          t3 = clock();
        }

        // select gait <- btn: A, B, X, Y
        if(msg_joy->buttons[0]){
          cmd.gait_type = 0;
        }
        if(msg_joy->buttons[1]){
          cmd.gait_type  = 1;
        }
        if(msg_joy->buttons[2]){
          cmd.gait_type  = 2;
        }
        if(msg_joy->buttons[3]){
          cmd.gait_type = 3;
        }

        // set robot height
        if (cmd.pose.position.z < MIN_HEIGHT)
          cmd.pose.position.z = MIN_HEIGHT;
        if(cmd.pose.position.z > MAX_HEIGHT)
          cmd.pose.position.z = MAX_HEIGHT;
        if(!msg_joy->buttons[4] && msg_joy->axes[7] > 0 && cmd.pose.position.z < MAX_HEIGHT ){
          cmd.pose.position.z += 5;
        }
        if(!msg_joy->buttons[4] && msg_joy->axes[7] < 0 && cmd.pose.position.z > MIN_HEIGHT ){
          cmd.pose.position.z -= 5;
        }

        // if walking mode is on and robot's height is not enough for walking, 
        // increase robot's height and swing step height
        if (cmd.states[1] == true && cmd.pose.position.z < 100){
          cmd.pose.position.z += 5;
          cmd.gait_step.z += 5;
        }

        // set step height
        if (cmd.gait_step.z > cmd.pose.position.z - MIN_HEIGHT)
          cmd.gait_step.z = cmd.pose.position.z - MIN_HEIGHT;
        if(msg_joy->buttons[4] && msg_joy->axes[7] > 0 && cmd.gait_step.z < cmd.pose.position.z - MIN_HEIGHT){      
          cmd.gait_step.z += 5;
        }
        if(msg_joy->buttons[4] && msg_joy->axes[7] < 0 && cmd.gait_step.z > 10 ){      
          cmd.gait_step.z -= 5;
        }

        // set eular angles
        if (!msg_joy->buttons[9] && !LJ_btn_sw){
          cmd.pose.orientation.x = -msg_joy->axes[0] * ROLL_RANGE;
          cmd.pose.orientation.y = msg_joy->axes[1] * PITCH_RANGE;
          cmd.pose.orientation.z = (msg_joy->axes[2] - msg_joy->axes[5])/2 * YAW_RANGE;
        }

        // set step length x y
        cmd.gait_step.x = msg_joy->axes[4] * MAX_STEP_LENGTH_X;
        cmd.gait_step.y = -msg_joy->axes[3] * MAX_STEP_LENGTH_Y;

        // set slant
        if (msg_joy->buttons[9]){
          LJ_btn_sw = 1;
          if (cmd.pose.position.x <= SLANT_X_MAX && cmd.pose.position.x >= SLANT_X_MIN){
            cmd.pose.position.x += msg_joy->axes[1]*5 ;
          }
          else if (cmd.pose.position.x < SLANT_X_MIN){
            cmd.pose.position.x = SLANT_X_MIN;
          } 
          else{
            cmd.pose.position.x = SLANT_X_MAX;
          }
          if (cmd.pose.position.y <= SLANT_Y_MAX && cmd.pose.position.y >= SLANT_Y_MIN){
            cmd.pose.position.y -= msg_joy->axes[0]*5 ;
          }
          else if (cmd.pose.position.y < SLANT_Y_MIN){
            cmd.pose.position.y = SLANT_Y_MIN;
          } 
          else{
            cmd.pose.position.y = SLANT_Y_MAX;
          }
        }
        if (msg_joy->axes[0] == 0 && msg_joy->axes[1] == 0){
          LJ_btn_sw = 0;
        }
        
      }
      
    }


  private:
    void topic_callback(const sensor_msgs::msg::Joy::SharedPtr msg_rx) 
    {
      joy_state_to_joy_cmd(msg_rx);

      publisher_ -> publish(cmd);

      // RCLCPP_INFO(this->get_logger(), "start: '%i'  | walk: %i  | R: %i  | P: %i | Y: %i  | Lx: %i  | Ly: %i  | Height: %i  | stepH: %i" , 
      //                 rob_ctrlVar.state.start, rob_ctrlVar.state.walk,  rob_ctrlVar.body.eularAng[0], rob_ctrlVar.body.eularAng[1], rob_ctrlVar.body.eularAng[2],
      //                 rob_ctrlVar.gaitParam.L[0], rob_ctrlVar.gaitParam.L[1], rob_ctrlVar.body.height, rob_ctrlVar.gaitParam.stepUp_h);
    }
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr subscription_;
    rclcpp::Publisher<hyperdog_msgs::msg::JoyCtrlCmds>::SharedPtr publisher_;
};


int main(int argc, char * argv[])
{
  
  rclcpp::init(argc, argv);  // Initialize ROS 2
  rclcpp::spin(std::make_shared<PublishingSubscriber>());  // Start processing data from the node as well as the callbacks
  rclcpp::shutdown(); // Shutdown the node when finished
  return 0;
}
