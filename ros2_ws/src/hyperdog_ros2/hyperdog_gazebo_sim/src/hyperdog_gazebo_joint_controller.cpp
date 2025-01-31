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

#include "chrono"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"


// using namespace std::chrono_literals;
using std::placeholders::_1;


class HyperDogGazeboJointCtrl : public rclcpp::Node
{   
    public:
        HyperDogGazeboJointCtrl()
        : Node("Hyperdog_gazebo_joint_ctrl_node")
        {
            subscription_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
                "hyperdog_jointController/commands", 30, std::bind(&HyperDogGazeboJointCtrl::topic_callback, this, _1));
            
            publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("gazebo_joint_controller/commands", 30);
        }
        
    
    private:
        void topic_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg_rx) const
        {   
            auto joint_angles = std_msgs::msg::Float64MultiArray();
            std::vector<double> angles[12];
            for(float ang : msg_rx->data){
                joint_angles.data.push_back(double(ang*M_PI/180));
            }
            publisher_->publish(joint_angles);
        } 
        rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr subscription_;
        rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr publisher_;    
};

int main(int argc, char* argv[]){
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<HyperDogGazeboJointCtrl>());
    rclcpp::shutdown();
    return 0;
}


 