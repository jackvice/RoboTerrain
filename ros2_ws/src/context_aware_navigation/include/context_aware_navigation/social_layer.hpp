#ifndef SOCIAL_LAYER_HPP_
#define SOCIAL_LAYER_HPP_

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"

#include "nav2_costmap_2d/layer.hpp"
#include "nav2_costmap_2d/layered_costmap.hpp"
#include "nav2_costmap_2d/costmap_layer.hpp"

#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "opencv4/opencv2/opencv.hpp"
#include "cv_bridge/cv_bridge.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"

namespace context_aware_navigation
{

class SocialLayer : public nav2_costmap_2d::CostmapLayer
{
public:
  SocialLayer();

  void onInitialize();
  void updateBounds(
    double robot_x, double robot_y, double robot_yaw, double * min_x,
    double * min_y,
    double * max_x,
    double * max_y);
  void updateCosts(
    nav2_costmap_2d::Costmap2D & master_grid,
    int min_i, int min_j, int max_i, int max_j) override;

  void reset() override{}

  bool isClearable() override {return false;}

  void imageCallback(
      sensor_msgs::msg::Image::ConstSharedPtr message);



private:
  double last_min_x_, last_min_y_, last_max_x_, last_max_y_;

  bool need_recalculation_;
  std::string target_frame;
  std::string base_frame;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener;

  geometry_msgs::msg::Pose recordedPose;
  builtin_interfaces::msg::Time recordedTime;
  cv_bridge::CvImagePtr cv_ptr;
  
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr social_map_sub_;
  cv::Mat social_map_rotated;
  cv::Mat social_map;
};

}  // namespace context_aware_navigation

#endif  // SOCIAL_LAYER_HPP_