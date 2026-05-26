#include "context_aware_navigation/interaction_layer.hpp"
#include <math.h>
#include "nav2_costmap_2d/costmap_math.hpp"
#include "nav2_costmap_2d/footprint.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "rclcpp/parameter_events_filter.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
using nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE;
using nav2_costmap_2d::LETHAL_OBSTACLE;
using nav2_costmap_2d::NO_INFORMATION;

namespace context_aware_navigation
{

    InteractionLayer::InteractionLayer(){}

    // This method is called at the end of plugin initialization.
    // It contains ROS parameter(s) declaration and initialization
    // of need_recalculation_ variable.
    void
    InteractionLayer::onInitialize()
    {
        auto node = node_.lock();
        if (!node) {
            throw std::runtime_error{"Failed to lock node"};
        }
        declareParameter("enabled", rclcpp::ParameterValue(true));
        declareParameter("interaction_cost", rclcpp::ParameterValue(150.0));
        declareParameter("keeptime", rclcpp::ParameterValue(2.0));
        node->get_parameter(name_ + "." + "enabled", enabled_);
        node->get_parameter(name_ + "." + "interaction_cost", interaction_cost);
        node->get_parameter(name_ + "." + "keeptime", keeptime);

        tf_buffer = std::make_unique<tf2_ros::Buffer>(node->get_clock());
        tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);
        interaction_sub_ = node->create_subscription<multi_person_tracker_interfaces::msg::BoundingBoxes>(
        "/interaction_bb", 10, std::bind(&InteractionLayer::interactionCallback, this, std::placeholders::_1));
        current_ = true;
    }

    // The method is called to ask the plugin: which area of costmap it needs to update.
    // Inside this method window bounds are re-calculated if need_recalculation_ is true
    // and updated independently on its value.
    void
    InteractionLayer::updateBounds(
        double /*robot_x*/, double /*robot_y*/, double /*robot_yaw*/, double *min_x,
        double *min_y, double *max_x, double *max_y)
    {
        *min_x = -std::numeric_limits<float>::max();
        *min_y = -std::numeric_limits<float>::max();
        *max_x = std::numeric_limits<float>::max();
        *max_y = std::numeric_limits<float>::max();

        auto node = node_.lock();
        if (!node) {
            throw std::runtime_error{"Failed to lock node"};
        }

        RCLCPP_DEBUG(
            node->get_logger(),
            "Update bounds: Min x: %f Min y: %f Max x: %f Max y: %f", *min_x,
            *min_y, *max_x, *max_y);
            }
    void
    InteractionLayer::interactionCallback(
      multi_person_tracker_interfaces::msg::BoundingBoxes::ConstSharedPtr message)
    {
        auto node = node_.lock();
        if (!node) {
            throw std::runtime_error{"Failed to lock node"};
        }
        BoundingBoxes=message;
    }

    // The method is called when costmap recalculation is required.
    // It updates the costmap within its window bounds.
    // Inside this method the costmap gradient is generated and is writing directly
    // to the resulting costmap master_grid without any merging with previous layers.
    void
    InteractionLayer::updateCosts(
        nav2_costmap_2d::Costmap2D &master_grid, int min_i, int min_j,
        int max_i,
        int max_j)
    {
        if (!enabled_)
        {
            return;
        }
        auto node = node_.lock();
        if (!node)
        {
            throw std::runtime_error{"Failed to lock node"};
        }
        setDefaultValue(nav2_costmap_2d::NO_INFORMATION);
        matchSize();
        uint8_t *costmap_array = getCharMap();
        unsigned int size_x = getSizeInCellsX(), size_y = getSizeInCellsY();
        min_i = std::max(0, min_i);
        min_j = std::max(0, min_j);
        max_i = std::min(static_cast<int>(size_x), max_i);
        max_j = std::min(static_cast<int>(size_y), max_j);
        geometry_msgs::msg::PoseStamped poseIn, poseOut;
        // center of costmap layer
        std::string global_frame_ = layered_costmap_->getGlobalFrameID();
        int x1,y1;
        if (BoundingBoxes != nullptr)
        {
            //only write ellipses that are not too old
            if(node->now().seconds()-BoundingBoxes->header.stamp.sec<keeptime){
                //create stamped pose of the bounding box in the message frame (most likely map)
                // poseIn.header=BoundingBox->header;
                // poseIn.pose.position.x=BoundingBox->center_x;
                // poseIn.pose.position.y=BoundingBox->center_y;
                // poseIn.pose.position.z=0;
                tf2::Quaternion q;
                // q.setRPY(0,0,0);//bounding boxes are never rotated
                // q=q.normalize();
                // poseIn.pose.orientation.x=q.x();
                // poseIn.pose.orientation.y=q.y();
                // poseIn.pose.orientation.z=q.z();
                // poseIn.pose.orientation.w=q.w();
                geometry_msgs::msg::TransformStamped t;
                //transform the pose from the message frame into the global frame of the costmap
                try {
                    // tf_buffer->transform(poseIn,poseOut,global_frame_,tf2::durationFromSec(5.0));
                    t = tf_buffer->lookupTransform(global_frame_,node->now(),BoundingBoxes->header.frame_id,BoundingBoxes->header.stamp,"map");

                } catch (const tf2::TransformException & ex) {
                RCLCPP_INFO(
                    node->get_logger(), "Could not transform %s to %s: %s",
                    global_frame_.c_str(), BoundingBoxes->header.frame_id.c_str(), ex.what());
                return;
                }
                q = tf2::Quaternion(t.transform.rotation.x,
                        t.transform.rotation.y,
                        t.transform.rotation.z,
                        t.transform.rotation.w);
                tf2::Matrix3x3 m(q);
                double roll, pitch, yaw;
                m.getRPY(roll, pitch, yaw);
                cv::Mat cv_costmap = cv::Mat(size_y,size_x,CV_8UC1, costmap_array);//make a cv::Mat from the current costmap
                for (int i=0;i<static_cast<int>(BoundingBoxes->boundingboxes.size());i++){
                //get coordinates of the center without bounding it so we can draw an ellipse outside of bounds of the cv::Mat
                    worldToMapNoBounds(BoundingBoxes->boundingboxes[i].center_x+t.transform.translation.x,BoundingBoxes->boundingboxes[i].center_y+t.transform.translation.y,x1,y1);
                    //draw ellipse into cv::Mat
                    cv::Point center = cv::Point(x1,y1);
                    cv::Size size = cv::Size(static_cast <int> (std::floor(BoundingBoxes->boundingboxes[i].width/(2*resolution_))),static_cast <int> (std::floor(BoundingBoxes->boundingboxes[i].height/(2*resolution_))));
                    cv::ellipse(cv_costmap,center,size,yaw*(180/M_PI),0,360,interaction_cost,-1);//draw interaction as ellipseat the correct angle
                }
                //write the array to the costmap
                costmap_array=cv_costmap.data;
                updateWithMax(master_grid, min_i, min_j, max_i, max_j); //update with max
                current_ = true;
            }
        }
    }
} // namespace nav2_gradient_costmap_plugin

// This is the macro allowing a nav2_gradient_costmap_plugin::GradientLayer class
// to be registered in order to be dynamically loadable of base type nav2_costmap_2d::Layer.
// Usually places in the end of cpp-file where the loadable class written.
#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(context_aware_navigation::InteractionLayer, nav2_costmap_2d::Layer)
