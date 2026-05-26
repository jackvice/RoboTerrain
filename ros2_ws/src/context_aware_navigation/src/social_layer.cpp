#include "context_aware_navigation/social_layer.hpp"
#include <math.h>
#include "nav2_costmap_2d/costmap_math.hpp"
#include "nav2_costmap_2d/footprint.hpp"
#include "rclcpp/parameter_events_filter.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
using nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE;
using nav2_costmap_2d::LETHAL_OBSTACLE;
using nav2_costmap_2d::NO_INFORMATION;

namespace context_aware_navigation
{

    SocialLayer::SocialLayer(){}

    // This method is called at the end of plugin initialization.
    // It contains ROS parameter(s) declaration and initialization
    // of need_recalculation_ variable.
    void
    SocialLayer::onInitialize()
    {
        auto node = node_.lock();
        if (!node) {
            throw std::runtime_error{"Failed to lock node"};
        }

        declareParameter("enabled", rclcpp::ParameterValue(true));
        node->get_parameter(name_ + "." + "enabled", enabled_);

        base_frame = layered_costmap_->getGlobalFrameID();

        tf_buffer = std::make_unique<tf2_ros::Buffer>(node->get_clock());
        tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

        social_map_sub_ = node->create_subscription<sensor_msgs::msg::Image>(
        "/social_map", 10, std::bind(&SocialLayer::imageCallback, this, std::placeholders::_1));
        current_ = true;

    }

    // The method is called to ask the plugin: which area of costmap it needs to update.
    // Inside this method window bounds are re-calculated if need_recalculation_ is true
    // and updated independently on its value.
    void
    SocialLayer::updateBounds(
        double /*robot_x*/, double /*robot_y*/, double /*robot_yaw*/, double *min_x,
        double *min_y, double *max_x, double *max_y)
    {
        //figure out the zone where we want to draw the ellipse
        
        if (social_map_rotated.data != NULL)
        {
            float minx=0;
            float maxx=social_map_rotated.size().width;
            float miny=0;
            float maxy=social_map_rotated.size().height;

            if (minx < *min_x)
            {
                *min_x =minx;
            }
            if (maxx > *max_x)
            {
                *max_x =maxx;
            }
            if (miny < *min_y)
            {
                *min_y =miny;
            }
            if (maxy > *max_y)
            {
                *max_y =maxy;
            }

        }
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
    SocialLayer::imageCallback(sensor_msgs::msg::Image::ConstSharedPtr message)
    {
        auto node = node_.lock();
        if (!node) {
            throw std::runtime_error{"Failed to lock node"};
        }
        
        // convert iamge message
        try
        {
            // convert and update pointer with new image
            //  TODO the rotation from this is not critcal through time but the translation is so consider saving the timestamp of the message
            cv_ptr = cv_bridge::toCvCopy(message);
            
            std::string fromFrameRel = message->header.frame_id.c_str();
            std::string mapFrame = "map";
            geometry_msgs::msg::TransformStamped pastRobotPose;

            // get a pose of the social map in the map frame at time of recording by looking up the old tf_transform
            try
            {
                pastRobotPose = tf_buffer->lookupTransform(
                    mapFrame,
                    message->header.frame_id.c_str(), 
                    message->header.stamp);
                        
                recordedPose.position.x=pastRobotPose.transform.translation.x;
                recordedPose.position.y=pastRobotPose.transform.translation.y;
                recordedPose.position.z=pastRobotPose.transform.translation.z;
                recordedPose.orientation.x = 0;
                recordedPose.orientation.y = 0;
                recordedPose.orientation.z = 0;
                recordedPose.orientation.w = 1;
                recordedTime = message->header.stamp;
            }
            catch (const tf2::TransformException &ex)
            {
                return;
            }

            cv_ptr->image.convertTo(social_map, CV_8UC1);   //make sure this is uint8
            cv::flip(social_map,social_map,0);              //flip since image coordinate frame is downwards positive
            social_map.copyTo(social_map);                  //safe a copy of this to prevent segfault
        }
        catch (cv_bridge::Exception &e)
        {
            return;
        }
    }


    // The method is called when costmap recalculation is required.
    // It updates the costmap within its window bounds.
    // Inside this method the costmap gradient is generated and is writing directly
    // to the resulting costmap master_grid without any merging with previous layers.
    void
    SocialLayer::updateCosts(
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
        if (social_map.data != NULL){
            if (social_map.empty()!=true){
                uint8_t *costmap_array = getCharMap();
                std::string mapFrame = "map";
                geometry_msgs::msg::TransformStamped t;
                geometry_msgs::msg::Pose poseOut;

                // get transform between from the past robot in the map frame to the current one in the odom frame
                // this is done in order to use the plugin for either global or local costmaps, since the map and odom frame shift
                try
                {
                    t = tf_buffer->lookupTransform(
                        base_frame.c_str(),
                        node->get_clock()->now(),
                        mapFrame,
                        recordedTime,
                        mapFrame,                       //fixed frame is map
                        rclcpp::Duration(0,10000000));  //10ms timeout

                    //apply transform
                    tf2::doTransform(recordedPose,poseOut,t);



                    // get rpy and rotate Image
                    tf2::Quaternion q(
                        poseOut.orientation.x,
                        poseOut.orientation.y,
                        poseOut.orientation.z,
                        poseOut.orientation.w);
                    tf2::Matrix3x3 m(q);
                    double roll, pitch, yaw;
                    m.getRPY(roll, pitch, yaw);
                    // https://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c
                    //  get rotation matrix for rotating the image around its center in pixel coordinates
                    cv::Point2f center((social_map.cols - 1) / 2.0, (social_map.rows - 1) / 2.0);
                    cv::Mat rot = cv::getRotationMatrix2D(center, -yaw * (180 / M_PI), 1.0);
                    // determine bounding rectangle, center not relevant
                    cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), social_map.size(), -yaw * (180 / M_PI)).boundingRect2f();
                    // adjust transformation matrix
                    rot.at<double>(0, 2) += bbox.width / 2.0 - social_map.cols / 2.0;
                    rot.at<double>(1, 2) += bbox.height / 2.0 - social_map.rows / 2.0;
                    cv::warpAffine(social_map, social_map_rotated, rot, bbox.size());
                    for (int j = 0; j < social_map_rotated.rows; j++) {
                        for (int i = 0; i < social_map_rotated.cols; i++) {
                            try{
                                //get real coordinates after translation
                                double wx,wy;
                                uint mx,my;
                                //get the physical position of the pixel in the social map
                                wx=((i-social_map_rotated.rows/2)*getResolution())+poseOut.position.x;
                                wy=((j-social_map_rotated.cols/2)*getResolution())+poseOut.position.y;
                                if(worldToMap(wx,wy,mx,my))//translate the position
                                {
                                    int index = getIndex(mx, my);
                                    uint8_t cost= static_cast<uint8_t>(social_map_rotated.at<uchar>(j,i));
                                    costmap_array[index] =cost;
                                }
                            }
                            catch(std::exception &e){return;}//sometimes the writing fails and throws
                        }
                    }
                }
                catch (const tf2::TransformException &ex)
                {
                    return;
                }
                
            }

        }
        updateWithMax(master_grid, min_i, min_j, max_i, max_j);
        current_ = true;

    }
} // namespace nav2_gradient_costmap_plugin




// This is the macro allowing a nav2_gradient_costmap_plugin::GradientLayer class
// to be registered in order to be dynamically loadable of base type nav2_costmap_2d::Layer.
// Usually places in the end of cpp-file where the loadable class written.
#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(context_aware_navigation::SocialLayer, nav2_costmap_2d::Layer)