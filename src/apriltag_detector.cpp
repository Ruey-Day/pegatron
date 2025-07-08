#include <memory>
#include <vector>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "geometry_msgs/msg/transform_stamped.hpp"

#include "tf2/LinearMath/Matrix3x3.h"      // <-- Added for rotation matrix conversion
#include "tf2/LinearMath/Quaternion.h"     // <-- Added for quaternion operations

#include "cv_bridge/cv_bridge.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "apriltag/apriltag.h"
#include "apriltag/tagStandard41h12.h"
#include "apriltag/common/matd.h"


class AprilTagDetector : public rclcpp::Node
{
public:
    AprilTagDetector() : Node("apriltag_detector")
    {
        // Initialize AprilTag detector
        tf = tagStandard41h12_create();
        td = apriltag_detector_create();
        apriltag_detector_add_family(td, tf);

        // Configure detector
        td->quad_decimate = this->declare_parameter("quad_decimate", 2.0);
        td->quad_sigma = this->declare_parameter("quad_sigma", 0.0);
        td->nthreads = this->declare_parameter("nthreads", 1);
        td->debug = this->declare_parameter("debug", 0);
        td->refine_edges = this->declare_parameter("refine_edges", 1);

        // Tag size in meters
        tag_size = this->declare_parameter("tag_size", 0.0922);

        // Create a message filter to synchronize image and camera_info
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/image_raw", 10, std::bind(&AprilTagDetector::imageCallback, this, std::placeholders::_1));
        
        camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera_info", 10, std::bind(&AprilTagDetector::cameraInfoCallback, this, std::placeholders::_1));

        // Create publishers
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        RCLCPP_INFO(this->get_logger(), "AprilTag detector node initialized");
    }

    ~AprilTagDetector()
    {
        // Cleanup.
        tagStandard41h12_destroy(tf);
        apriltag_detector_destroy(td);
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {   
        if (!camera_info_received_) {
            RCLCPP_INFO(this->get_logger(), "Waiting for camera info...");
            return;
        }

        try {   
            RCLCPP_INFO(this->get_logger(), "Processing image frame");
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "rgb8");
            cv::Mat color = cv_ptr->image;
            cv::Mat gray;
            cv::cvtColor(color, gray, cv::COLOR_RGB2GRAY);

            image_u8_t im{ 
                static_cast<int32_t>(gray.cols),   // width
                static_cast<int32_t>(gray.rows),   // height
                static_cast<int32_t>(gray.cols),   // stride (assuming a continuous image row)
                gray.data                          // pointer to image data
            };
            
            // Detect AprilTags
            zarray_t *detections = apriltag_detector_detect(td, &im);
            RCLCPP_INFO(this->get_logger(), "Number of detections: %d", zarray_size(detections));

            // Create camera matrix from camera info
            cv::Mat cameraMat = (cv::Mat_<double>(3, 3) << 
                    610.46296007, 0.0, 321.0857477,
                    0.0, 609.56569972, 256.30448812,
                    0.0, 0.0, 1.0);
            cv::Mat distCoeffs = (cv::Mat_<double>(1,5) << 0.04497951,  0.58119324,  0.00343644, -0.00335564, -2.43301257);

            // Define 3D model points for a square tag
            // These are the coordinates of the tag corners in the tag's local coordinate system
            // We assume the tag is centered at the origin, with side length equal to tag_size
            const double half_size = tag_size / 2.0;
            std::vector<cv::Point3d> objectPoints = {
                cv::Point3d(-half_size, half_size, 0.0),  // Bottom-left
                cv::Point3d( half_size, half_size, 0.0),  // Bottom-right
                cv::Point3d( half_size, -half_size, 0.0),  // Top-right
                cv::Point3d(-half_size, -half_size, 0.0)   // Top-left
            };

            for (int i = 0; i < zarray_size(detections); i++) {
                apriltag_detection_t *det;
                zarray_get(detections, i, &det);
                
                // Extract the tag corners as image points
                std::vector<cv::Point2d> imagePoints = {
                    cv::Point2d(det->p[0][0], det->p[0][1]),  // Bottom-left
                    cv::Point2d(det->p[1][0], det->p[1][1]),  // Bottom-right
                    cv::Point2d(det->p[2][0], det->p[2][1]),  // Top-right
                    cv::Point2d(det->p[3][0], det->p[3][1])   // Top-left
                };

                // Use OpenCV's PnP solver with SOLVEPNP_IPPE_SQUARE method
                cv::Mat rvec, tvec;
                bool success = cv::solvePnP(objectPoints, imagePoints, cameraMat, distCoeffs, 
                                           rvec, tvec, false, cv::SOLVEPNP_IPPE_SQUARE);
                
                if (success) {
                    RCLCPP_INFO(this->get_logger(), "Found pose for tag ID: %d", det->id);
                    
                    // Convert rotation vector to rotation matrix
                    cv::Mat rotMat;
                    cv::Rodrigues(rvec, rotMat);
                    
                    // Debug: Print the rotation matrix and translation vector
                    RCLCPP_INFO(this->get_logger(), "Tag %d rotation vector: [%f, %f, %f]", 
                                det->id, rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2));
                    RCLCPP_INFO(this->get_logger(), "Tag %d translation vector: [%f, %f, %f]", 
                                det->id, tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

                    //usleep(2000000);
                    
                    // Publish the transform
                    publishTagTransform(det->id, rotMat, tvec, msg->header.stamp);
                } else {
                    RCLCPP_WARN(this->get_logger(), "Failed to estimate pose for tag ID: %d", det->id);
                }
            }
            
            apriltag_detections_destroy(detections);
        }
        catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "CV bridge exception: %s", e.what());
        }
        catch (std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Exception: %s", e.what());
        }
    }

    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        if (!camera_info_received_) {
            camera_info_K = msg->k;
            camera_info_D = msg->d;
            camera_info_received_ = true;
            RCLCPP_INFO(this->get_logger(), "Camera info received");
        }
    }

    void publishTagTransform(int tag_id, const cv::Mat &rotMat, const cv::Mat &tvec, const rclcpp::Time &stamp)
    {
        geometry_msgs::msg::TransformStamped transformStamped;
        transformStamped.header.stamp = stamp;
        // Set the parent frame (adjust if needed)
        transformStamped.header.frame_id = "camera_color_frame"; // <-- You may change this as needed
        transformStamped.child_frame_id = "tag_" + std::to_string(tag_id);

        // Set translation from the tvec (assuming tvec is of type CV_64F)
        transformStamped.transform.translation.x = tvec.at<double>(0);
        transformStamped.transform.translation.y = tvec.at<double>(1);
        transformStamped.transform.translation.z = tvec.at<double>(2);

        // Convert the OpenCV rotation matrix to a tf2::Quaternion.
        tf2::Matrix3x3 tf3d(
            rotMat.at<double>(0, 0), rotMat.at<double>(0, 1), rotMat.at<double>(0, 2),
            rotMat.at<double>(1, 0), rotMat.at<double>(1, 1), rotMat.at<double>(1, 2),
            rotMat.at<double>(2, 0), rotMat.at<double>(2, 1), rotMat.at<double>(2, 2)
        );
        tf2::Quaternion q;
        tf3d.getRotation(q);

        transformStamped.transform.rotation.x = q.x();
        transformStamped.transform.rotation.y = q.y();
        transformStamped.transform.rotation.z = q.z();
        transformStamped.transform.rotation.w = q.w();

        // Publish the transform
        tf_broadcaster_->sendTransform(transformStamped);
    }
    // AprilTag detector
    apriltag_family_t *tf = tagStandard41h12_create();
    apriltag_detector_t *td = apriltag_detector_create();
    double tag_size;

    // Camera info
    std::array<double, 9> camera_info_K;
    std::vector<double> camera_info_D;
    bool camera_info_received_ = false;

    // ROS subscriptions and publishers
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr twist_pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<AprilTagDetector>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}