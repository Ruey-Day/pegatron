#include <memory>
#include <vector>
#include <string>
#include <map>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/static_transform_broadcaster.h"
#include "geometry_msgs/msg/transform_stamped.hpp"

#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Transform.h"
#include "geometry_msgs/msg/pose.hpp"

#include "cv_bridge/cv_bridge.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "apriltag/apriltag.h"
#include "apriltag/tagStandard41h12.h"
#include "apriltag/common/matd.h"

struct TagPose {
    double x, y, z;
    double qx, qy, qz, qw;
};

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
        tag_size = this->declare_parameter("tag_size", 0.0556);

        // Initialize fixed tag poses (you'll need to set these according to your setup)
        initializeTagPoses();

        // Create a message filter to synchronize image and camera_info
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "camera/camera/color/image_raw", 10, std::bind(&AprilTagDetector::imageCallback, this, std::placeholders::_1));
        
        camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera/camera/color/camera_info", 10, std::bind(&AprilTagDetector::cameraInfoCallback, this, std::placeholders::_1));
        
        // Create publishers
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        static_tf_broadcaster_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);

        // Publish static transforms for the fixed tag poses
        publishStaticTagTransforms();

        RCLCPP_INFO(this->get_logger(), "AprilTag camera localization node initialized");
    }

    ~AprilTagDetector()
    {
        // Cleanup.
        tagStandard41h12_destroy(tf);
        apriltag_detector_destroy(td);
    }

private:
    void initializeTagPoses()
    {
        // Define your fixed tag poses here
        // Example: Tag 0 at origin, Tag 1 at (1, 0, 0), etc.
        // You'll need to measure and set these according to your actual setup
        
        // Example configuration - replace with your actual tag positions
        fixed_tag_poses_[0] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};  // Tag 0 at origin
        fixed_tag_poses_[1] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};  // Tag 1 at (1,0,0)
        // Add more tags as needed
        
        RCLCPP_INFO(this->get_logger(), "Initialized %zu fixed tag poses", fixed_tag_poses_.size());
    }

    void publishStaticTagTransforms()
    {
        std::vector<geometry_msgs::msg::TransformStamped> static_transforms;
        
        for (const auto& [tag_id, pose] : fixed_tag_poses_) {
            geometry_msgs::msg::TransformStamped static_transform;
            static_transform.header.stamp = this->get_clock()->now();
            static_transform.header.frame_id = "world";  // or "map" - your world frame
            static_transform.child_frame_id = "tag_" + std::to_string(tag_id) + "_fixed";
            
            static_transform.transform.translation.x = pose.x;
            static_transform.transform.translation.y = pose.y;
            static_transform.transform.translation.z = pose.z;
            static_transform.transform.rotation.x = pose.qx;
            static_transform.transform.rotation.y = pose.qy;
            static_transform.transform.rotation.z = pose.qz;
            static_transform.transform.rotation.w = pose.qw;
            
            static_transforms.push_back(static_transform);
        }
        
        static_tf_broadcaster_->sendTransform(static_transforms);
        RCLCPP_INFO(this->get_logger(), "Published static transforms for %zu tags", static_transforms.size());
    }

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
                camera_info_K[0], 0.0, camera_info_K[2],
                0.0, camera_info_K[4], camera_info_K[5],
                0.0, 0.0, 1.0);
            cv::Mat distCoeffs = cv::Mat(camera_info_D);

            // Define 3D model points for a square tag
            const double half_size = tag_size / 2.0;
            std::vector<cv::Point3d> objectPoints = {
                cv::Point3d(-half_size, half_size, 0.0),  // Bottom-left
                cv::Point3d( half_size, half_size, 0.0),  // Bottom-right
                cv::Point3d( half_size, -half_size, 0.0),  // Top-right
                cv::Point3d(-half_size, -half_size, 0.0)   // Top-left
            };

            // Store camera poses from all detected tags for potential fusion
            std::vector<tf2::Transform> camera_poses;
            std::vector<int> detected_tag_ids;

            for (int i = 0; i < zarray_size(detections); i++) {
                apriltag_detection_t *det;
                zarray_get(detections, i, &det);
                
                // Check if this tag has a known fixed pose
                if (fixed_tag_poses_.find(det->id) == fixed_tag_poses_.end()) {
                    RCLCPP_WARN(this->get_logger(), "Tag ID %d not found in fixed poses, skipping", det->id);
                    continue;
                }
                
                // Extract the tag corners as image points
                std::vector<cv::Point2d> imagePoints = {
                    cv::Point2d(det->p[0][0], det->p[0][1]),  // Bottom-left
                    cv::Point2d(det->p[1][0], det->p[1][1]),  // Bottom-right
                    cv::Point2d(det->p[2][0], det->p[2][1]),  // Top-right
                    cv::Point2d(det->p[3][0], det->p[3][1])   // Top-left
                };

                // Use OpenCV's PnP solver to get tag pose relative to camera
                cv::Mat rvec, tvec;
                bool success = cv::solvePnP(objectPoints, imagePoints, cameraMat, distCoeffs, 
                                           rvec, tvec, false, cv::SOLVEPNP_IPPE_SQUARE);
                if (success) {
                    RCLCPP_INFO(this->get_logger(), "Found camera pose using tag ID: %d", det->id);
                    
                    // Convert rotation vector to rotation matrix
                    cv::Mat rotMat;
                    cv::Rodrigues(rvec, rotMat);
                    
                    // Calculate camera pose relative to the fixed tag
                    tf2::Transform camera_pose = calculateCameraPose(det->id, rotMat, tvec);
                    camera_poses.push_back(camera_pose);
                    detected_tag_ids.push_back(det->id);
                    
                    // Publish individual camera pose from this tag
                    publishCameraPose(det->id, rotMat, tvec, msg->header.stamp);
                } else {
                    RCLCPP_WARN(this->get_logger(), "Failed to estimate pose for tag ID: %d", det->id);
                }
            }
            
            // If multiple tags detected, publish a fused camera pose
            if (camera_poses.size() > 1) {
                publishFusedCameraPose(camera_poses, detected_tag_ids, msg->header.stamp);
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

    tf2::Transform calculateCameraPose(int tag_id, const cv::Mat &rotMat, const cv::Mat &tvec)
    {
        // The solvePnP gives us the transformation from tag to camera
        // We need to invert this to get camera position relative to tag
        
        // Create tf2::Transform from the tag-to-camera transformation
        tf2::Matrix3x3 tag_to_cam_rot(
            rotMat.at<double>(0, 0), rotMat.at<double>(0, 1), rotMat.at<double>(0, 2),
            rotMat.at<double>(1, 0), rotMat.at<double>(1, 1), rotMat.at<double>(1, 2),
            rotMat.at<double>(2, 0), rotMat.at<double>(2, 1), rotMat.at<double>(2, 2)
        );
        
        tf2::Vector3 tag_to_cam_trans(
            tvec.at<double>(0),
            tvec.at<double>(1),
            tvec.at<double>(2)
        );
        
        tf2::Transform tag_to_camera(tag_to_cam_rot, tag_to_cam_trans);
        
        // Invert to get camera-to-tag transformation
        tf2::Transform camera_to_tag = tag_to_camera.inverse();
        
        // Get the fixed tag pose from world frame
        const TagPose& fixed_tag_pose = fixed_tag_poses_[tag_id];
        
        // Create transform from world to fixed tag
        tf2::Quaternion tag_quat(fixed_tag_pose.qx, fixed_tag_pose.qy, fixed_tag_pose.qz, fixed_tag_pose.qw);
        tf2::Vector3 tag_pos(fixed_tag_pose.x, fixed_tag_pose.y, fixed_tag_pose.z);
        tf2::Transform world_to_tag(tag_quat, tag_pos);
        
        // Calculate camera pose in world frame: world_to_camera = world_to_tag * camera_to_tag
        return world_to_tag * camera_to_tag;
    }

    void publishCameraPose(int tag_id, const cv::Mat &rotMat, const cv::Mat &tvec, const rclcpp::Time &stamp)
    {
        tf2::Transform world_to_camera = calculateCameraPose(tag_id, rotMat, tvec);
        
        // Publish the camera transform
        geometry_msgs::msg::TransformStamped transformStamped;
        transformStamped.header.stamp = stamp;
        transformStamped.header.frame_id = "world";  // or "map" - your world frame
        transformStamped.child_frame_id = "camera_pose_from_tag_" + std::to_string(tag_id);
        
        tf2::Vector3 cam_pos = world_to_camera.getOrigin();
        tf2::Quaternion cam_quat = world_to_camera.getRotation();
        
        transformStamped.transform.translation.x = cam_pos.x();
        transformStamped.transform.translation.y = cam_pos.y();
        transformStamped.transform.translation.z = cam_pos.z();
        transformStamped.transform.rotation.x = cam_quat.x();
        transformStamped.transform.rotation.y = cam_quat.y();
        transformStamped.transform.rotation.z = cam_quat.z();
        transformStamped.transform.rotation.w = cam_quat.w();
        
        tf_broadcaster_->sendTransform(transformStamped);
        
        RCLCPP_INFO(this->get_logger(), "Published camera pose from tag %d: pos[%f, %f, %f]", 
                    tag_id, cam_pos.x(), cam_pos.y(), cam_pos.z());
    }

    void publishFusedCameraPose(const std::vector<tf2::Transform>& camera_poses, 
                               const std::vector<int>& tag_ids, 
                               const rclcpp::Time& stamp)
    {
        if (camera_poses.empty()) return;
        
        // Simple averaging of positions (you could implement more sophisticated fusion)
        tf2::Vector3 avg_position(0, 0, 0);
        tf2::Quaternion avg_quaternion(0, 0, 0, 0);
        
        for (const auto& pose : camera_poses) {
            avg_position += pose.getOrigin();
            tf2::Quaternion q = pose.getRotation();
            avg_quaternion += q;
        }
        
        avg_position /= camera_poses.size();
        avg_quaternion.normalize();
        
        // Publish the fused camera pose
        geometry_msgs::msg::TransformStamped transformStamped;
        transformStamped.header.stamp = stamp;
        transformStamped.header.frame_id = "world";
        transformStamped.child_frame_id = "camera_pose_fused";
        
        transformStamped.transform.translation.x = avg_position.x();
        transformStamped.transform.translation.y = avg_position.y();
        transformStamped.transform.translation.z = avg_position.z();
        transformStamped.transform.rotation.x = avg_quaternion.x();
        transformStamped.transform.rotation.y = avg_quaternion.y();
        transformStamped.transform.rotation.z = avg_quaternion.z();
        transformStamped.transform.rotation.w = avg_quaternion.w();
        
        tf_broadcaster_->sendTransform(transformStamped);
        
        std::string tag_list = "";
        for (int id : tag_ids) {
            tag_list += std::to_string(id) + " ";
        }
        RCLCPP_INFO(this->get_logger(), "Published fused camera pose from tags: %s", tag_list.c_str());
    }
    
    // AprilTag detector
    apriltag_family_t *tf = tagStandard41h12_create();
    apriltag_detector_t *td = apriltag_detector_create();
    double tag_size;

    // Fixed tag poses in world frame
    std::map<int, TagPose> fixed_tag_poses_;

    // Camera info
    std::array<double, 9> camera_info_K;
    std::vector<double> camera_info_D;
    bool camera_info_received_ = false;

    // ROS subscriptions and publishers
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<AprilTagDetector>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}