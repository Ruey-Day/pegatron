cmake_minimum_required(VERSION 3.8)
project(apriltag_detector)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
include_directories(${cv_bridge_INCLUDE_DIRS})


# Manual AprilTag configuration
# Define the include and lib directories for AprilTag
set(APRILTAG_INCLUDE_DIR "/usr/include" CACHE PATH "AprilTag include directory")
set(APRILTAG_LIB_DIR "/usr/lib/x86_64-linux-gnu" CACHE PATH "AprilTag lib directory")

# Check if the library exists
if(NOT EXISTS "${APRILTAG_LIB_DIR}/libapriltag.so")
  message(FATAL_ERROR "AprilTag library not found in ${APRILTAG_LIB_DIR}. Please update APRILTAG_LIB_DIR.")
endif()

# Create the executable
add_executable(apriltag_detector src/apriltag_detector.cpp)
target_include_directories(apriltag_detector PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${APRILTAG_INCLUDE_DIR}
  ${cv_bridge_INCLUDE_DIRS}
)
target_compile_features(apriltag_detector PUBLIC c_std_99 cxx_std_17)

# Link against the necessary libraries
target_link_libraries(apriltag_detector
  ${rclcpp_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${geometry_msgs_LIBRARIES}
  ${tf2_LIBRARIES}
  ${tf2_ros_LIBRARIES}
  ${tf2_geometry_msgs_LIBRARIES}
  ${cv_bridge_LIBRARIES}
  ${OpenCV_LIBS}
  ${APRILTAG_LIB_DIR}/libapriltag.so
)

ament_target_dependencies(apriltag_detector
  rclcpp
  sensor_msgs
  geometry_msgs
  tf2_ros
  cv_bridge
  OpenCV
)

# Install the executable
install(TARGETS apriltag_detector
  DESTINATION lib/${PROJECT_NAME})

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
