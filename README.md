mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/introlab/rtabmap.git
git clone -b humble-devel https://github.com/introlab/rtabmap_ros.git
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y

sudo apt update
sudo apt install ros-humble-rosidl-default-generators
sudo apt install ros-humble-desktop
sudo apt install ros-humble-ros-base

cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
