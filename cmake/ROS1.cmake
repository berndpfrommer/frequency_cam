#
# Copyright 2022 Bernd Pfrommer <bernd.pfrommer@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_compile_options(-Wall -Wextra -pedantic -Werror)
add_definitions(-DUSING_ROS_1)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  nodelet
  rosbag
  event_camera_msgs
  event_camera_codecs
  image_transport
  cv_bridge
  std_msgs)

include_directories(
  include
  ${catkin_INCLUDE_DIRS})

catkin_package()

# code common to nodelet and node
add_library(frequency_cam src/frequency_cam.cpp src/image_maker.cpp src/frequency_cam_ros1.cpp)
target_link_libraries(frequency_cam ${catkin_LIBRARIES})

# nodelet
add_library(frequency_cam_nodelet src/frequency_cam_nodelet.cpp)
target_link_libraries(frequency_cam_nodelet frequency_cam ${catkin_LIBRARIES})

# node
add_executable(frequency_cam_node src/frequency_cam_node_ros1.cpp)
target_link_libraries(frequency_cam_node frequency_cam ${catkin_LIBRARIES})


#############
## Install ##
#############

install(TARGETS frequency_cam_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(TARGETS frequency_cam frequency_cam_nodelet
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION})

install(FILES nodelet_plugins.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  FILES_MATCHING PATTERN "*.launch")


#############
## Testing ##
#############

# To be done...
