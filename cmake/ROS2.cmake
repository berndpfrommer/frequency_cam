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


add_compile_options(-Wall -Wextra -Wpedantic -Werror)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_auto REQUIRED)
find_package(ament_cmake_ros REQUIRED)

find_package(OpenCV REQUIRED)

set(ROS2_DEPENDENCIES
  "rclcpp"
  "rclcpp_components"
  "rosbag2_cpp"
  "event_camera_msgs"
  "event_camera_codecs"
  "image_transport"
  "cv_bridge"
  "std_msgs")

foreach(pkg ${ROS2_DEPENDENCIES})
  find_package(${pkg} REQUIRED)
endforeach()

if(${cv_bridge_VERSION} GREATER "3.3.0")
  add_definitions(-DUSE_CV_BRIDGE_HPP)
endif()

if(${rclcpp_VERSION} VERSION_GREATER_EQUAL "28.0.0")
  add_definitions(-DUSE_MATCHED_EVENTS)
endif()

if(${image_transport_VERSION} VERSION_GREATER_EQUAL "6.3.0")
  add_definitions(-DIMAGE_TRANSPORT_USE_QOS)
endif()

if(${image_transport_VERSION} VERSION_GREATER_EQUAL "6.4.0")
  add_definitions(-DIMAGE_TRANSPORT_USE_NODEINTERFACE)
endif()

#
# ---- frequency_cam shared library/component
#
add_library(${PROJECT_NAME} SHARED
  src/frequency_cam.cpp src/image_maker.cpp src/frequency_cam_ros2.cpp)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    rclcpp::rclcpp
    rclcpp_components::component
    rosbag2_cpp::rosbag2_cpp
    ${event_camera_msgs_TARGETS}
    ${event_camera_codecs_TARGETS}
    image_transport::image_transport
    cv_bridge::cv_bridge
    ${std_msgs_TARGETS}
)

target_include_directories(${PROJECT_NAME} PUBLIC include)


rclcpp_components_register_nodes(frequency_cam "frequency_cam::FrequencyCamROS")

add_executable(frequency_cam_node src/frequency_cam_node_ros2.cpp)
target_link_libraries(frequency_cam_node ${PROJECT_NAME})
target_include_directories(frequency_cam_node PRIVATE include)

#
# cpu benchmark
#
add_executable(cpu_benchmark src/cpu_benchmark.cpp)
target_link_libraries(cpu_benchmark ${PROJECT_NAME})
target_include_directories(cpu_benchmark PRIVATE include)

# -------- installation

# the shared library goes into the global lib dir so it can
# be used as a composable node by other projects

install(TARGETS
  ${PROJECT_NAME}
  DESTINATION lib)

# the node must go into the project specific lib directory or else
# the launch file will not find it

install(TARGETS
  frequency_cam_node
  cpu_benchmark
  DESTINATION lib/${PROJECT_NAME}/)

# install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
  FILES_MATCHING PATTERN "*.py")


if(BUILD_TESTING)
  find_package(ament_cmake REQUIRED)
  find_package(ament_cmake_copyright REQUIRED)
  find_package(ament_cmake_cppcheck REQUIRED)
  find_package(ament_cmake_cpplint REQUIRED)
  find_package(ament_cmake_clang_format REQUIRED)
  find_package(ament_cmake_flake8 REQUIRED)
  find_package(ament_cmake_lint_cmake REQUIRED)
  find_package(ament_cmake_pep257 REQUIRED)
  find_package(ament_cmake_xmllint REQUIRED)

  ament_copyright()
  ament_cppcheck(LANGUAGE c++)
  ament_cpplint(FILTERS "-build/include,-runtime/indentation_namespace")
  ament_clang_format()
  ament_flake8()
  ament_lint_cmake()
  ament_pep257()
  ament_xmllint()
endif()


ament_package()
