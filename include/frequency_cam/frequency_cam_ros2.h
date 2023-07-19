// -*-c++-*--------------------------------------------------------------------
// Copyright 2022 Bernd Pfrommer <bernd.pfrommer@gmail.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FREQUENCY_CAM__FREQUENCY_CAM_ROS2_H_
#define FREQUENCY_CAM__FREQUENCY_CAM_ROS2_H_

#include <event_camera_codecs/decoder_factory.h>

#include <event_camera_msgs/msg/event_packet.hpp>
#include <image_transport/image_transport.hpp>
#include <queue>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

#include "frequency_cam/frequency_cam.h"
#include "frequency_cam/image_maker.h"

namespace frequency_cam
{
class FrequencyCamROS : public rclcpp::Node
{
public:
  using EventPacket = event_camera_msgs::msg::EventPacket;
  explicit FrequencyCamROS(const rclcpp::NodeOptions & options);

  FrequencyCamROS(const FrequencyCamROS &) = delete;
  FrequencyCamROS & operator=(const FrequencyCamROS &) = delete;

private:
  void frameTimerExpired();
  void statisticsTimerExpired();
  bool initialize();
  void eventMsg(const EventPacket::ConstSharedPtr msg);
  void playEventsFromBag(const std::string & bagName, const std::string & bagTopic);
  void makeAndWriteFrame(uint64_t debugTime, const std::string & path, uint32_t frameCount);
  void readFrameTimes();
  bool initializeOnFirstMessage(EventPacket::ConstSharedPtr msg, uint64_t * firstTime);
  // ------ variables ----
  rclcpp::Time lastPubTime_{0};
  rclcpp::Subscription<EventPacket>::SharedPtr eventSub_;
  rclcpp::TimerBase::SharedPtr frameTimer_;
  rclcpp::TimerBase::SharedPtr statsTimer_;
  image_transport::Publisher imagePub_;
  event_camera_codecs::DecoderFactory<EventPacket, FrequencyCam> decoderFactory_;
  std_msgs::msg::Header header_;
  std::queue<uint64_t> frameTimes_;

  uint32_t width_{0};          // image width
  uint32_t height_{0};         // image height
  double eventImageDt_{0.01};  // time between published images
  bool overlayEvents_{false};
  bool useLogFrequency_{false};
  FrequencyCam cam_;
  ImageMaker imageMaker_;
  // --- statistics
  uint64_t totTime_{0};
  uint64_t msgCount_{0};
  int64_t lastSeq_{0};
  int64_t droppedSeq_{0};
  // --- debugging
  uint16_t debugX_;
  uint16_t debugY_;
};
}  // namespace frequency_cam
#endif  // FREQUENCY_CAM__FREQUENCY_CAM_ROS2_H_
