// -*-c++-*----------------------------------------------------------------------------------------
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

#include "frequency_cam/frequency_cam_ros2.h"

#ifdef USE_CV_BRIDGE_HPP
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include <event_camera_codecs/decoder.h>
#include <math.h>

#include <algorithm>  // std::sort, std::stable_sort, std::clamp
#include <filesystem>
#include <fstream>
#include <image_transport/image_transport.hpp>
#include <opencv2/imgcodecs.hpp>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>

namespace frequency_cam
{
using EventPacket = event_camera_msgs::msg::EventPacket;

FrequencyCamROS::FrequencyCamROS(const rclcpp::NodeOptions & options)
: Node("frequency_cam", options)
{
  if (!initialize()) {
    RCLCPP_ERROR(get_logger(), "frequency cam startup failed!");
    throw std::runtime_error("startup of FrequencyCam node failed!");
  }
}

bool FrequencyCamROS::initialize()
{
  const double fps = std::max(declare_parameter<double>("publishing_frequency", 20.0), 1.0);
  eventImageDt_ = 1.0 / fps;
  RCLCPP_INFO_STREAM(get_logger(), "publishing frequency: " << fps);

  useLogFrequency_ = declare_parameter<bool>("use_log_frequency", false);
  imageMaker_.setUseLogFrequency(useLogFrequency_);
  overlayEvents_ = declare_parameter<bool>("overlay_events", false);
  imageMaker_.setOverlayEvents(overlayEvents_);

  rmw_qos_profile_t qosProf = rmw_qos_profile_default;
  imagePub_ = image_transport::create_publisher(this, "~/image_raw", qosProf);
  debugX_ = static_cast<uint16_t>(declare_parameter<int>("debug_x", 320));
  debugY_ = static_cast<uint16_t>(declare_parameter<int>("debug_y", 240));
  imageMaker_.setDebugX(debugX_);
  imageMaker_.setDebugY(debugY_);

  const double minFreq = declare_parameter<double>("min_frequency", 1.0);
  const double maxFreq =
    std::max(minFreq + 1e-5, declare_parameter<double>("max_frequency", minFreq * 2));
  RCLCPP_INFO_STREAM(get_logger(), "frequency range: " << minFreq << " -> " << maxFreq);
  imageMaker_.setFrequencyLimits(minFreq, maxFreq);
  imageMaker_.setLegendWidth(declare_parameter<int>("legend_width", 100));
  imageMaker_.setLegendValues(
    declare_parameter<std::vector<double>>("legend_frequencies", std::vector<double>()));
  imageMaker_.setLegendNumBins(declare_parameter<int>("legend_num_bins", 11));
  imageMaker_.setNumSigDigits(declare_parameter<int>("legend_num_sig_digits", 3));
  cam_.initialize(
    minFreq, maxFreq, declare_parameter<double>("cutoff_period", 5.0),
    declare_parameter<int>("num_timeout_cycles", 2.0), debugX_, debugY_);

  const std::string bag = this->declare_parameter<std::string>("bag_file", "");
  if (bag.empty()) {
    // start statistics timer only when not playing from bag
    statsTimer_ = rclcpp::create_timer(
      this, this->get_clock(), rclcpp::Duration::from_seconds(2.0),
      [=]() { this->statisticsTimerExpired(); });
    // for ROS2 frame timer and subscriber are initialized right away
    frameTimer_ = rclcpp::create_timer(
      this, this->get_clock(), rclcpp::Duration::from_seconds(eventImageDt_),
      [=]() { this->frameTimerExpired(); });

    const size_t EVENT_QUEUE_DEPTH(1000);
    auto qos = rclcpp::QoS(rclcpp::KeepLast(EVENT_QUEUE_DEPTH)).best_effort().durability_volatile();

    eventSub_ = this->create_subscription<EventPacket>(
      "~/events", qos, std::bind(&FrequencyCamROS::eventMsg, this, std::placeholders::_1));
  } else {
    // reading from bag is only for debugging
    readFrameTimes();
    playEventsFromBag(bag, declare_parameter<std::string>("bag_topic", "/event_camera/events"));
  }
  return (true);
}

void FrequencyCamROS::readFrameTimes()
{
  const std::string fname = this->declare_parameter<std::string>("frame_time_file", "");
  if (!fname.empty()) {
    std::ifstream file(fname);
    if (!file.is_open()) {
      RCLCPP_ERROR_STREAM(this->get_logger(), "cannot open frame time file: " << fname);
    } else {
      uint64_t t;
      while (file >> t) {
        frameTimes_.push(t);
      }
    }
    RCLCPP_INFO_STREAM(
      this->get_logger(), "read " << frameTimes_.size() << " frame times from " << fname);
  }
}

static EventPacket::ConstSharedPtr get_next_message(
  rosbag2_cpp::Reader * reader, const std::string & bagTopic)
{
  auto bagmsg = reader->read_next();
  rclcpp::SerializedMessage serializedMsg(*bagmsg->serialized_data);
  rclcpp::Serialization<EventPacket> serialization;

  if (bagmsg->topic_name == bagTopic) {
    EventPacket::SharedPtr msg(new EventPacket());
    serialization.deserialize_message(&serializedMsg, &(*msg));
    return (msg);
  }
  return (EventPacket::SharedPtr());
}

void FrequencyCamROS::playEventsFromBag(const std::string & bagName, const std::string & bagTopic)
{
  imageMaker_.setScale(this->declare_parameter<double>("scale_image", 1.0));
  uint64_t currentFrameTime(frameTimes_.empty() ? 0 : frameTimes_.front());

  if (!frameTimes_.empty()) {
    frameTimes_.pop();  // since we are already using the front(), if at all
  }
  uint32_t frameCount(0);
  const uint64_t delta_t = static_cast<uint64_t>(eventImageDt_ * 1e9);
  rosbag2_cpp::Reader reader;

  reader.open(bagName);
  const std::string path = this->declare_parameter<std::string>("path", "./frames");
  std::filesystem::create_directories(path);

  const auto t_0 = std::chrono::high_resolution_clock::now();

  while (reader.has_next()) {
    auto msg = get_next_message(&reader, bagTopic);
    if (msg) {
      if (height_ == 0) {  // must get first frame time
        uint64_t firstTime{0};
        if (!initializeOnFirstMessage(msg, &firstTime)) {
          continue;  // no first time found!
        }
        if (frameTimes_.empty()) {
          currentFrameTime = firstTime + delta_t;
        }
      }
      auto decoder = decoderFactory_.getInstance(*msg);
      uint64_t nextTime{0};
      // loop will exit when all events in message have been processed
      while (decoder->decodeUntil(*msg, &cam_, currentFrameTime, &nextTime)) {
        if (frameTimes_.empty()) {  // free running mode
          for (; currentFrameTime <= nextTime; currentFrameTime += delta_t) {
            makeAndWriteFrame(currentFrameTime, path, frameCount++);
          }
        } else {
          // use loop in case multiple frames fit inbetween two events
          while (!frameTimes_.empty() && currentFrameTime <= nextTime) {
            makeAndWriteFrame(currentFrameTime, path, frameCount++);
            currentFrameTime = frameTimes_.front();
            frameTimes_.pop();
          }
        }
      }
    } else {
      RCLCPP_WARN(get_logger(), "skipped invalid message type in bag!");
    }
  }
  // event count
  size_t numEvents;
  cam_.getStatistics(&numEvents);
  const auto t_1 = std::chrono::high_resolution_clock::now();
  const auto totTime = std::chrono::duration_cast<std::chrono::microseconds>(t_1 - t_0).count();
  RCLCPP_INFO_STREAM(get_logger(), "played bag at rate: " << (numEvents / totTime) << " Mev/s");
  rclcpp::shutdown();
}

void FrequencyCamROS::makeAndWriteFrame(
  uint64_t debugTime, const std::string & path, uint32_t frameCount)
{
  cv::Mat eventImg;
  cv::Mat freqImg =
    cam_.makeFrequencyAndEventImage(&eventImg, overlayEvents_, useLogFrequency_, eventImageDt_);
  const cv::Mat window = imageMaker_.make(debugTime, freqImg, eventImg);
  char fname[256];
  snprintf(fname, sizeof(fname) - 1, "/frame_%05u.jpg", frameCount);
  cv::imwrite(path + fname, window);
}

bool FrequencyCamROS::initializeOnFirstMessage(
  EventPacket::ConstSharedPtr msg, uint64_t * firstTime)
{
  if (msg->events.empty()) {
    return (false);
  }
  auto decoder = decoderFactory_.getInstance(*msg);
  if (!decoder) {
    RCLCPP_ERROR_STREAM(get_logger(), "invalid encoding: " << msg->encoding);
    throw(std::runtime_error("invalid encoding!"));
  }

  if (!decoder->findFirstSensorTime(*msg, firstTime)) {
    return (false);
  }
  height_ = msg->height;
  width_ = msg->width;
  header_ = msg->header;  // copy frame id
  lastSeq_ = msg->seq - 1;
  const uint64_t t_off = (msg->encoding == "mono")
                           ? *firstTime
                           : (rclcpp::Time(msg->header.stamp).nanoseconds() - *firstTime);
  cam_.initializeState(width_, height_, *firstTime, t_off);
  return (true);
}

void FrequencyCamROS::eventMsg(EventPacket::ConstSharedPtr msg)
{
  const auto t_start = std::chrono::high_resolution_clock::now();
  uint64_t firstTime{0};
  if (height_ == 0 && !initializeOnFirstMessage(msg, &firstTime)) {
    return;
  }
  header_.stamp = msg->header.stamp;
  // decode will produce callbacks to cam_
  auto decoder = decoderFactory_.getInstance(*msg);
  decoder->decode(*msg, &cam_);

  msgCount_++;
  droppedSeq_ += static_cast<int64_t>(msg->seq) - lastSeq_ - 1;
  lastSeq_ = static_cast<int64_t>(msg->seq);
  const auto t_end = std::chrono::high_resolution_clock::now();
  totTime_ += std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
}

void FrequencyCamROS::frameTimerExpired()
{
  if (imagePub_.getNumSubscribers() != 0 && height_ != 0) {
    cv::Mat eventImg;
    cv::Mat freqImg =
      cam_.makeFrequencyAndEventImage(&eventImg, overlayEvents_, useLogFrequency_, eventImageDt_);
    const cv::Mat window =
      imageMaker_.make(this->get_clock()->now().nanoseconds(), freqImg, eventImg);
    imagePub_.publish(cv_bridge::CvImage(header_, "bgr8", window).toImageMsg());
  }
}

void FrequencyCamROS::statisticsTimerExpired()
{
  size_t numEvents;
  cam_.getStatistics(&numEvents);
  const double N = static_cast<double>(numEvents);
  const double t = static_cast<double>(totTime_);
  if (numEvents > 0 && t > 0 && msgCount_ > 0) {
    RCLCPP_INFO(
      get_logger(), "%6.2f Mev/s, %8.2f msgs/s, %8.2f nsec/ev  %6.0f usec/msg, drop: %3ld", N / t,
      msgCount_ * 1.0e6 / t, 1e3 * t / N, t / msgCount_, droppedSeq_);
    cam_.resetStatistics();
    totTime_ = 0;
    msgCount_ = 0;
    droppedSeq_ = 0;
  } else {
    RCLCPP_INFO(get_logger(), "no events received");
  }
}
}  // namespace frequency_cam

RCLCPP_COMPONENTS_REGISTER_NODE(frequency_cam::FrequencyCamROS)
