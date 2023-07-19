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

#include "frequency_cam/frequency_cam_ros1.h"

#include <cv_bridge/cv_bridge.h>

#include <chrono>

namespace frequency_cam
{
FrequencyCamROS::FrequencyCamROS(const ros::NodeHandle & nh) : nh_(nh)
{
  if (!initialize()) {
    ROS_ERROR("frequency cam  startup failed!");
    throw std::runtime_error("startup of FrequencyCamROS failed!");
  }
}

bool FrequencyCamROS::initialize()
{
  const double fps = std::max(nh_.param<double>("publishing_frequency", 20.0), 1.0);
  eventImageDt_ = 1.0 / fps;
  ROS_INFO_STREAM("publishing frequency: " << fps);

  useLogFrequency_ = nh_.param<bool>("use_log_frequency", false);
  imageMaker_.setUseLogFrequency(useLogFrequency_);
  overlayEvents_ = nh_.param<bool>("overlay_events", false);
  imageMaker_.setOverlayEvents(overlayEvents_);

  image_transport::ImageTransport it(nh_);
  imagePub_ = it.advertise(
    "image_raw", 1,
    boost::bind(&FrequencyCamROS::imageConnectCallback, this, boost::placeholders::_1),
    boost::bind(&FrequencyCamROS::imageConnectCallback, this, boost::placeholders::_1));
  statsTimer_ = nh_.createTimer(ros::Duration(2.0), &FrequencyCamROS::statisticsTimerExpired, this);
  debugX_ = static_cast<uint16_t>(nh_.param<int>("debug_x", 320));
  debugY_ = static_cast<uint16_t>(nh_.param<int>("debug_y", 240));
  imageMaker_.setDebugX(debugX_);
  imageMaker_.setDebugY(debugY_);
  const double minFreq = nh_.param<double>("min_frequency", 1.0);
  const double maxFreq = std::max(minFreq + 1e-5, nh_.param<double>("max_frequency", minFreq * 2));
  ROS_INFO_STREAM("frequency range: " << minFreq << " -> " << maxFreq);
  imageMaker_.setFrequencyLimits(minFreq, maxFreq);
  imageMaker_.setLegendWidth(nh_.param<int>("legend_width", 100));
  imageMaker_.setLegendValues(
    nh_.param<std::vector<double>>("legend_frequencies", std::vector<double>()));
  imageMaker_.setLegendNumBins(nh_.param<int>("legend_num_bins", 11));
  imageMaker_.setNumSigDigits(nh_.param<int>("legend_num_sig_digits", 3));
  cam_.initialize(
    minFreq, maxFreq, nh_.param<double>("cutoff_period", 5.0),
    nh_.param<int>("num_timeout_cycles", 2.0), debugX_, debugY_);
  return (true);
}

void FrequencyCamROS::imageConnectCallback(const image_transport::SingleSubscriberPublisher &)
{
  if (imagePub_.getNumSubscribers() != 0) {
    if (!isSubscribedToEvents_) {
      frameTimer_ =
        nh_.createTimer(ros::Duration(eventImageDt_), &FrequencyCamROS::frameTimerExpired, this);
      eventSub_ = nh_.subscribe("events", 1000 /*qsize */, &FrequencyCamROS::eventMsg, this);
      isSubscribedToEvents_ = true;
    }
  } else {
    if (isSubscribedToEvents_) {
      eventSub_.shutdown();  // unsubscribe
      isSubscribedToEvents_ = false;
      frameTimer_.stop();
    }
  }
}

void FrequencyCamROS::eventMsg(const EventPacket::ConstPtr & msg)
{
  const auto t_start = std::chrono::high_resolution_clock::now();
  if (msg->events.empty()) {
    return;
  }
  auto decoder = decoderFactory_.getInstance(*msg);
  if (!decoder) {
    ROS_INFO_STREAM("invalid encoding: " << msg->encoding);
    return;
  }
  header_.stamp = msg->header.stamp;
  if (height_ == 0) {
    uint64_t t;
    if (!decoder->findFirstSensorTime(*msg, &t)) {
      return;
    }
    height_ = msg->height;
    width_ = msg->width;
    header_ = msg->header;  // copy frame id
    header_.seq = seq_;     // set ROS1 sequence to zero
    lastSeq_ = msg->seq - 1;
    cam_.initializeState(width_, height_, t, ros::Time(header_.stamp).toNSec());
  }
  // decode will produce callbacks to cam_
  decoder->decode(*msg, &cam_);

  // update statistics
  msgCount_++;
  droppedSeq_ += static_cast<int64_t>(msg->seq) - lastSeq_ - 1;
  lastSeq_ = static_cast<int64_t>(msg->seq);
  const auto t_end = std::chrono::high_resolution_clock::now();
  totTime_ += std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
}

void FrequencyCamROS::frameTimerExpired(const ros::TimerEvent &)
{
  if (imagePub_.getNumSubscribers() != 0 && height_ != 0) {
    header_.seq++;
    cv::Mat eventImg;
    cv::Mat freqImg =
      cam_.makeFrequencyAndEventImage(&eventImg, overlayEvents_, useLogFrequency_, eventImageDt_);
    const cv::Mat window = imageMaker_.make(ros::Time::now().toNSec(), freqImg, eventImg);
    imagePub_.publish(cv_bridge::CvImage(header_, "bgr8", window).toImageMsg());
  }
}

void FrequencyCamROS::statisticsTimerExpired(const ros::TimerEvent &)
{
  size_t numEvents;
  cam_.getStatistics(&numEvents);
  const double N = static_cast<double>(numEvents);
  const double t = static_cast<double>(totTime_);
  if (numEvents > 0 && t > 0 && msgCount_ > 0) {
    ROS_INFO(
      "%6.2f Mev/s, %8.2f msgs/s, %8.2f nsec/ev  %6.0f usec/msg, drop: %3ld", N / t,
      msgCount_ * 1.0e6 / t, 1e3 * t / N, t / msgCount_, droppedSeq_);
    cam_.resetStatistics();
    totTime_ = 0;
    msgCount_ = 0;
    droppedSeq_ = 0;
  } else {
    if (imagePub_.getNumSubscribers() != 0) {
      ROS_INFO("no events received");
    }
  }
}

}  // namespace frequency_cam
