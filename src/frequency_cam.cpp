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

#include "frequency_cam/frequency_cam.h"

#include <cv_bridge/cv_bridge.h>
#include <event_array_codecs/decoder.h>
#include <math.h>

#include <algorithm>  // std::sort, std::stable_sort, std::clamp
#include <filesystem>
#include <fstream>
#include <image_transport/image_transport.hpp>
#include <numeric>  // std::iota
#include <opencv2/imgcodecs.hpp>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <sstream>

#ifdef DEBUG  // the debug flag must be set in the header file
std::ofstream debug("freq.txt");
std::ofstream debug_readout("readout.txt");
#endif

namespace frequency_cam
{
using EventArray = event_array_msgs::msg::EventArray;

FrequencyCam::FrequencyCam(const rclcpp::NodeOptions & options) : Node("frequency_cam", options)
{
  if (!initialize()) {
    RCLCPP_ERROR(get_logger(), "frequency cam  startup failed!");
    throw std::runtime_error("startup of FrequencyCam node failed!");
  }
}

FrequencyCam::~FrequencyCam() { delete[] state_; }

static void compute_alpha_beta(const double T_cut, double * alpha, double * beta)
{
  // compute the filter coefficients alpha and beta (see paper)
  const double omega_cut = 2 * M_PI / T_cut;
  const double phi = 2 - std::cos(omega_cut);
  *alpha = (1.0 - std::sin(omega_cut)) / std::cos(omega_cut);
  *beta = phi - std::sqrt(phi * phi - 1.0);  // see paper
}

bool FrequencyCam::initialize()
{
  rmw_qos_profile_t qosProf = rmw_qos_profile_default;
  imagePub_ = image_transport::create_publisher(this, "~/frequency_image", qosProf);
  const size_t EVENT_QUEUE_DEPTH(1000);
  auto qos = rclcpp::QoS(rclcpp::KeepLast(EVENT_QUEUE_DEPTH)).best_effort().durability_volatile();
#ifdef DEBUG
  useSensorTime_ = false;
#else
  useSensorTime_ = this->declare_parameter<bool>("use_sensor_time", false);
#endif
  const std::string bag = this->declare_parameter<std::string>("bag_file", "");
  freq_[0] = this->declare_parameter<double>("min_frequency", 1.0);
  freq_[0] = std::max(freq_[0], 0.1);
  freq_[1] = this->declare_parameter<double>("max_frequency", -1.0);
  dtMax_ = 1.0 / freq_[0];
  dtMaxHalf_ = 0.5 * dtMax_;
  dtMin_ = 1.0 / (freq_[1] >= freq_[0] ? freq_[1] : 1.0);
  dtMinHalf_ = 0.5 * dtMin_;
  RCLCPP_INFO_STREAM(get_logger(), "minimum frequency: " << freq_[0]);
  RCLCPP_INFO_STREAM(get_logger(), "maximum frequency: " << freq_[1]);
  useLogFrequency_ = this->declare_parameter<bool>("use_log_frequency", false);
  tfFreq_[0] = useLogFrequency_ ? LogTF::tf(std::max(freq_[0], 1e-8)) : freq_[0];
  tfFreq_[1] = useLogFrequency_ ? LogTF::tf(std::max(freq_[1], 1e-7)) : freq_[1];
  legendWidth_ = this->declare_parameter<int>("legend_width", 100);
  legendValues_ =
    this->declare_parameter<std::vector<double>>("legend_frequencies", std::vector<double>());
  if (legendValues_.empty()) {
    legendBins_ = this->declare_parameter<int>("legend_bins", 11);
  }
  timeoutCycles_ = this->declare_parameter<int>("num_timeout_cycles", 2.0);
  overlayEvents_ = this->declare_parameter<bool>("overlay_events", false);
  const double T_prefilter = std::max(1.0, this->declare_parameter<double>("cutoff_period", 6));
  double alpha_prefilter, beta_prefilter;
  compute_alpha_beta(T_prefilter, &alpha_prefilter, &beta_prefilter);

  // compute IIR filter coefficient from alpha and beta (see paper)
  c_[0] = alpha_prefilter + beta_prefilter;
  c_[1] = -alpha_prefilter * beta_prefilter;
  c_p_ = 0.5 * (1 + beta_prefilter);

#ifdef DEBUG
  debugX_ = static_cast<uint16_t>(this->declare_parameter<int>("debug_x", 320));
  debugY_ = static_cast<uint16_t>(this->declare_parameter<int>("debug_y", 240));
#endif

  eventImageDt_ =
    1.0 / std::max(this->declare_parameter<double>("publishing_frequency", 20.0), 1.0);

  if (bag.empty()) {
    eventSub_ = this->create_subscription<EventArray>(
      "~/events", qos, std::bind(&FrequencyCam::callbackEvents, this, std::placeholders::_1));
    pubTimer_ = rclcpp::create_timer(
      this, this->get_clock(), rclcpp::Duration::from_seconds(eventImageDt_),
      [=]() { this->publishImage(); });
    statsTimer_ = rclcpp::create_timer(
      this, this->get_clock(), rclcpp::Duration::from_seconds(2.0), [=]() { this->statistics(); });
  } else {
    // reading from bag is only for debugging
    playEventsFromBag(bag);
  }
  return (true);
}

void FrequencyCam::playEventsFromBag(const std::string & bagName)
{
  rclcpp::Time lastFrameTime(0);
  rosbag2_cpp::Reader reader;
  reader.open(bagName);
  rclcpp::Serialization<EventArray> serialization;
  const auto delta_t = rclcpp::Duration::from_seconds(eventImageDt_);
  bool hasValidTime = false;
  uint32_t frameCount(0);
  const std::string path = this->declare_parameter<std::string>("path", "./frames");
  std::filesystem::create_directories(path);

  while (reader.has_next()) {
    auto bagmsg = reader.read_next();
    rclcpp::SerializedMessage serializedMsg(*bagmsg->serialized_data);
    EventArray::SharedPtr msg(new EventArray());
    serialization.deserialize_message(&serializedMsg, &(*msg));
    if (msg) {
      const rclcpp::Time t(msg->header.stamp);
      callbackEvents(msg);
      if (hasValidTime) {
        if (t - lastFrameTime > delta_t) {
          const cv::Mat img = makeImage((lastFrameTime + delta_t).nanoseconds());
          lastFrameTime = lastFrameTime + delta_t;
          char fname[256];
          snprintf(fname, sizeof(fname) - 1, "/frame_%05u.jpg", frameCount);
          cv::imwrite(path + fname, img);
          frameCount++;
        }
      } else {
        hasValidTime = true;
        lastFrameTime = t;
      }
    } else {
      RCLCPP_WARN(get_logger(), "skipped invalid message type in bag!");
    }
  }
  RCLCPP_INFO_STREAM(get_logger(), "played bag at rate: " << (eventCount_ / totTime_) << " Mev/s");
}

void FrequencyCam::initializeState(uint32_t width, uint32_t height, uint32_t t)
{
  RCLCPP_INFO_STREAM(
    get_logger(), "state image size is: " << (width * height * sizeof(State)) / (1 << 20)
                                          << "MB (better fit into CPU cache)");
  width_ = width;
  height_ = height;
  state_ = new State[width * height];
  for (size_t i = 0; i < width * height; i++) {
    State & s = state_[i];
    s.t_flip_up_down = t;
    s.t_flip_down_up = t;
    s.L_km1 = 0;
    s.L_km2 = 0;
    s.period = -1;
    s.polarity = -1;  // should set this to zero?
  }
}

static void compute_max(const cv::Mat & img, double * maxVal)
{
  {
    // no max frequency specified, calculate highest frequency
    cv::Point minLoc, maxLoc;
    double minVal;
    cv::minMaxLoc(img, &minVal, maxVal, &minLoc, &maxLoc);
  }
}

/*
 * format frequency labels for opencv
 */
static std::string format_freq(double v)
{
  std::stringstream ss;
  //ss << " " << std::fixed << std::setw(7) << std::setprecision(1) << v;
  ss << " " << std::setw(6) << (int)v;
  return (ss.str());
}

static void draw_labeled_rectangle(
  cv::Mat * window, int x_off, int y_off, int height, const std::string & text,
  const cv::Vec3b & color)
{
  const cv::Point tl(x_off, y_off);                      // top left
  const cv::Point br(window->cols - 1, y_off + height);  // bottom right
  cv::rectangle(*window, tl, br, color, -1 /* filled */, cv::LINE_8);
  // place text
  const cv::Point tp(x_off - 2, y_off + height / 2 - 2);
  const cv::Scalar textColor = CV_RGB(0, 0, 0);
  cv::putText(
    *window, text, tp, cv::FONT_HERSHEY_PLAIN, 1.0, textColor, 2.0 /*thickness*/,
    cv::LINE_AA /* anti-alias */);
}

std::vector<float> FrequencyCam::findLegendValuesAndText(
  const double minVal, const double maxVal, std::vector<std::string> * text) const
{
  std::vector<float> values;
  if (legendValues_.empty()) {
    // no explicit legend values are provided
    // use equidistant bins between min and max val.
    // This could be either in linear or log space.
    // If the range of values is large, round to next integer
    const double range = maxVal - minVal;
    bool round_it = !useLogFrequency_ && (range / (legendBins_ - 1)) > 2.0;
    for (size_t i = 0; i < legendBins_; i++) {
      const double raw_v = minVal + (static_cast<float>(i) / (legendBins_ - 1)) * range;
      const double v = round_it ? std::round(raw_v) : raw_v;
      values.push_back(v);
      text->push_back(format_freq(useLogFrequency_ ? LogTF::inv(v) : v));
    }
  } else {
    // legend values are explicitly given
    for (const auto & lv : legendValues_) {
      values.push_back(useLogFrequency_ ? LogTF::tf(lv) : lv);
      text->push_back(format_freq(lv));
    }
  }
  return (values);
}

void FrequencyCam::addLegend(cv::Mat * window, const double minVal, const double maxVal) const
{
  const int x_off = window->cols - legendWidth_;  // left border of legend
  const double range = maxVal - minVal;
  std::vector<std::string> text;
  std::vector<float> values = findLegendValuesAndText(minVal, maxVal, &text);
  if (!values.empty()) {
    cv::Mat valueMat(values.size(), 1, CV_32FC1);
    for (size_t i = 0; i < values.size(); i++) {
      valueMat.at<float>(i, 0) = values[i];
    }
    // rescale values matrix the same way as original image
    cv::Mat scaledValueMat;
    cv::convertScaleAbs(valueMat, scaledValueMat, 255.0 / range, -minVal * 255.0 / range);
    cv::Mat colorCode;
    cv::applyColorMap(scaledValueMat, colorCode, colorMap_);
    // draw filled rectangles and text labels
    const int height = window->rows / values.size();  // integer division
    for (size_t i = 0; i < values.size(); i++) {
      const int y_off = static_cast<float>(i) / values.size() * window->rows;
      draw_labeled_rectangle(window, x_off, y_off, height, text[i], colorCode.at<cv::Vec3b>(i, 0));
    }
  } else {
    // for some reason or the other no legend could be drawn
    cv::Mat roiLegend = (*window)(cv::Rect(x_off, 0, legendWidth_, window->rows));
    roiLegend.setTo(CV_RGB(0, 0, 0));
  }
}

cv::Mat FrequencyCam::makeFrequencyAndEventImage(cv::Mat * eventImage) const
{
  if (overlayEvents_) {
    *eventImage = cv::Mat::zeros(height_, width_, CV_8UC1);
  }
  if (useLogFrequency_) {
    return (
      overlayEvents_ ? makeTransformedFrequencyImage<LogTF, EventFrameUpdater>(eventImage)
                     : makeTransformedFrequencyImage<LogTF, NoEventFrameUpdater>(eventImage));
  }
  return (
    overlayEvents_ ? makeTransformedFrequencyImage<NoTF, EventFrameUpdater>(eventImage)
                   : makeTransformedFrequencyImage<NoTF, NoEventFrameUpdater>(eventImage));
}

cv::Mat FrequencyCam::makeImage(uint64_t t) const
{
  cv::Mat eventImg;
  cv::Mat rawImg = makeFrequencyAndEventImage(&eventImg);
#ifdef DEBUG
  const double v = rawImg.at<float>(debugY_, debugX_);
  debug_readout << std::fixed << std::setprecision(6) << (t * 1e-9) << " "
                << (useLogFrequency_ ? LogTF::inv(v) : NoTF::inv(v)) << std::endl;
#else
  (void)t;
#endif
  cv::Mat scaled;
  double minVal = tfFreq_[0];
  double maxVal = tfFreq_[1];
  if (freq_[1] < 0) {
    compute_max(rawImg, &maxVal);
  }
  const double range = maxVal - minVal;
  cv::convertScaleAbs(rawImg, scaled, 255.0 / range, -minVal * 255.0 / range);
  cv::Mat window(scaled.rows, scaled.cols + legendWidth_, CV_8UC3);
  cv::Mat colorImg = window(cv::Rect(0, 0, scaled.cols, scaled.rows));
  cv::applyColorMap(scaled, colorImg, colorMap_);
  colorImg.setTo(CV_RGB(0, 0, 0), rawImg == 0);  // render invalid points black
  if (overlayEvents_) {
    const cv::Scalar eventColor = CV_RGB(127, 127, 127);
    // only show events where no frequency is detected
    colorImg.setTo(eventColor, (rawImg == 0) & eventImg);
  }
  if (legendWidth_ > 0) {
    addLegend(&window, minVal, maxVal);
  }
  return (window);
}

void FrequencyCam::publishImage()
{
  if (imagePub_.getNumSubscribers() != 0 && height_ != 0) {
    const cv::Mat window = makeImage(this->get_clock()->now().nanoseconds());
    header_.stamp = lastTime_;
    imagePub_.publish(cv_bridge::CvImage(header_, "bgr8", window).toImageMsg());
  }
}

void FrequencyCam::callbackEvents(EventArrayConstPtr msg)
{
  const auto t_start = std::chrono::high_resolution_clock::now();
  const auto time_base =
    useSensorTime_ ? msg->time_base : rclcpp::Time(msg->header.stamp).nanoseconds();
  lastTime_ = rclcpp::Time(msg->header.stamp);
  auto decoder = event_array_codecs::Decoder::getInstance(msg->encoding);
  decoder->setTimeBase(time_base);
  if (state_ == 0 && !msg->events.empty()) {
    uint64_t t;
    if (!decoder->findFirstSensorTime(msg->events.data(), msg->events.size(), &t)) {
      return;
    }
#ifdef DEBUG
    timeOffset_ = (t / 1000) - shorten_time(t);  // high bits lost by shortening
#endif
    initializeState(msg->width, msg->height, shorten_time(t) - 1 /* - 1usec */);
    header_ = msg->header;  // copy frame id
    lastSeq_ = msg->seq - 1;
  }
  decoder->decode(msg->events.data(), msg->events.size(), this);
  msgCount_++;
  droppedSeq_ += static_cast<int64_t>(msg->seq) - lastSeq_ - 1;
  lastSeq_ = static_cast<int64_t>(msg->seq);

  const auto t_end = std::chrono::high_resolution_clock::now();
  totTime_ += std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
}

void FrequencyCam::statistics()
{
  if (eventCount_ > 0 && totTime_ > 0) {
    const double usec = static_cast<double>(totTime_);
    RCLCPP_INFO(
      get_logger(), "%6.2f Mev/s, %8.2f msgs/s, %8.2f nsec/ev  %6.0f usec/msg, drop: %3ld",
      double(eventCount_) / usec, msgCount_ * 1.0e6 / usec, 1e3 * usec / (double)eventCount_,
      usec / msgCount_, droppedSeq_);
    eventCount_ = 0;
    totTime_ = 0;
    msgCount_ = 0;
    droppedSeq_ = 0;
  }
}

std::ostream & operator<<(std::ostream & os, const FrequencyCam::Event & e)
{
  os << std::fixed << std::setw(10) << std::setprecision(6) << e.t * 1e-6 << " " << (int)e.polarity
     << " " << e.x << " " << e.y;
  return (os);
}

}  // namespace frequency_cam

RCLCPP_COMPONENTS_REGISTER_NODE(frequency_cam::FrequencyCam)
