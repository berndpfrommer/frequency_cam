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

#include "frequency_cam/image_maker.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

// #define DEBUG
#ifdef DEBUG
static std::ofstream readout("readout.txt");
#endif

namespace frequency_cam
{
// round and format taken from here:
// https://stackoverflow.com/questions/17211122/formatting-n-significant-digits-in-c-without-scientific-notation
int round(double number)
{
  return (number >= 0) ? static_cast<int>(number + 0.5) : static_cast<int>(number - 0.5);
}

std::string format(double f, int w, int num_sig_digits)
{
  if (f == 0) {
    return "0";
  }
  /*digits before decimal point*/
  int d = static_cast<int>(::ceil(::log10(f < 0 ? -f : f)));
  const int dd = num_sig_digits - d;
  double order = ::pow(10., dd);
  std::stringstream ss;
  ss << std::fixed << std::setw(w) << std::setprecision(std::max(dd, 0))
     << round(f * order) / order;
  return ss.str();
}

void ImageMaker::setFrequencyLimits(double min, double max)
{
  freq_[0] = min;
  freq_[0] = std::max(freq_[0], 0.1);
  freq_[1] = max;
  tfFreq_[0] = useLogFrequency_ ? std::log10(std::max(freq_[0], 1e-8)) : freq_[0];
  tfFreq_[1] = useLogFrequency_ ? std::log10(std::max(freq_[1], 1e-7)) : freq_[1];
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
static std::string format_freq(double v, int n_sig_dig)
{
#if 0
  std::stringstream ss;
  ss << " " << std::setw(6) << round(v);
  return (ss.str());
#else
  return (format(v, 6, n_sig_dig));
#endif
}

static void draw_labeled_rectangle(
  cv::Mat * window, int x_off, int y_off, int height, const std::string & text, double scale,
  const cv::Vec3b & color)
{
  const double fontScale = scale * 1.0;
  const cv::Point tl(x_off, y_off);                      // top left
  const cv::Point br(window->cols - 1, y_off + height);  // bottom right
  cv::rectangle(*window, tl, br, color, -1 /* filled */, cv::LINE_8);
  // place text
  const cv::Point tp(x_off - 2, y_off + height / 2 - 2);
  const cv::Scalar textColor = CV_RGB(0, 0, 0);
  cv::putText(
    *window, text, tp, cv::FONT_HERSHEY_PLAIN, fontScale, textColor, 2.0 * scale /*thickness*/,
    cv::LINE_AA /* anti-alias */);
}

std::vector<float> ImageMaker::findLegendValuesAndText(
  const double minVal, const double maxVal, std::vector<std::string> * text) const
{
  std::vector<float> values;
  if (legendValues_.empty()) {
    // no explicit legend values are provided
    // use equidistant bins between min and max val.
    // This could be either in linear or log space.
    // If the range of values is large, round to next integer
    const double range = maxVal - minVal;
    bool round_it = !useLogFrequency_ && (range / (legendNumBins_ - 1)) > 2.0;
    for (size_t i = 0; i < legendNumBins_; i++) {
      const double raw_v = minVal + (static_cast<float>(i) / (legendNumBins_ - 1)) * range;
      const double v = round_it ? std::round(raw_v) : raw_v;
      values.push_back(v);
      text->push_back(format_freq(useLogFrequency_ ? std::pow(10.0, v) : v, sigDigits_));
    }
  } else {
    // legend values are explicitly given
    for (const auto & lv : legendValues_) {
      values.push_back(useLogFrequency_ ? std::log10(lv) : lv);
      text->push_back(format_freq(lv, sigDigits_));
    }
  }
  return (values);
}

void ImageMaker::addLegend(cv::Mat * window, const double minVal, const double maxVal) const
{
  if (scale_ != 1.0) {
    cv::resize(*window, *window, cv::Size(), scale_, scale_);
  }
  const int x_off =
    window->cols - static_cast<int>(legendWidth_ * scale_);  // left border of legend
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
      draw_labeled_rectangle(
        window, x_off, y_off, height, text[i], scale_, colorCode.at<cv::Vec3b>(i, 0));
    }
  } else {
    // for some reason or the other no legend could be drawn
    cv::Mat roiLegend = (*window)(cv::Rect(x_off, 0, legendWidth_, window->rows));
    roiLegend.setTo(CV_RGB(0, 0, 0));
  }
}

cv::Mat ImageMaker::make(uint64_t t, const cv::Mat & rawImg, const cv::Mat & eventImg) const
{
#ifdef DEBUG
  const double v = rawImg.at<float>(debugY_, debugX_);
  readout << std::fixed << std::setprecision(6) << (t * 1e-9) << " "
          << (useLogFrequency_ ? std::pow(10.0, v) : v) << std::endl;
#else
  (void)t;
#endif
  cv::Mat scaled;
  double minVal = tfFreq_[0];
  double maxVal = tfFreq_[1];
  if (freq_[1] < 0) {
    compute_max(rawImg, &maxVal);
    if (maxVal < tfFreq_[0]) {
      maxVal = tfFreq_[0] * 10;
    }
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

}  // namespace frequency_cam
