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

#ifndef FREQUENCY_CAM__IMAGE_MAKER_H_
#define FREQUENCY_CAM__IMAGE_MAKER_H_

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace frequency_cam
{
class ImageMaker
{
public:
  ImageMaker() {}
  ImageMaker(const ImageMaker &) = delete;
  ImageMaker & operator=(const ImageMaker &) = delete;

  void setFrequencyLimits(double min, double max);
  void setUseLogFrequency(bool b) { useLogFrequency_ = b; }
  void setOverlayEvents(bool b) { overlayEvents_ = b; }
  void setLegendWidth(int w) { legendWidth_ = w; }
  void setDebugX(uint16_t v) { debugX_ = v; }
  void setDebugY(uint16_t v) { debugY_ = v; }
  void setLegendValues(const std::vector<double> & vals) { legendValues_ = vals; }
  void setLegendNumBins(size_t numBins) { legendNumBins_ = numBins; }
  void setScale(double s) { scale_ = s; }
  void setNumSigDigits(int n) { sigDigits_ = n; }
  // returns complete window
  cv::Mat make(uint64_t t, const cv::Mat & rawImg, const cv::Mat & eventImg) const;

private:
  std::vector<float> findLegendValuesAndText(
    const double minVal, const double maxVal, std::vector<std::string> * text) const;
  void addLegend(cv::Mat * img, const double minVal, const double maxVal) const;

  // ------ variables ----
  bool useLogFrequency_{false};
  bool overlayEvents_{false};
  int legendWidth_{0};                // legend width (unscaled)
  std::vector<double> legendValues_;  // explicit legend values
  size_t legendNumBins_{6};           // how many bins to create if none given
  double freq_[2]{-1.0, -1.0};        // frequency range
  double tfFreq_[2]{0, 1.0};          // transformed frequency range
  double scale_{1.0};                 // how much to scale resolution
  int sigDigits_{3};                  // number of significant digits for legend
  constexpr static cv::ColormapTypes colorMap_{cv::COLORMAP_JET};
  //
  // ------------------ debugging stuff
  uint16_t debugX_{0};
  uint16_t debugY_{0};
};
}  // namespace frequency_cam
#endif  // FREQUENCY_CAM__IMAGE_MAKER_H_
