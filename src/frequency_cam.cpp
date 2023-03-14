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

#include <math.h>

#include <fstream>
#include <iomanip>
#include <iostream>

namespace frequency_cam
{

int roundUp(const int numToRound, const int multiple)
{
    if (multiple == 0) {
      return numToRound;
    }

    int remainder = numToRound % multiple;
    if (remainder == 0) {
      return numToRound;
    }

    return numToRound + multiple - remainder;
}

FrequencyCam::~FrequencyCam() {
  csv_file_.close();
  delete[] state_;

  std::cout << "Number of external triggers: " << nrExtTriggers_ << std::endl;
  std::cout << "Number of time synchronization matches: " << nrSyncMatches_ << std::endl;
  std::cout << "Number of detected wands: " << nrDetectedWands_ << std::endl;
}

static void compute_alpha_beta(const double T_cut, double * alpha, double * beta)
{
  // compute the filter coefficients alpha and beta (see paper)
  const double omega_cut = 2 * M_PI / T_cut;
  const double phi = 2 - std::cos(omega_cut);
  *alpha = (1.0 - std::sin(omega_cut)) / std::cos(omega_cut);
  *beta = phi - std::sqrt(phi * phi - 1.0);  // see paper
}

bool FrequencyCam::initialize(
  double minFreq, double maxFreq, double cutoffPeriod, int timeoutCycles, uint16_t debugX,
  uint16_t debugY)
{
#ifdef DEBUG  // the debug flag must be set in the header file
  debug_.open("freq.txt", std::ofstream::out);
#endif

  freq_[0] = std::max(minFreq, 0.1);
  freq_[1] = maxFreq;
  dtMax_ = 1.0 / freq_[0];
  dtMaxHalf_ = 0.5 * dtMax_;
  dtMin_ = 1.0 / (freq_[1] >= freq_[0] ? freq_[1] : 1.0);
  dtMinHalf_ = 0.5 * dtMin_;
  timeoutCycles_ = timeoutCycles;
  const double T_prefilter = cutoffPeriod;
  double alpha_prefilter, beta_prefilter;
  compute_alpha_beta(T_prefilter, &alpha_prefilter, &beta_prefilter);

  // compute IIR filter coefficient from alpha and beta (see paper)
  c_[0] = alpha_prefilter + beta_prefilter;
  c_[1] = -alpha_prefilter * beta_prefilter;
  c_p_ = 0.5 * (1 + beta_prefilter);

  debugX_ = debugX;
  debugY_ = debugY;
  return (true);
}

void FrequencyCam::initializeState(uint32_t width, uint32_t height, uint64_t t_full, uint64_t t_off)
{
  const uint32_t t = shorten_time(t_full) - 1;
#ifdef DEBUG
  timeOffset_ = (t_off / 1000) - shorten_time(t_off);  // safe high bits lost by shortening
#else
  (void)t_off;
#endif

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
    s.set_time_and_polarity(t, 0);
  }
}

std::optional<cv::Mat> FrequencyCam::makeFrequencyAndEventImage(
  cv::Mat * evImg, bool overlayEvents, bool useLogFrequency, float dt)
{
  if (hasValidTime_ || !externalTriggers_.empty()) {
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
    uint64_t difference = 1e6;
    uint64_t trigger_time = 0;
    uint64_t event_time = 0;

    std::vector<uint64_t>::iterator it = eventTimesNs_.end();
    std::vector<uint64_t>::iterator iterator_to_remove = externalTriggers_.end();
    if (hasValidTime_) {
      // Get the smallest difference
      auto it = std::min_element(eventTimesNs_.begin(), eventTimesNs_.end(), [&value = sensor_time_] (uint64_t a, uint64_t b) {
            uint64_t diff_a =  (a > value) ? a - value : value - a;
            uint64_t diff_b = (value > b) ? value - b : b - value;
            return diff_a < diff_b;
      });
      if (it != eventTimesNs_.end()) {
        difference = (*it > sensor_time_) ? *it - sensor_time_ : sensor_time_ - *it;
        trigger_time = sensor_time_;
        event_time = *it;
      }
    } else {
      uint64_t min_difference = 1e6;
      // for (const auto& trigger_time_i : externalTriggers_) {
      for (auto trigger_it = externalTriggers_.begin(); trigger_it != externalTriggers_.end(); trigger_it++) {
        it = std::min_element(eventTimesNs_.begin(), eventTimesNs_.end(), [&value = *trigger_it] (uint64_t a, uint64_t b) {
              uint64_t diff_a =  (a > value) ? a - value : value - a;
              uint64_t diff_b = (value > b) ? value - b : b - value;
              return diff_a < diff_b;
        });
        if (it != eventTimesNs_.end()) {
          difference = (*it > *trigger_it) ? *it - *trigger_it : *trigger_it - *it;
          if (difference < min_difference) {
            trigger_time = *trigger_it;
            event_time = *it;
            iterator_to_remove = trigger_it;
            // std::cout << "event time: " << event_time << std::endl;
            // std::cout << "trigger time: " << trigger_time << std::endl;
            // // externalTriggers_.erase(it);
            // std::cout << "difference: " << difference << std::endl;
            min_difference = difference;
          }
        }
      }
      difference = min_difference;
    }
    eventTimesNs_.clear();

    if (difference < 500 * 1e3) {
      // std::cout << "event time: " << event_time << std::endl;
      // std::cout << "trigger time: " << trigger_time << std::endl;
      // std::cout << "difference: " << difference << std::endl;
      if (!hasValidTime_ && !externalTriggers_.empty() && iterator_to_remove != externalTriggers_.end()) {
        externalTriggers_.erase(iterator_to_remove);
        // std::cout << "externalTriggers_.size(): " << externalTriggers_.size() << std::endl;
      }
      hasValidTime_ = false;
      // eventTimesNs_.clear();
      nrSyncMatches_++;

      if (overlayEvents) {
        *evImg = cv::Mat::zeros(height_, width_, CV_8UC1);
      }
      if (useLogFrequency) {
        return (
          overlayEvents ? makeTransformedFrequencyImage<LogTF, EventFrameUpdater>(evImg, dt)
                        : makeTransformedFrequencyImage<LogTF, NoEventFrameUpdater>(evImg, dt));
      }
      return (
        overlayEvents ? makeTransformedFrequencyImage<NoTF, EventFrameUpdater>(evImg, dt)
                      : makeTransformedFrequencyImage<NoTF, NoEventFrameUpdater>(evImg, dt));
    }
  }
  return {};
}

void FrequencyCam::getStatistics(size_t * numEvents) const { *numEvents = eventCount_; }

void FrequencyCam::resetStatistics() { eventCount_ = 0; }

void FrequencyCam::setTriggers(const std::string & triggers_file) {
  std::string line;
  std::ifstream myfile;
  myfile.open(triggers_file);

  if(!myfile.is_open()) {
    std::cerr << "Error open" << std::endl;
  }

  while(getline(myfile, line)) {
    uint64_t time_stamp = std::stoi(line);
    externalTriggers_.emplace_back(time_stamp * 1000);
  }
  // for (const auto& element: externalTriggers_) {
  //   std::cout << "Trigger time stamp: " << element << std::endl;
  // }

}

std::ostream & operator<<(std::ostream & os, const FrequencyCam::Event & e)
{
  os << std::fixed << std::setw(10) << std::setprecision(6) << e.t * 1e-6 << " "
     << static_cast<int>(e.polarity) << " " << e.x << " " << e.y;
  return (os);
}

}  // namespace frequency_cam
