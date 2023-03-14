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

#ifndef FREQUENCY_CAM__FREQUENCY_CAM_H_
#define FREQUENCY_CAM__FREQUENCY_CAM_H_

#include <event_array_codecs/event_processor.h>

#include <atomic>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <eigen3/Eigen/Dense>

// #define DEBUG

namespace frequency_cam
{

int roundUp(const int numToRound, const int multiple);

class FrequencyCam : public event_array_codecs::EventProcessor
{
public:
  FrequencyCam() : csv_file_("frequency_points.csv") { csv_file_ << "timestamp,x,y,frequency.\n"; }
  ~FrequencyCam();

  FrequencyCam(const FrequencyCam &) = delete;
  FrequencyCam & operator=(const FrequencyCam &) = delete;

  // ------------- inherited from EventProcessor
  inline void eventCD(uint64_t sensor_time, uint16_t ex, uint16_t ey, uint8_t polarity) override
  {
    // std::cout << "Event: time stamp: " << sensor_time << std::endl;
    Event e(shorten_time(sensor_time), ex, ey, polarity);
    updateState(&state_[e.y * width_ + e.x], e);
    lastEventTime_ = e.t;
    lastEventTimeNs_ = sensor_time;
    eventCount_++;
    eventTimesNs_.emplace_back(sensor_time);
  }
  void eventExtTrigger(uint64_t sensor_time, uint8_t edge, uint8_t /*id*/) override
  {
    if (!eventExtTriggerInitialized_) {
      lasteExternalEdge_ = edge;
      eventExtTriggerInitialized_ = true;
    } else {
      if (lasteExternalEdge_ == edge) {
        std::cerr << "Missed an external trigger edge" << std::endl;
      }
      // Take second event (falling edge) since this is the end of the exposure time
      // of the FB camera
      if (edge == 0) {
        sensor_time_ = sensor_time;
        hasValidTime_ = true;
        nrExtTriggers_++;
      }
      lasteExternalEdge_ = edge;
    }
    // std::cout << "External trigger: sensor_time: " << sensor_time << ", edge: " << std::to_string(edge) << std::endl;
  }

  void finished() override {}
  void rawData(const char *, size_t) override {}
  // ------------- end of inherited from EventProcessor

  bool initialize(
    double minFreq, double maxFreq, double cutoffPeriod, int timeoutCycles, uint16_t debugX,
    uint16_t debugY);

  void initializeState(uint32_t width, uint32_t height, uint64_t t_first, uint64_t t_off);

  // returns frequency image
  std::optional<cv::Mat> makeFrequencyAndEventImage(
    cv::Mat * eventImage, bool overlayEvents, bool useLogFrequency, float dt);

  void getStatistics(size_t * numEvents) const;
  void resetStatistics();

  void setTriggers(const std::string & triggers_file);

private:
  struct Event  // event representation for convenience
  {
    explicit Event(uint32_t ta = 0, uint16_t xa = 0, uint16_t ya = 0, bool p = false)
    : t(ta), x(xa), y(ya), polarity(p)
    {
    }
    // variables
    uint32_t t;
    uint16_t x;
    uint16_t y;
    bool polarity;
  };
  friend std::ostream & operator<<(std::ostream & os, const Event & e);

  // define the per-pixel filter state
  typedef float variable_t;
  typedef uint32_t state_time_t;
  struct State
  {
    inline bool polarity() const { return (last_t_pol & (1 << 31)); }
    inline state_time_t lastTime() const { return (last_t_pol & ~(1 << 31)); }
    inline void set_time_and_polarity(state_time_t t, bool p)
    {
      last_t_pol = (static_cast<uint8_t>(p) << 31) | (t & ~(1 << 31));
    }
    // ------ variables
    state_time_t t_flip_up_down;  // time of last flip
    state_time_t t_flip_down_up;  // time of last flip
    variable_t L_km1;             // brightness lagged once
    variable_t L_km2;             // brightness lagged twice
    variable_t period;            // estimated period
    state_time_t last_t_pol;      // last polarity and time
  };

  inline void updateState(State * state, const Event & e)
  {
    State & s = *state;
    // raw change in polarity, will be 0 or +-1
    const float dp = e.polarity - s.polarity();
    // run the filter (see paper)
    const auto L_k = c_[0] * s.L_km1 + c_[1] * s.L_km2 + c_p_ * dp;
    if (L_k < 0 && s.L_km1 > 0) {
      // approximate reconstructed brightness crossed zero from above
      const float dt_ud = (e.t - s.t_flip_up_down) * 1e-6;  // "ud" = up_down
      if (dt_ud >= dtMin_ && dt_ud <= dtMax_) {
        // up-down (most accurate) dt is within valid range, use it!
        s.period = dt_ud;
      } else {
        // full period looks screwy, but maybe period can be computed from half cycle
        const float dt_du = (e.t - s.t_flip_down_up) * 1e-6;
        if (s.period > 0) {
          // If there already is a valid period established, check if it can be cleared out.
          // If it's not stale, don't update it because it would mean overriding a full-period
          // estimate with a half-period estimate.
          const float to = s.period * timeoutCycles_;  // timeout
          if (dt_ud > to && dt_du > 0.5 * to) {
            s.period = 0;  // stale pixel
          }
        } else {
          if (dt_du >= dtMinHalf_ && dt_du <= dtMaxHalf_) {
            // half-period estimate seems reasonable, make do with it
            s.period = 2 * dt_du;
          }
        }
      }
      s.t_flip_up_down = e.t;
    } else if (L_k > 0 && s.L_km1 < 0) {
      // approximate reconstructed brightness crossed zero from below
      const float dt_du = (e.t - s.t_flip_down_up) * 1e-6;  // "du" = down_up
      if (dt_du >= dtMin_ && dt_du <= dtMax_ && s.period <= 0) {
        // only use down-up transition if there is no established period
        // because it is less accurate than up-down transition
        s.period = dt_du;
      } else {
        const float dt_ud = (e.t - s.t_flip_up_down) * 1e-6;
        if (s.period > 0) {
          // If there already is a valid period established, check if it can be cleared out.
          // If it's not stale, don't update it because it would mean overriding a full-period
          // estimate with a half-period estimate.
          const float to = s.period * timeoutCycles_;  // timeout
          if (dt_du > to && dt_ud > 0.5 * to) {
            s.period = 0;  // stale pixel
          }
        } else {
          if (dt_ud >= dtMinHalf_ && dt_ud <= dtMaxHalf_) {
            // half-period estimate seems reasonable, make do with it
            s.period = 2 * dt_ud;
          }
        }
      }
      s.t_flip_down_up = e.t;
    }
#ifdef DEBUG
    if (e.x == debugX_ && e.y == debugY_) {
      const double dt = (e.t - std::max(s.t_flip_up_down, s.t_flip_down_up)) * 1e-6;
      debug_ << e.t + timeOffset_ << " " << dp << " " << L_k << " " << s.L_km1 << " " << s.L_km2
             << " " << dt << " " << s.period << " " << dtMin_ << " " << dtMax_ << std::endl;
    }
#endif
    s.L_km2 = s.L_km1;
    s.L_km1 = L_k;
    s.set_time_and_polarity(e.t, e.polarity);
  }

  struct NoTF
  {
    static double tf(double f) { return (f); }
    static double inv(double f) { return (f); }
  };
  struct LogTF
  {
    static double tf(double f) { return (std::log10(f)); }
    static double inv(double f) { return (std::pow(10.0, f)); }
  };
  struct EventFrameUpdater
  {
    static void update(cv::Mat * img, int ix, int iy, double dt, double dtMax)
    {
      if (dt < dtMax) {
        img->at<uint8_t>(iy, ix) = 255;
      }
    }
  };

  struct NoEventFrameUpdater
  {
    static void update(cv::Mat *, int, int, double, double) {}
  };

  template <class T, class U>
  cv::Mat makeTransformedFrequencyImage(cv::Mat * eventFrame, float eventImageDt)
  {
    std::map<double, std::vector<std::tuple<int, int>>> frequency_points;
    // const int min_range_1 = 2800;
    // const int max_range_1 = 3200;
    // const int min_range_2 = 3800;
    // const int max_range_2 = 4200;
    const int min_range_2 = 1500;
    const int max_range_2 = 2500;
    cv::Mat rawImg(height_, width_, CV_32FC1, 0.0);
    const double maxDt = 1.0 / freq_[0] * timeoutCycles_;
    const double minFreq = T::tf(freq_[0]);
    for (uint32_t iy = 0; iy < height_; iy++) {
      for (uint32_t ix = 0; ix < width_; ix++) {
        const size_t offset = iy * width_ + ix;
        const State & state = state_[offset];
        // compute time since last touched
        const double dtEvent = (lastEventTime_ - state.lastTime()) * 1e-6;
        U::update(eventFrame, ix, iy, dtEvent, eventImageDt);
        if (state.period > 0) {
          const double dt =
            (lastEventTime_ - std::max(state.t_flip_up_down, state.t_flip_down_up)) * 1e-6;
          const double f = 1.0 / std::max(state.period, decltype(state.period)(1e-6));
          // filter out any pixels that have not been updated recently
          if (dt < maxDt * timeoutCycles_ && dt * f < timeoutCycles_) {
            auto frequency = std::max(T::tf(f), minFreq);
            if (
              // (frequency > min_range_1 && frequency < max_range_1) ||
              (frequency > min_range_2 && frequency < max_range_2)) {
              frequency = roundUp(frequency, 500);
              if (1 == frequency_points.count(frequency)) {
                frequency_points[frequency].emplace_back(ix, iy);
              } else {
                std::vector<std::tuple<int, int>> point{{ix, iy}};
                frequency_points[frequency] = point;
              }
            }
            rawImg.at<float>(iy, ix) = frequency;
          } else {
            rawImg.at<float>(iy, ix) = 0;  // mark as invalid
          }
        }
      }
    }

    std::vector<std::tuple<int, int, int>> filtered_frequency_points;
    std::vector<std::size_t> number_of_points;
    for (const auto & frequency_point : frequency_points) {
      std::vector<std::size_t> assigned_indices;
      for (std::size_t i = 0; i < frequency_point.second.size(); ++i) {
        if (std::count(assigned_indices.begin(), assigned_indices.end(), i)) {
          continue;
        }

        std::vector<std::size_t> candidate_indices;
        std::vector<double> x_values;
        std::vector<double> y_values;
        auto x = std::get<0>(frequency_point.second.at(i));
        auto y = std::get<1>(frequency_point.second.at(i));

        std::size_t counts = 0;
        for (std::size_t j = i + 1; j < frequency_point.second.size(); ++j) {
          if (std::count(assigned_indices.begin(), assigned_indices.end(), j)) {
            continue;
          }
          auto x_candidate = std::get<0>(frequency_point.second.at(j));
          auto y_candidate = std::get<1>(frequency_point.second.at(j));

          if ((std::fabs(x - x_candidate) < 20) && (std::fabs(y - y_candidate) < 20)) {
            candidate_indices.emplace_back(j);
            x_values.emplace_back(x_candidate);
            y_values.emplace_back(y_candidate);
            counts++;
          }
        }

        if (10 < counts) {
          candidate_indices.emplace_back(i);
          x_values.emplace_back(x);
          y_values.emplace_back(y);
          auto mean_x = std::reduce(x_values.begin(), x_values.end());
          mean_x /= x_values.size();
          auto mean_y = std::reduce(y_values.begin(), y_values.end());
          mean_y /= y_values.size();

          bool insert = true;
          for (const auto & point : filtered_frequency_points) {
            x = std::get<0>(point);
            y = std::get<1>(point);

            if ((std::fabs(x - mean_x) < 30) && (std::fabs(y - mean_y) < 30)) {
              insert = false;
              break;
            }
          }
          if (insert) {
            filtered_frequency_points.emplace_back(mean_x, mean_y, frequency_point.first);
            number_of_points.emplace_back(x_values.size());
          }

          assigned_indices.insert(
            assigned_indices.end(), candidate_indices.begin(), candidate_indices.end());
        }
      }
    }

    // if (!filtered_frequency_points.empty()) {
    if (filtered_frequency_points.size() == 3) {

      // Eigen::Matrix3f matrix;
      // matrix << std::get<0>(filtered_frequency_points.at(0)), std::get<1>(filtered_frequency_points.at(0)), 1.0,
      //           std::get<0>(filtered_frequency_points.at(1)), std::get<1>(filtered_frequency_points.at(1)), 1.0,
      //           std::get<0>(filtered_frequency_points.at(2)), std::get<1>(filtered_frequency_points.at(2)), 1.0;
      // Eigen::FullPivLU<Eigen::Matrix3f> lu_decomp(matrix);
      // auto rank = lu_decomp.rank();
      // std::cout << "Rank: " << rank << std::endl;
      // // result = rank([x2-x1, y2-y1; x3-x1, y3-y1]) < 2;
    if (!filtered_frequency_points.empty()) {
      std::cout << "Filtered points:" << std::endl;
      std::cout << "time stamp: " << lastEventTime_ << std::endl;
      //for (const auto & filtered_point : filtered_points) {
      for (std::size_t i = 0; i < filtered_frequency_points.size(); ++i) {
        std::cout << "x: " << std::get<0>(filtered_frequency_points.at(i))
                  << ", y: " << std::get<1>(filtered_frequency_points.at(i))
                  << ", frequency: " << std::get<2>(filtered_frequency_points.at(i))
                  << ", number of points: " << number_of_points.at(i) << std::endl;
        csv_file_ << lastEventTime_ << "," << std::get<0>(filtered_frequency_points.at(i)) << ","
                  << std::get<1>(filtered_frequency_points.at(i)) << ","
                  << std::get<2>(filtered_frequency_points.at(i)) << "\n";
        cv::circle(
          rawImg,
          {std::get<0>(filtered_frequency_points.at(i)),
           std::get<1>(filtered_frequency_points.at(i))},
          2, CV_RGB(4500, 4500, 4500), 4);
      }

      nrDetectedWands_++;
    }

    return (rawImg);
  }

  static inline uint32_t shorten_time(uint64_t t)
  {
    return (static_cast<uint32_t>((t / 1000) & 0xFFFFFFFF));
  }

  // ------ variables ----
  State * state_{0};
  double freq_[2]{-1.0, -1.0};  // frequency range
  double tfFreq_[2]{0, 1.0};    // transformed frequency range
  uint32_t width_{0};           // image width
  uint32_t height_{0};          // image height
  uint64_t eventCount_{0};
  uint32_t lastEventTime_;
  std::vector<uint64_t> eventTimesNs_{0};
  // ---------- variables for state update
  variable_t c_[2];
  variable_t c_p_{0};
  variable_t dtMin_{0};
  variable_t dtMax_{1.0};
  variable_t dtMinHalf_{0};
  variable_t dtMaxHalf_{0.5};
  variable_t timeoutCycles_{2.0};  // how many silent cycles until freq is invalid
  //
  // ------------------ debugging stuff
  std::ofstream debug_;
  uint16_t debugX_{0};
  uint16_t debugY_{0};
  std::atomic<bool> hasValidTime_{false};
  uint64_t timeOffset_{0};
  uint64_t sensor_time_;
  bool eventExtTriggerInitialized_{false};
  std::size_t nrExtTriggers_{0};
  std::size_t nrSyncMatches_{0};
  std::size_t nrDetectedWands_{0};
  uint8_t lasteExternalEdge_;

  std::ofstream csv_file_;
  std::vector<uint64_t> externalTriggers_;
  uint64_t lastEventTimeNs_;
};
std::ostream & operator<<(std::ostream & os, const FrequencyCam::Event & e);
}  // namespace frequency_cam
#endif  // FREQUENCY_CAM__FREQUENCY_CAM_H_
