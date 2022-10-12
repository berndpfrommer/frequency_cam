// -*-c++-*---------------------------------------------------------------------------------------
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

#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <iostream>

void usage()
{
  std::cout << "usage:" << std::endl;
  std::cout << "cpu_benchmark [-n <num_iter>]" << std::endl;
}

int main(int argc, char * argv[])
{
  int opt;
  int numIter(10000);
  while ((opt = getopt(argc, argv, "n:")) != -1) {
    switch (opt) {
      case 'n':
        numIter = atoi(optarg);
        break;
      default:
        std::cout << "unknown option: " << opt << std::endl;
        usage();
        return (-1);
        break;
    }
  }
  const auto start = std::chrono::high_resolution_clock::now();
  const float c1(0.5), c2(-0.5), cp(1.0);
  const float dp[2] = {0.2, -0.2};
  float L(0);
  float L_lag1(0);
  float L_lag2(0);
  for (int i = 0; i < numIter; i++) {
    L = L_lag1 * c1 + L_lag2 * c2 + cp * dp[i % 2];  // fused multiply add
    L_lag2 = L_lag1;
    L_lag1 = L;
    L = L_lag1 * c1 + L_lag2 * c2 + cp * dp[(i + 1) % 2];  // fused multiply add
    L_lag2 = L_lag1;
    L_lag1 = L;
    L = L_lag1 * c1 + L_lag2 * c2 + cp * dp[(i + 2) % 2];  // fused multiply add
    L_lag2 = L_lag1;
    L_lag1 = L;
    L = L_lag1 * c1 + L_lag2 * c2 + cp * dp[(i + 3) % 2];  // fused multiply add
    L_lag2 = L_lag1;
    L_lag1 = L;
    L = L_lag1 * c1 + L_lag2 * c2 + cp * dp[(i + 4) % 2];  // fused multiply add
    L_lag2 = L_lag1;
    L_lag1 = L;
    L = L_lag1 * c1 + L_lag2 * c2 + cp * dp[(i + 5) % 2];  // fused multiply add
    L_lag2 = L_lag1;
    L_lag1 = L;
    L = L_lag1 * c1 + L_lag2 * c2 + cp * dp[(i + 6) % 2];  // fused multiply add
    L_lag2 = L_lag1;
    L_lag1 = L;
    L = L_lag1 * c1 + L_lag2 * c2 + cp * dp[(i + 7) % 2];  // fused multiply add
    L_lag2 = L_lag1;
    L_lag1 = L;
  }
  const decltype(start) final = std::chrono::high_resolution_clock::now();
  auto totalDuration = std::chrono::duration_cast<std::chrono::microseconds>(final - start);
  std::cout << "end result: " << L << std::endl;
  std::cout << "million iterations per second: " << numIter / totalDuration.count() << std::endl;
  return 0;
}
