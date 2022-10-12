# FrequencyCam: real-time frequency visualization with event based cameras

This repository has a ROS/ROS2 node for frequency analysis with event
based cameras.

## Supported platforms

Currently tested on Ubuntu 20.04 under ROS2 Galactic.


## How to build
Create a workspace (``~/ws``), clone this repo, and use ``wstool``
to pull in the remaining dependencies:

```
mkdir -p ~/ws/src
cd ~/ws
git clone https://github.com/berndpfrommer/frequency_cam.git src/frequency_cam
wstool init src src/frequency_cam/frequency_cam.rosinstall
# to update an existing space:
# wstool merge -t src src/frequency_cam/frequency_cam.rosinstall
# wstool update -t src
```

The build procedure is standard for ROS1 (catkin) and ROS2, so here is
only the ROS2 syntax:

```
cd ~/ws
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo  # (optionally add -DCMAKE_EXPORT_COMPILE_COMMANDS=1)
```

## How to use

```
ros2 launch frequency_cam frequency_cam.launch.py
```

### Parameters (see launch file):

- ``use_sim_time``: set this to true when playing from bag (and play
  bag with ``--clock``)
- ``min_frequency``: lower bound for detected frequency
- ``max_frequency``: upper bound for detected frequency
- ``cutoff_period``: number of events to use for the filter, see
  paper. When in doubt set to 5 (default).
- ``overlay_events``: mark as grey dots any events that happened
  during this frame for which no frequency could be found.
- ``publishing_frequency``: frequency (in Hz) at which frequency image is
  published (defaults to 20Hz)
- ``bag_file``: only supported under ROS2: play from bag file and
  write generated frames to disk. If no bag file is giving the node
  subscibes to topics.
  
### Topics

- ``~/image_raw``: topic under which frequency image is published.
- ``~/events``: event topic to subscribe to.


## License

This software is issued under the Apache License Version 2.0.
