#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright 2022 Bernd Pfrommer <bernd.pfrommer@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
"""Read ROS2 bag with event_array messages into array of metavision SDK events."""

# See also::
# https://github.com/ros2/rosbag2/blob/master/rosbag2_py/test/test_sequential_reader.py

import time
import rclpy.time
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
import argparse
import numpy as np
from event_types import EventCD


def get_rosbag_options(path, serialization_format='cdr'):
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3')

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format)

    return storage_options, converter_options


def make_reader(bag_path, topic):
    storage_options, converter_options = get_rosbag_options(bag_path)

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()

    type_map = {topic_types[i].name: topic_types[i].type
                for i in range(len(topic_types))}

    storage_filter = rosbag2_py.StorageFilter(topics=[topic])
    reader.set_filter(storage_filter)
    return reader, type_map


class ArrayConverter():
    def convert(msg, time_base):
        width = msg.width
        height = msg.height
        # unpack all events in the message
        packed = np.frombuffer(msg.events, dtype=np.uint64)
        y = np.bitwise_and(
            np.right_shift(packed, 48), 0x7FFF).astype(np.uint16)
        x = np.bitwise_and(
            np.right_shift(packed, 32), 0xFFFF).astype(np.uint16)
        t = (np.bitwise_and(packed, 0xFFFFFFFF)
             + time_base).astype(np.int64) // 1000
        p = np.right_shift(packed, 63).astype(np.int16)
        return width, height, (x, y, p, t)


class EventCDConverter():
    def convert(self, msg, time_base):
        width = msg.width
        height = msg.height
        # unpack all events in the message
        packed = np.frombuffer(msg.events, dtype=np.uint64)
        evs = np.empty(len(msg.events) // 8, dtype=EventCD)
        evs['y'] = np.bitwise_and(
            np.right_shift(packed, 48), 0x7FFF).astype(np.uint16)
        evs['x'] = np.bitwise_and(
            np.right_shift(packed, 32), 0xFFFF).astype(np.uint16)
        # empirically cannot use more than 48 bits of the full time stamp
        evs['t'] = ((np.bitwise_and(packed, 0xFFFFFFFF)
                    + time_base).astype(np.int64) // 1000) & 0xFFFFFFFFFFF
        evs['p'] = np.right_shift(packed, 63).astype(np.int16)
        return width, height, evs

    def offset(self, time_base):
        return ((time_base // 1000) & ~0xFFFFFFFFFFF)


def read_bag(bag_path, topic, use_sensor_time=False, converter=EventCDConverter()):
    reader, type_map = make_reader(bag_path, topic)
    start_time = time.time()
    num_events = 0
    num_msgs = 0

    events = []
#     prev_t = 0
#     prev_tb = 0
#    prev_dec = 0
    offset = 0
    while reader.has_next():
        (topic, data, t_rec) = reader.read_next()
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)
        time_base = msg.time_base if use_sensor_time else \
            (rclpy.time.Time().from_msg(msg.header.stamp).nanoseconds)
        offset = converter.offset(time_base)
        width, height, evs = converter.convert(msg, time_base)
#        print('time base: ', msg.time_base,
#              rclpy.time.Time().from_msg(msg.header.stamp).nanoseconds, ' diff: ',
#              rclpy.time.Time().from_msg(msg.header.stamp).nanoseconds - prev_t,
#              msg.time_base - prev_tb, evs['t'][0] - prev_dec)
#        prev_t = rclpy.time.Time().from_msg(msg.header.stamp).nanoseconds
#        prev_tb = msg.time_base
#        prev_dec = evs['t'][0]
        events.append(evs)
        num_events += evs.shape[0]
        num_msgs += 1

    dt = time.time() - start_time
    print(f'took {dt:2f}s to process {num_msgs}, rate: {num_msgs / dt} ' +
          f'msgs/s, {num_events * 1e-6 / dt} Mev/s')
    return events, (width, height), offset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='read and decode events from bag.')
    parser.add_argument('--bag', '-b', action='store', default=None,
                        required=True, help='bag file to read events from')
    parser.add_argument('--topic', help='Event topic to read',
                        default='/event_camera/events', type=str)
    args = parser.parse_args()

    events, res = read_bag(args.bag, args.topic)

    if len(events) > 0:
        print('test printout:\n', events[0])
