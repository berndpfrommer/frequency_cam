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
"""
reads ROS2 bag with event_array messages into array of metavision SDK events
https://github.com/ros2/rosbag2/blob/master/rosbag2_py/test/test_sequential_reader.py
"""

import time
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
        evs['t'] = (np.bitwise_and(packed, 0xFFFFFFFF)
                    + time_base).astype(np.int64) // 1000
        evs['p'] = np.right_shift(packed, 63).astype(np.int16)
        return width, height, evs
        

def read_bag(bag_path, topic, converter=EventCDConverter()):
    reader, type_map = make_reader(bag_path, topic)
    start_time = time.time()
    num_events = 0
    num_msgs = 0

    events = []
    while reader.has_next():
        (topic, data, t_rec) = reader.read_next()
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)
        width, height, evs = converter.convert(msg, msg.time_base)
        events.append(evs)
        num_events += evs.shape[0]
        num_msgs += 1
    
    dt = time.time() - start_time
    print(f'processed {num_msgs / dt} msgs/s, {num_events * 1e-6 / dt} Mev/s')
    return events, (width, height)


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
