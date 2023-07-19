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
"""Read ROS2 bag with event_camera messages into array of metavision SDK events."""

# See also::
# https://github.com/ros2/rosbag2/blob/master/rosbag2_py/test/test_sequential_reader.py

import time
import sys
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
import argparse
import numpy as np
from event_types import EventCD
from rclpy.time import Time


class BagReader():
    def __init__(self, bag_name, topics):
        bag_path = str(bag_name)
        storage_options, converter_options = self.get_rosbag_options(bag_path)
        self.reader = rosbag2_py.SequentialReader()
        self.reader.open(storage_options, converter_options)
        topic_types = self.reader.get_all_topics_and_types()
        self.type_map = {topic_types[i].name: topic_types[i].type
                         for i in range(len(topic_types))}
        storage_filter = rosbag2_py.StorageFilter(topics=[topics])
        self.reader.set_filter(storage_filter)

    def has_next(self):
        return self.reader.has_next()

    def read_next(self):
        (topic, data, t_rec) = self.reader.read_next()
        msg_type = get_message(self.type_map[topic])
        msg = deserialize_message(data, msg_type)
        return (topic, msg, t_rec)

    def get_rosbag_options(self, path, serialization_format='cdr'):
        storage_options = rosbag2_py.StorageOptions(uri=path,
                                                    storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format=serialization_format,
            output_serialization_format=serialization_format)
        return storage_options, converter_options


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


def decode_packet(data, time_base):
    # Unpack all events in the message
    # This decoding is redundant but was needed to make the old
    # code work
    packed = np.frombuffer(data, dtype=np.uint64)
    y = np.bitwise_and(
        np.right_shift(packed, 48), 0x7FFF).astype(np.uint16)
    x = np.bitwise_and(
        np.right_shift(packed, 32), 0xFFFF).astype(np.uint16)
    t = np.bitwise_and(packed, 0xFFFFFFFF) + time_base
    p = np.right_shift(packed, 63).astype(np.uint16)
    return (t, x, y, p)


def read_bag(bag_path, topic, use_sensor_time=False,
             converter=EventCDConverter(), skip=0, max_read=None):
    bag = BagReader(bag_path, topic)
    start_time = time.time()
    num_events = 0
    num_msgs = 0

    events = []
    offset = 0
    while bag.has_next():
        topic, msg, t_rec = bag.read_next()
        time_base = msg.time_base if use_sensor_time else \
            Time().from_msg(msg.header.stamp).nanoseconds
        offset = converter.offset(time_base)
        width, height, evs = converter.convert(msg, time_base)
        start_idx = max(0, min(skip - num_events, evs.shape[0]))
        end_idx = evs.shape[0] if max_read is None else \
            min(max_read - num_events, evs.shape[0])
        events.append(evs[start_idx:end_idx])
        num_events += end_idx - start_idx
        num_msgs += 1
        if end_idx < evs.shape[0]:
            break

    dt = time.time() - start_time
    print(f'took {dt:2f}s to process {num_msgs}, rate: {num_msgs / dt} ' +
          f'msgs/s, {num_events * 1e-6 / dt} Mev/s')
    return events, (width, height), offset, num_events, num_msgs


def read_as_list(fname, topic, use_sensor_time=True, skip=0, max_read=None):
    """
    Read events as a list.

    Returns tuple with:
     - 2d list (in row major order) of lists with timestamps
          and polarities as tuples
        - sensor resolution
    """
    print('reading bag: ', fname)
    print('topic: ', topic)
    print('using sensor time: ', use_sensor_time)
    event_count = [0, 0]
    if max_read is None:
        max_read = sys.maxsize
    bag = BagReader(fname, topic)
    t0 = time.time()

    data, res = None, None
    cnt, skipped = 0, 0

    while bag.has_next():
        topic, msg, t_rec = bag.read_next()
        if data is None:
            res = (int(msg.width), int(msg.height))
            data = [[] for i in range(res[0] * res[1])]
        if skipped < skip:
            skipped += len(msg.events)
            continue
        time_base = msg.time_base if use_sensor_time else \
            Time.from_msg(msg.header.stamp).nanoseconds
        t, x, y, p = decode_packet(msg.events, time_base)
        # convert to uint32 to avoid uint16 arithmetic!
        idx = y.astype(np.uint32) * res[0] + x.astype(np.uint32)
        cnt += p.shape[0]
        num_on = np.count_nonzero(p)
        event_count[0] += p.shape[0] - num_on
        event_count[1] += num_on
        for i in range(t.shape[0]):
            data[idx[i]].append((t[i], p[i]))

        if cnt > max_read:
            break
    t1 = time.time()
    dt = t1 - t0
    print(f'took {dt:.3f}s to read {cnt} events ({cnt * 1e-6 / dt:.3f} Mevs)',
          f' @ resolution: {res}\n',
          f'# of OFF: {event_count[0]:8d}\n # of ON:  {event_count[1]:8d}')

    return data, res


def read_as_array(bag_path, topic, use_sensor_time=True,
                  skip=0, max_read=None):
    """
    Read events as an array.

    returns tuple with:
    - 2d list (in row major order) of numpy arrays with timestamps
      and polarities as columns
    - sensor resolution
    """
    data, res = read_as_list(bag_path, topic, use_sensor_time, skip, max_read)
    if data is not None:
        t0 = time.time()
        # turn the data in each x, y cell into numpy array
        data_array = [np.array(d) for d in data]
        dt = time.time() - t0
        print(f'took {dt:.3f}s to convert to numpy arrays!')
        return data_array, res
    return [], (0, 0)


def merge_array_list(array_list, num_events, dtype):
    events = np.empty((num_events), dtype=dtype)
    idx = 0
    for a in array_list:
        events[idx:(idx + a.shape[0])] = a
        idx += a.shape[0]
    return events


def read_events_for_pixels(bag_path, pixel_list, topic,
                           use_sensor_time=True, skip=0, max_read=None):
    start_time = time.time()
    array_list, res, _, num_events, num_msgs = read_bag(
        bag_path, topic, use_sensor_time, EventCDConverter(), skip, max_read)

    events = merge_array_list(array_list, num_events, dtype=EventCD)

    # create empty list
    data = [[] for i in range(res[0] * res[1])]
    # fill only for requested pixel indexes
    idx = events['x'].astype(np.uint32) \
        + events['y'].astype(np.uint32) * res[0]

    for pixel in pixel_list:
        match_idx = idx == pixel
        if np.count_nonzero(match_idx) > 0:
            data[pixel] = np.column_stack(
                (events['t'][match_idx], events['p'][match_idx]))

    dt = time.time() - start_time
    print(f'events: {events.shape[0]} in {num_msgs / dt} msgs/s, ' +
          f'{num_events * 1e-6 / dt} Mev/s')
    return data, res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='read and decode events from bag.')
    parser.add_argument('--bag', '-b', action='store', default=None,
                        required=True, help='bag file to read events from')
    parser.add_argument('--topic', help='Event topic to read',
                        default='/event_camera/events', type=str)
    args = parser.parse_args()

    events, res, _, _, _ = read_bag(args.bag, args.topic)

    if len(events) > 0:
        print('test printout:\n', events[0])
