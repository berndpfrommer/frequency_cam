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

import argparse  # noqa: I100
import sys  # noqa: I100

import numpy as np  # noqa: I100
from rclpy.serialization import deserialize_message  # noqa: I100
import rosbag2_py  # noqa: I100
from rosidl_runtime_py.utilities import get_message  # noqa: I100

from event_camera_py import Decoder  # noqa: I100
from event_camera_py import UniqueDecoder  # noqa: I100


class BagReader:
    """Reads ros2 bags."""

    def __init__(self, bag_name, topics):
        bag_path = str(bag_name)
        storage_options, converter_options = self.get_rosbag_options(bag_path)
        self.reader = rosbag2_py.SequentialReader()
        self.reader.open(storage_options, converter_options)
        topic_types = self.reader.get_all_topics_and_types()
        self.type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}
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
        storage_options = rosbag2_py.StorageOptions(uri=path)
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format=serialization_format,
            output_serialization_format=serialization_format,
        )
        return storage_options, converter_options


def read_events(
    bag_name, topic, max_read=sys.maxsize, skip=0
) -> tuple[list[np.array], tuple, int]:
    bag = BagReader(bag_name, topic)
    decoder = Decoder()

    all_events = []
    event_count = 0
    width = None
    height = None
    while bag.has_next():
        topic, msg, t_rec = bag.read_next()
        decoder.decode_array(
            msg.encoding,
            msg.width,
            msg.height,
            msg.time_base,
            np.frombuffer(msg.events, dtype=np.uint8),
        )
        if width is None:
            width = msg.width
            height = msg.height

        evs = decoder.get_cd_events()
        if evs.size:
            event_count += evs.shape[0]
            if event_count < skip:
                continue
            if event_count <= max_read:
                all_events.append(evs)
            else:
                event_count -= evs.shape[0]
                all_events.append(evs[0 : (max_read - event_count)])  # noqa: E203
                break
    return all_events, (width, height), event_count


def read_unique_events(
    bag_name, topic, max_read=sys.maxsize, skip=0
) -> tuple[list[np.array], tuple, int]:
    bag = BagReader(bag_name, topic)
    decoder = UniqueDecoder()
    # decoder = Decoder()

    all_events = []
    event_count = 0
    width = None
    height = None
    while bag.has_next():
        topic, msg, t_rec = bag.read_next()
        decoder.decode(msg)

        if width is None:
            width = msg.width
            height = msg.height

        evsl = decoder.get_cd_event_packets()
        for evs in evsl:
            if evs.size:
                event_count += evs.shape[0]
                if event_count < skip:
                    continue
                if event_count <= max_read:
                    all_events.append(evs)
                else:
                    event_count -= evs.shape[0]
                    headroom = max_read - event_count
                    all_events.append(evs[0:headroom])
                    break
    return all_events, (width, height), event_count


def filter_events(all_events, pixel_list):
    num_pix = len(pixel_list)
    filtered_events = [np.empty((0, 2), np.int32)] * num_pix
    for evs in all_events:
        for i, p in enumerate(pixel_list):
            ev = evs[(evs['x'] == p[0]) & (evs['y'] == p[1])]
            if ev.shape[0] > 0:
                a = np.stack((ev['t'], ev['p']), axis=1)
                filtered_events[i] = np.append(filtered_events[i], a, axis=0)
    return filtered_events


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='read and decode events from bag.')
    parser.add_argument(
        '--bag',
        '-b',
        action='store',
        default=None,
        required=True,
        help='bag file to read events from',
    )
    parser.add_argument(
        '--topic', help='Event topic to read', default='/event_camera/events', type=str
    )
    parser.add_argument(
        '--max_read', help='max number of events to read', default=sys.maxsize, type=int
    )
    parser.add_argument(
        '--skip', help='number of events to skip at beginning', default=0, type=int
    )
    args = parser.parse_args()

    events, res, count = read_events(args.bag, args.topic, args.max_read, args.skip)

    if len(events) > 0:
        print('test printout:\n', events[0])
