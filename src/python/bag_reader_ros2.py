import time

from rclpy.serialization import deserialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message

from event_camera_py import Decoder


class BagReader:
    """Convenience class for reading ROS2 bags."""

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

    def read_all_events(self, skip=0, max_read=None):
        """Returns list of event packets and sensor geometry"""
        start_time = time.time()
        num_events = 0
        num_msgs = 0
        all_events = []  # array of arrays
        decoder = Decoder()
        while self.has_next():
            _, msg, t_rec = self.read_next()
            decoder.decode(msg)
            events = decoder.get_cd_events()
            width, height = msg.width, msg.height
            start_idx = max(0, min(skip - num_events, events.shape[0]))
            end_idx = (
                events.shape[0]
                if max_read is None
                else min(max_read - num_events, events.shape[0])
            )
            all_events.append(events[start_idx:end_idx])
            num_events += end_idx - start_idx
            num_msgs += 1
            if end_idx < events.shape[0]:
                break
            dt = time.time() - start_time
        print(
            f'took {dt:2f}s to read {num_msgs} msgs, rate: {num_msgs / dt} '
            + f'msgs/s, {num_events * 1e-6 / dt} Mev/s'
        )
        return all_events, (width, height)
