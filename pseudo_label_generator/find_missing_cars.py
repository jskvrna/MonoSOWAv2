
import tensorflow.compat.v1 as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils
import os


with open("/path/to/open_waymo/OpenPCDet/data/waymo/ImageSets" + "/train.txt", 'r') as f:
    random_indexes = [line.strip() for line in f.readlines()]


seg_num_min = 0
seg_num_max = len(random_indexes)

for segment_index in range(seg_num_min, seg_num_max):
    file_name = "/path/to/open_waymo/OpenPCDet/data/waymo/raw_data/" + random_indexes[segment_index]
    file_name_2 = random_indexes[segment_index]
    dataset = tf.data.TFRecordDataset(file_name, compression_type='')
    print("Segment: ", file_name)

    for i, data in enumerate(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        if not os.path.exists("/path/to/output/frames_waymo_mvit/" + "cars/" + file_name_2 + "/" + str(i) + ".lzma"):
            print("Missing: ", file_name_2, i, "frame idx: ", segment_index)