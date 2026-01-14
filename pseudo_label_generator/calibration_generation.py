import os
import tensorflow.compat.v1 as tf
import numpy as np
import pickle

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

tf.enable_eager_execution()

waymo_path = "/path/to/open_waymo/"  # (redacted)
frames_path = "/path/to/output/frames_waymo/"  # (redacted)

with open("/path/to/open_waymo/ImageSets/train.txt", 'r') as f:  # (redacted)
    random_indexes = [line.strip() for line in f.readlines()]

if not os.path.isdir(frames_path + "calib/"):
    os.mkdir(frames_path + "calib/")

for segment_index in range(0, len(random_indexes)):
    file_name = "/path/to/open_waymo/raw_data/" + random_indexes[segment_index]  # (redacted)
    dataset = tf.data.TFRecordDataset(file_name, compression_type='')
    file_name = random_indexes[segment_index]

    if not os.path.isdir(frames_path + "calib/" + file_name):
        os.mkdir(frames_path + "calib/" + file_name)

    for i, data in enumerate(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        calibration = [cc for cc in frame.context.camera_calibrations]

        with open(frames_path + "calib/" + file_name + "/" + str(i) + '.pkl', 'wb') as f:
            pickle.dump(calibration, f)
        '''
        with open(frames_path + "calib/" + file_name + "/" + str(i) + '.pkl', 'rb') as f:
            loaded = pickle.load(f)

            cam_intrinsic = []
            cam_size = []

            for i in range(len(loaded)):
                cam_intrinsic.append(np.array(loaded[i].intrinsic))
                cam_size.append(np.array([loaded[i].height, loaded[i].width]))
            print(cam_intrinsic)
            print(cam_size)
            break
            
        '''


