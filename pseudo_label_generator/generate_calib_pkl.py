import pickle
import pykitti
import numpy as np

output_merged_frames_dir = "/path/to/output/frames/"
path_train = "/path/to/openpcdet_modified/data/kitti/ImageSets/train.txt"
train_rand = '/path/to/2D_to_3D_Annotations/3d/data/train_rand.txt'
train_map = '/path/to/2D_to_3D_Annotations/3d/data/train_mapping.txt'
complete_kitti_path = '/path/to/KITTI_complete/'

random_indexes = []
mapping_data = []

with open(train_rand, 'r') as f:
    line = f.readline().strip()
    random_indexes = line.split(',')

with open(train_map, 'r') as f:
    for line in f:
        mapping_data.append(line.strip().split(' '))

traind_idxs = []
with open(path_train, 'r') as file:
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace and newline characters
        if line.isdigit() and len(line) == 6:  # Check if the line contains a 6-digit number
            traind_idxs.append(int(line))  # Convert the number to an integer and add to the array
traind_idxs = np.array(traind_idxs)

calib_data = []
for idx in range(len(random_indexes)):
    if idx % 100 == 0:
        print("Done: ", idx, " of ", len(random_indexes))
    if idx not in traind_idxs:
        calib_data.append([])
        continue

    map_data_cur = mapping_data[int(random_indexes[idx]) - 1]

    kitti_data = pykitti.raw(complete_kitti_path, map_data_cur[0], map_data_cur[1].split("_")[-2])

    calib_data.append({'P_rect_00': kitti_data.calib.P_rect_00, 'T_cam2_velo': kitti_data.calib.T_cam2_velo, 'T_cam0_velo': kitti_data.calib.T_cam0_velo})

with open(output_merged_frames_dir + "calib2" + '.pkl', 'wb') as f:
    pickle.dump(calib_data, f)