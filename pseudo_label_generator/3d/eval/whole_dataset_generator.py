import shutil, os

random_indexes = []

#Read the file which contains the ordering of the scenes
with open('/path/to/KITTI/object_detection/devkit_object/mapping/train_rand.txt', 'r') as f:  # (redacted)
    line = f.readline().strip()

    random_indexes = line.split(',')

mapping_data = []

with open('/path/to/KITTI/object_detection/devkit_object/mapping/train_mapping.txt', 'r') as f:  # (redacted)
    for line in f:
        mapping_data.append(line.strip().split(' '))

index = 0
for rnd_idx in random_indexes:
    map_data_cur = mapping_data[int(rnd_idx)]
    path_to_folder = '/path/to/KITTI/complete_sequences/' + map_data_cur[0] + '/' + map_data_cur[1] + '/'

    if not os.path.exists('/path/to/custom_kitti/image_2_add/' + f'{index:0>6}'):
        f_timestamp = open(path_to_folder + 'oxts/' + 'timestamps.txt', 'r')  # (redacted)
        timestamps = []
        for line in f_timestamp:
            timestamps.append(line.strip().split(' '))

        f_time_out = open('/path/to/custom_kitti/timestamps/' + str(index) + '.txt', 'w')  # (redacted)

        os.mkdir('/path/to/custom_kitti/image_2_add/' + f'{index:0>6}')
        os.mkdir('/path/to/custom_kitti/velodyne_add/' + f'{index:0>6}')
        os.mkdir('/path/to/custom_kitti/odx_add/' + f'{index:0>6}')

        file_number = int(map_data_cur[2])
        for i in range(-30,30):
            #First copy the img
            path_to_img = path_to_folder + 'image_02/data/' + f'{file_number+i:0>10}' + '.png'
            path_to_velo = path_to_folder + 'velodyne_points/data/' + f'{file_number+i:0>10}' + '.bin'
            path_to_odo = path_to_folder + 'oxts/data/' + f'{file_number+i:0>10}' + '.txt'

            path_save_img = '/path/to/custom_kitti/image_2_add/' + f'{index:0>6}' + '/' + str(i) + '.png'  # (redacted)
            path_save_velo = '/path/to/custom_kitti/velodyne_add/' + f'{index:0>6}' + '/' + str(i) + '.bin'  # (redacted)
            path_save_odo = '/path/to/custom_kitti/odx_add/' + f'{index:0>6}' + '/' + str(i) + '.txt'  # (redacted)

            if os.path.isfile(path_to_img) and os.path.isfile(path_to_velo) and os.path.isfile(path_to_odo):
                shutil.copy(path_to_img, path_save_img)
                shutil.copy(path_to_velo, path_save_velo)
                shutil.copy(path_to_odo, path_save_odo)

                f_time_out.write(str(i) + " " + str(timestamps[file_number+i][1]) + "\n")

        f_timestamp.close()
        f_time_out.close()
    index += 1
