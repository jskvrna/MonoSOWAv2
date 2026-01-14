import os, shutil, sys

# Check if the folder path argument is provided
if len(sys.argv) < 2:
    print("Please provide the folder path as an argument.")
    sys.exit(1)

replace_only_labels = True

#First we want to copy the training data.
kitti_training_path = "/path/to/datasets/KITTI/object_detection/training"
custom_kitti_path = "/path/to/OpenPCDet/data/kitti/training"
complete_kitti_path = "/path/to/datasets/KITTI/complete_sequences/"
our_labels_path = sys.argv[1]

if not replace_only_labels:
    print("Starting copying calib data")
    source_calib_path = os.path.join(kitti_training_path, "calib")
    destination_calib_path = os.path.join(custom_kitti_path, "calib")

    os.mkdir(destination_calib_path)

    for calib_file in sorted(os.listdir(source_calib_path)):
        shutil.copy(os.path.join(source_calib_path, calib_file), destination_calib_path)

    print("Calib data copied")
    print("Starting copying velodyne data, might take longer time")

    source_velodyne_path = os.path.join(kitti_training_path, "velodyne")
    destination_velodyne_path = os.path.join(custom_kitti_path, "velodyne")

    os.mkdir(destination_velodyne_path)

    for velodyne_file in sorted(os.listdir(source_velodyne_path)):
        shutil.copy(os.path.join(source_velodyne_path, velodyne_file), destination_velodyne_path)

    print("Velodyne files copied")
    print("Starting copying images")

    source_image_path = os.path.join(kitti_training_path, "image_2")
    destination_image_path = os.path.join(custom_kitti_path, "image_2")

    os.mkdir(destination_image_path)

    for image_file in sorted(os.listdir(source_image_path)):
        shutil.copy(os.path.join(source_image_path, image_file), destination_image_path)

    print("Images copied")
    print("Starting copying labels")

    source_label_path = os.path.join(kitti_training_path, "label_2")
    destination_label_path = os.path.join(custom_kitti_path, "label_2")

    os.mkdir(destination_label_path)

    for label_file in sorted(os.listdir(source_label_path)):
        shutil.copy(os.path.join(source_label_path, label_file), destination_label_path)

    print("Labels copied")
    print("Starting copying planes")

    source_planes_path = "/path/to/OpenPCDet/data/kitti/training/planes"
    destination_planes_path = os.path.join(custom_kitti_path, "planes")

    os.mkdir(destination_planes_path)

    for planes_file in sorted(os.listdir(source_planes_path)):
        shutil.copy(os.path.join(source_planes_path, planes_file), destination_planes_path)

    print("Planes copied")


with open(train_rand, 'r') as f:
    line = f.readline().strip()
    random_indexes = line.split(',')

mapping_data = []
with open(train_map, 'r') as f:
    for line in f:
        mapping_data.append(line.strip().split(' '))

print("Copying our data")
orig_data_offset = 7481
#Now lets start copying our data for training.
for pic_index in range(len(random_indexes)):
    if pic_index % 100 == 0:
        print(pic_index)

    map_data_cur = mapping_data[int(random_indexes[int(pic_index)])]
    velo_path = os.path.join(complete_kitti_path, map_data_cur[0], map_data_cur[1], 'velodyne_points/data/',
                                  map_data_cur[2] + '.bin')
    image_path = os.path.join(complete_kitti_path, map_data_cur[0], map_data_cur[1], 'image_02/data/',
                             map_data_cur[2] + '.png')
    label_path = os.path.join(our_labels_path, str(pic_index).zfill(10) + '.txt')
    calib_path = os.path.join(kitti_training_path, "calib/000002.txt")
train_rand = '/path/to/2D_to_3D_Annotations/3d/data/rand.txt'
train_map = '/path/to/2D_to_3D_Annotations/3d/data/mapping.txt'

    plane_path = "/path/to/OpenPCDet/data/kitti/training/planes/000002.txt"

    new_name = orig_data_offset + pic_index

    velo_dest_path = os.path.join(custom_kitti_path, "velodyne/" + str(new_name).zfill(6) + ".bin")
    image_dest_path = os.path.join(custom_kitti_path, "image_2/" + str(new_name).zfill(6) + ".png")
    label_dest_path = os.path.join(custom_kitti_path, "label_2/" + str(new_name).zfill(6) + ".txt")
    calib_dest_path = os.path.join(custom_kitti_path, "calib/" + str(new_name).zfill(6) + ".txt")
    plane_dest_path = os.path.join(custom_kitti_path, "planes/" + str(new_name).zfill(6) + ".txt")

    if not replace_only_labels:
        shutil.copy(velo_path, velo_dest_path)
        shutil.copy(image_path, image_dest_path)
        shutil.copy(calib_path, calib_dest_path)
        shutil.copy(plane_path, plane_dest_path)

    shutil.copy(label_path, label_dest_path)


val_numbers = list(range(7481))
train_numbers = list(range(7481, len(random_indexes) + 7481))
train_numbers_formatted = [str(num).zfill(6) for num in train_numbers]
val_numbers_formatted = [str(num).zfill(6) for num in val_numbers]
# Write train numbers to train.txt
with open("ImageSets/train.txt", "w") as train_file:
    train_file.write("\n".join(map(str, sorted(train_numbers_formatted))))

# Write validation numbers to val.txt
with open("ImageSets/val.txt", "w") as val_file:
    val_file.write("\n".join(map(str, sorted(val_numbers_formatted))))

#Now we need to prepare the labels. Delete the score at the end of each line and also if the file is empty we need to fill something
# Get a list of all .txt files in the folder
folder_path = os.path.join(custom_kitti_path, "label_2/")
txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
print("Cleaning and preparing the labels")
# Process each text file
for file_name in txt_files:
    file_path = os.path.join(folder_path, file_name)

    with open(file_path, 'r+') as file:
        lines = file.readlines()
        file.seek(0)  # Reset file position to the beginning

        for line in lines:
            values = line.strip().split(' ')

            if len(values) > 15:
                values = values[:15]  # Remove the last value if there are more than 16

            file.write(' '.join(values) + '\n')  # Write the modified line back to the file

        if not lines:
            zeros_line = "DontCare -1 -1 -10 0.00 0.00 0.00 0.00 -1 -1 -1 -1000 -1000 -1000 -10"
            file.write(zeros_line + '\n')  # Write the zeros line if the file is empty

        file.truncate()  # Remove any remaining content after processing

print("Removing old database")
if os.path.exists("gt_database"):
    shutil.rmtree("gt_database")

files_to_remove = ['kitti_dbinfos_train.pkl', 'kitti_infos_test.pkl', 'kitti_infos_train.pkl', 'kitti_infos_trainval.pkl', 'kitti_infos_val.pkl']

for filename in files_to_remove:
    if os.path.exists(filename):
        os.remove(filename)
