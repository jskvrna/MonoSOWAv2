import os, shutil, sys

num_to_copy = 1856 #371
start_idx = 7481

#First we want to copy the training data.
kitti_training_path = "/path/to/OpenPCDet/data/kitti/training/"  # (redacted)
custom_kitti_path = "/path/to/openpcdet_modified_50pct/data/kitti/training/"  # (redacted)

with open("/path/to/openpcdet_modified_50pct/data/kitti/ImageSets/train.txt", 'r') as f:  # (redacted)
    train_idx = [line.strip() for line in f.readlines()]
print(train_idx)

print("Starting copying calib data")
source_calib_path = os.path.join(kitti_training_path, "calib/")
destination_calib_path = os.path.join(custom_kitti_path, "calib/")

for i, idx in enumerate(train_idx):
    if i >= num_to_copy:
        break
    shutil.copy(source_calib_path + idx + '.txt', os.path.join(destination_calib_path, f"{(start_idx + i):06d}" + '.txt'))

print("Calib data copied")
print("Starting copying velodyne data, might take longer time")

source_velodyne_path = os.path.join(kitti_training_path, "velodyne/")
destination_velodyne_path = os.path.join(custom_kitti_path, "velodyne/")

for i, idx in enumerate(train_idx):
    if i >= num_to_copy:
        break
    shutil.copy(source_velodyne_path + idx + '.bin', os.path.join(destination_velodyne_path, f"{(start_idx + i):06d}" + '.bin'))

print("Velodyne files copied")
print("Starting copying images")

source_image_path = os.path.join(kitti_training_path, "image_2/")
destination_image_path = os.path.join(custom_kitti_path, "image_2/")

for i, idx in enumerate(train_idx):
    if i >= num_to_copy:
        break
    shutil.copy(source_image_path + idx + '.png', os.path.join(destination_image_path, f"{(start_idx + i):06d}" + '.png'))

print("Images copied")
print("Starting copying labels")

source_label_path = os.path.join(kitti_training_path, "label_2/")
destination_label_path = os.path.join(custom_kitti_path, "label_2/")

for i, idx in enumerate(train_idx):
    if i >= num_to_copy:
        break
    shutil.copy(source_label_path + idx + '.txt', os.path.join(destination_label_path, f"{(start_idx + i):06d}" + '.txt'))

print("Labels copied")
print("Starting copying planes")

source_planes_path = "/path/to/OpenPCDet2/data/kitti/training/planes/"  # (redacted)
destination_planes_path = os.path.join(custom_kitti_path, "planes/")

for i, idx in enumerate(train_idx):
    if i >= num_to_copy:
        break
    shutil.copy(source_planes_path + idx + '.txt', os.path.join(destination_planes_path, f"{(start_idx + i):06d}" + '.txt'))

print("Planes copied")

# Define the start and end numbers
start = start_idx
end = start_idx + num_to_copy

# Open the file in append mode
with open('/path/to/openpcdet_modified_50pct/data/kitti/ImageSets/train.txt', 'a') as file:  # (redacted)
    # Write the numbers to the file
    file.write("\n")
    for i in range(start, end):
        file.write(f"{i:06d}\n")
