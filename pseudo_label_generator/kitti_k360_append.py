import os
import glob
import shutil

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# User to update this path
kitti_root = "/path/to/kitti_root" # REPLACE THIS with the actual path to KITTI dataset

# Paths from reference script
k360_root = "/path/to/KITTI/"  # (redacted)
output_folder = "/path/to/output/kitti_combined/"  # (redacted)
k360_pseudo_label_path = "/path/to/output/labels_k360/"  # (redacted)

# K360 Sequences
training_sequences = ["2013_05_28_drive_0000_sync",
                      "2013_05_28_drive_0002_sync",
                      "2013_05_28_drive_0004_sync",
                      "2013_05_28_drive_0005_sync",
                      "2013_05_28_drive_0006_sync",
                      "2013_05_28_drive_0009_sync"]

validation_sequences = ["2013_05_28_drive_0003_sync",
                        "2013_05_28_drive_0007_sync"]

testing_sequences = ["2013_05_28_drive_0010_sync"]

# ---------------------------------------------------------
# SETUP
# ---------------------------------------------------------

if kitti_root == "/path/to/kitti_root":
    print("WARNING: kitti_root is set to a placeholder. Please update the script with the correct path.")
    # We continue, assuming the user might update it or run it to see what happens.
    # But it will likely fail if the path doesn't exist.

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

dirs_to_create = [
    "training/calib", "training/image_2", "training/label_2", "training/label_pseudo",
    "testing/calib", "testing/image_2", "testing/label_2", "testing/label_pseudo",
    "ImageSets"
]

for d in dirs_to_create:
    p = os.path.join(output_folder, d)
    if not os.path.exists(p):
        os.makedirs(p)

# ---------------------------------------------------------
# PROCESS TRAINING DATA (KITTI + K360 Train + K360 Val)
# ---------------------------------------------------------

cur_img_index = 0
train_indices = []
val_indices = []

# 1. KITTI Training
print("Processing KITTI Training data...")
if os.path.exists(os.path.join(kitti_root, "training", "image_2")):
    kitti_train_imgs = sorted(glob.glob(os.path.join(kitti_root, "training", "image_2", "*.png")))
    for image_path in kitti_train_imgs:
        img_name = os.path.basename(image_path)
        img_id = img_name.split('.')[0]
        
        src_calib = os.path.join(kitti_root, "training", "calib", img_id + ".txt")
        src_label = os.path.join(kitti_root, "training", "label_2", img_id + ".txt")
        
        if not os.path.exists(src_calib) or not os.path.exists(src_label):
            continue
            
        dst_img = os.path.join(output_folder, "training", "image_2", str(cur_img_index).zfill(6) + ".png")
        dst_calib = os.path.join(output_folder, "training", "calib", str(cur_img_index).zfill(6) + ".txt")
        dst_label = os.path.join(output_folder, "training", "label_2", str(cur_img_index).zfill(6) + ".txt")
        dst_label_pseudo = os.path.join(output_folder, "training", "label_pseudo", str(cur_img_index).zfill(6) + ".txt")
        
        shutil.copy(image_path, dst_img)
        shutil.copy(src_calib, dst_calib)
        shutil.copy(src_label, dst_label)
        shutil.copy(src_label, dst_label_pseudo) # Copy GT to pseudo for KITTI
        
        train_indices.append(cur_img_index)
        cur_img_index += 1
else:
    print(f"KITTI training folder not found at {os.path.join(kitti_root, 'training', 'image_2')}")

print(f"Processed {len(train_indices)} KITTI training images.")

# 2. K360 Training
print("Processing K360 Training data...")
for folder in sorted(os.listdir(k360_root)):
    if folder in training_sequences:
        cur_folder = os.path.join(k360_root, folder)
        for image in sorted(glob.glob(os.path.join(cur_folder, "image_00/data_rect/", "*.png"))):
            img_number = os.path.basename(image).split(".")[0]
            
            src_calib = os.path.join(cur_folder, "calib", str(img_number) + ".txt")
            src_label = os.path.join(cur_folder, "label_00", str(img_number) + ".txt")
            src_pseudo_label = os.path.join(k360_pseudo_label_path, str(folder) + '_' + str(img_number) + ".txt")
            
            if not os.path.exists(src_calib) or not os.path.exists(src_label):
                continue
            
            dst_img = os.path.join(output_folder, "training", "image_2", str(cur_img_index).zfill(6) + ".png")
            dst_calib = os.path.join(output_folder, "training", "calib", str(cur_img_index).zfill(6) + ".txt")
            dst_label_pseudo = os.path.join(output_folder, "training", "label_pseudo", str(cur_img_index).zfill(6) + ".txt")
            
            shutil.copy(image, dst_img)
            shutil.copy(src_calib, dst_calib)
            
            if os.path.exists(src_pseudo_label):
                shutil.copy(src_pseudo_label, dst_label_pseudo)
            else:
                print(f"Pseudo label not found for {folder} {img_number}")
                with open(dst_label_pseudo, 'w') as f: f.write("")
            
            train_indices.append(cur_img_index)
            cur_img_index += 1

# 3. K360 Validation (Appended to Training set but tracked in val.txt)
print("Processing K360 Validation data...")
for folder in sorted(os.listdir(k360_root)):
    if folder in validation_sequences:
        cur_folder = os.path.join(k360_root, folder)
        for image in sorted(glob.glob(os.path.join(cur_folder, "image_00/data_rect/", "*.png"))):
            img_number = os.path.basename(image).split(".")[0]
            
            src_calib = os.path.join(cur_folder, "calib", str(img_number) + ".txt")
            src_label = os.path.join(cur_folder, "label_00", str(img_number) + ".txt")
            
            if not os.path.exists(src_calib) or not os.path.exists(src_label):
                continue
            
            dst_img = os.path.join(output_folder, "training", "image_2", str(cur_img_index).zfill(6) + ".png")
            dst_calib = os.path.join(output_folder, "training", "calib", str(cur_img_index).zfill(6) + ".txt")
            dst_label_pseudo = os.path.join(output_folder, "training", "label_pseudo", str(cur_img_index).zfill(6) + ".txt")
            
            shutil.copy(image, dst_img)
            shutil.copy(src_calib, dst_calib)
            shutil.copy(src_label, dst_label_pseudo) # Copy GT to pseudo for Val
            
            val_indices.append(cur_img_index)
            cur_img_index += 1

# Write ImageSets
with open(os.path.join(output_folder, "ImageSets", "train.txt"), 'w') as f:
    for i in train_indices:
        f.write(str(i).zfill(6) + "\n")

with open(os.path.join(output_folder, "ImageSets", "val.txt"), 'w') as f:
    for i in val_indices:
        f.write(str(i).zfill(6) + "\n")

# ---------------------------------------------------------
# PROCESS TESTING DATA (KITTI Test + K360 Test)
# ---------------------------------------------------------
cur_test_index = 0
test_indices = []

# 4. KITTI Testing
print("Processing KITTI Testing data...")
if os.path.exists(os.path.join(kitti_root, "testing", "image_2")):
    kitti_test_imgs = sorted(glob.glob(os.path.join(kitti_root, "testing", "image_2", "*.png")))
    for image_path in kitti_test_imgs:
        img_name = os.path.basename(image_path)
        img_id = img_name.split('.')[0]
        
        src_calib = os.path.join(kitti_root, "testing", "calib", img_id + ".txt")
        # No labels for KITTI test
        
        if not os.path.exists(src_calib):
            continue
            
        dst_img = os.path.join(output_folder, "testing", "image_2", str(cur_test_index).zfill(6) + ".png")
        dst_calib = os.path.join(output_folder, "testing", "calib", str(cur_test_index).zfill(6) + ".txt")
        
        shutil.copy(image_path, dst_img)
        shutil.copy(src_calib, dst_calib)
        
        test_indices.append(cur_test_index)
        cur_test_index += 1
else:
    print(f"KITTI testing folder not found at {os.path.join(kitti_root, 'testing', 'image_2')}")

# 5. K360 Testing
print("Processing K360 Testing data...")
for folder in sorted(os.listdir(k360_root)):
    if folder in testing_sequences:
        cur_folder = os.path.join(k360_root, folder)
        for image in sorted(glob.glob(os.path.join(cur_folder, "image_00/data_rect/", "*.png"))):
            img_number = os.path.basename(image).split(".")[0]
            
            src_calib = os.path.join(cur_folder, "calib", str(img_number) + ".txt")
            src_label = os.path.join(cur_folder, "label_00", str(img_number) + ".txt")
            
            if not os.path.exists(src_calib) or not os.path.exists(src_label):
                continue
            
            dst_img = os.path.join(output_folder, "testing", "image_2", str(cur_test_index).zfill(6) + ".png")
            dst_calib = os.path.join(output_folder, "testing", "calib", str(cur_test_index).zfill(6) + ".txt")
            dst_label = os.path.join(output_folder, "testing", "label_2", str(cur_test_index).zfill(6) + ".txt")
            
            shutil.copy(image, dst_img)
            shutil.copy(src_calib, dst_calib)
            shutil.copy(src_label, dst_label)
            
            test_indices.append(cur_test_index)
            cur_test_index += 1

with open(os.path.join(output_folder, "ImageSets", "test.txt"), 'w') as f:
    for i in test_indices:
        f.write(str(i).zfill(6) + "\n")

print("Done.")
