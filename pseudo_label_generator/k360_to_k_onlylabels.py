import os
import glob
import shutil
import argparse

def main():
    # parser = argparse.ArgumentParser(description="Organize KITTI-360 labels and pseudo labels into KITTI format.")
    # parser.add_argument('--data_folder', type=str, default="/path/to/KITTI/",
    #                     help='Path to the KITTI-360 data folder (redacted).')
    # parser.add_argument('--pseudo_label_folder', type=str, required=True,
    #                     help='Path to the folder containing pseudo labels.')
    # parser.add_argument('--output_kitti_folder', type=str, default="/path/to/output/test_kitti",
    #                     help='Path to the output folder for KITTI formatted data (redacted).')
    # args = parser.parse_args()

    # data_folder = args.data_folder
    # pseudo_label_folder = args.pseudo_label_folder
    # output_kitti_folder = args.output_kitti_folder

    data_folder = "/path/to/KITTI/"
    output_kitti_folder = "/path/to/output/test_kitti"
    pseudo_label_folder = os.path.join(data_folder, "label_pseudo")

    training_sequences = ["2013_05_28_drive_0000_sync",
                          "2013_05_28_drive_0002_sync",
                          "2013_05_28_drive_0004_sync",
                          "2013_05_28_drive_0005_sync",
                          "2013_05_28_drive_0006_sync",
                          "2013_05_28_drive_0009_sync"]

    validation_sequences = ["2013_05_28_drive_0003_sync",
                            "2013_05_28_drive_0007_sync"]

    testing_sequences = ["2013_05_28_drive_0010_sync"]

    # Create base directories
    os.makedirs(output_kitti_folder, exist_ok=True)
    os.makedirs(os.path.join(output_kitti_folder, "training"), exist_ok=True)
    os.makedirs(os.path.join(output_kitti_folder, "testing"), exist_ok=True)
    os.makedirs(os.path.join(output_kitti_folder, "ImageSets"), exist_ok=True)

    # Create label directories
    os.makedirs(os.path.join(output_kitti_folder, "training", 'label_2'), exist_ok=True)
    os.makedirs(os.path.join(output_kitti_folder, "training", 'label_pseudo'), exist_ok=True)
    os.makedirs(os.path.join(output_kitti_folder, "testing", 'label_2'), exist_ok=True)
    os.makedirs(os.path.join(output_kitti_folder, "testing", 'label_pseudo'), exist_ok=True)

    cur_img_index = 0
    print("Processing training sequences...")
    for folder in sorted(os.listdir(data_folder)):
        if folder in training_sequences:
            cur_folder = os.path.join(data_folder, folder)
            # Iterate over labels directly to find corresponding frames
            for label_file in sorted(glob.glob(os.path.join(cur_folder, "label_00", "*.txt"))):
                img_number = os.path.basename(label_file).split(".")[0]
                cur_label = label_file
                cur_pseudo_label = os.path.join(pseudo_label_folder, f"{folder}_{img_number}.txt")

                shutil.copy(cur_label, os.path.join(output_kitti_folder, "training", "label_2", f"{cur_img_index:06d}.txt"))
                if os.path.exists(cur_pseudo_label):
                    shutil.copy(cur_pseudo_label, os.path.join(output_kitti_folder, "training", "label_pseudo", f"{cur_img_index:06d}.txt"))
                else:
                    print(f"Pseudo label not found for {folder} {img_number}")
                    # Create an empty pseudo label file
                    with open(os.path.join(output_kitti_folder, "training", "label_pseudo", f"{cur_img_index:06d}.txt"), 'w') as f:
                        pass
                cur_img_index += 1
                if cur_img_index % 1000 == 0:
                    print(f"Processed {cur_img_index} training labels...")

    num_of_training = cur_img_index
    print(f"Total training labels: {num_of_training}")
    with open(os.path.join(output_kitti_folder, "ImageSets", "train.txt"), 'w') as f:
        for i in range(cur_img_index):
            f.write(f"{i:06d}\n")

    print("Processing validation sequences...")
    for folder in sorted(os.listdir(data_folder)):
        if folder in validation_sequences:
            cur_folder = os.path.join(data_folder, folder)
            for label_file in sorted(glob.glob(os.path.join(cur_folder, "label_00", "*.txt"))):
                cur_label = label_file
                shutil.copy(cur_label, os.path.join(output_kitti_folder, "training", "label_2", f"{cur_img_index:06d}.txt"))
                # For validation, use ground truth as pseudo labels
                shutil.copy(cur_label, os.path.join(output_kitti_folder, "training", "label_pseudo", f"{cur_img_index:06d}.txt"))
                cur_img_index += 1
                if cur_img_index % 1000 == 0:
                    print(f"Processed {cur_img_index} (training+validation) labels...")

    num_of_validation = cur_img_index - num_of_training
    print(f"Total validation labels: {num_of_validation}")
    with open(os.path.join(output_kitti_folder, "ImageSets", "val.txt"), 'w') as f:
        for i in range(num_of_training, cur_img_index):
            f.write(f"{i:06d}\n")

    cur_img_index = 0
    print("Processing testing sequences...")
    for folder in sorted(os.listdir(data_folder)):
        if folder in testing_sequences:
            cur_folder = os.path.join(data_folder, folder)
            for label_file in sorted(glob.glob(os.path.join(cur_folder, "label_00", "*.txt"))):
                cur_label = label_file
                shutil.copy(cur_label, os.path.join(output_kitti_folder, "testing", "label_2", f"{cur_img_index:06d}.txt"))
                # For testing, use ground truth as pseudo labels
                shutil.copy(cur_label, os.path.join(output_kitti_folder, "testing", "label_pseudo", f"{cur_img_index:06d}.txt"))
                cur_img_index += 1
                if cur_img_index % 1000 == 0:
                    print(f"Processed {cur_img_index} testing labels...")

    print(f"Total testing labels: {cur_img_index}")
    with open(os.path.join(output_kitti_folder, "ImageSets", "test.txt"), 'w') as f:
        for i in range(cur_img_index):
            f.write(f"{i:06d}\n")

if __name__ == '__main__':
    main()
