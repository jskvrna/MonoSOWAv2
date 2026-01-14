import glob
import numpy as np
import os
from PIL import Image
import torch
import time
import random
from tqdm import tqdm

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config, change_dataset

def shuffle_with_seed(folders):
    # Generate a unique seed using the current time and process ID
    unique_seed = int(time.time() * 1000) + os.getpid()
    #unique_seed = 666
    random.seed(unique_seed)
    random.shuffle(folders)
    print(f"Using seed: {unique_seed}")
    return folders

def load_calibration(calibration_file):
    with open(calibration_file, 'r') as f:
        lines = f.readlines()

    # Find the projection matrix for the rectified camera 00
    # We look for "P_rect_00" in the calibration file
    for line in lines:
        if 'P_rect_00' in line:
            # The line looks like: 'P_rect_00: 552.554261 0.000000 ...'
            values = np.array([float(value) for value in line.split(':')[1].strip().split()])
            P = values.reshape(3, 4)
            return P
    return None


if __name__ == "__main__":
    conf = get_config("zoedepth", "eval", config_version="kitti", config_mode="eval")
    conf = change_dataset(conf, new_dataset="kitti")
    model_zoe_k = build_model(conf)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    zoe = model_zoe_k.to(DEVICE)

    path_to_inputs = "/path/to/KITTI/" # IMPORTANT: Update this path
    output_path_frames = "/path/to/output/k360_zoe/"

    # KITTI's typical focal length. The model was trained on KITTI.
    # This is used to scale the depth prediction for the K360 dataset.
    # KITTI images are 1242x375, a common focal length is ~721.
    # The ZoeDepth model config for KITTI uses a focal length of 721.5377.
    focal_kitti = 721.5377 

    for folder in shuffle_with_seed(os.listdir(path_to_inputs)):
        sequence_path = os.path.join(path_to_inputs, folder)
        if not os.path.isdir(sequence_path):
            print(f"Skipping '{folder}' because it is not a directory.")
            continue
        print("Current sequence for pseudo_lidar: ", folder)
        
        output_base = os.path.join(output_path_frames, "lidar_raw/")
        output_sequence_path = os.path.join(output_base, folder)
        if not os.path.exists(output_sequence_path):
            os.makedirs(output_sequence_path)

        # load calibration
        calibration_file_path = os.path.join(path_to_inputs, 'calibration', 'perspective.txt')
        if not os.path.exists(calibration_file_path):
            print(f"Skipping sequence '{folder}' because calibration file was not found.")
            continue
        
        P = load_calibration(calibration_file_path)
        if P is None:
            print(f"Skipping sequence '{folder}' because calibration could not be loaded.")
            continue

        f_x, f_y = P[0, 0], P[1, 1]
        c_x, c_y = P[0, 2], P[1, 2]

        print(f"Using focal lengths: f_x={f_x}, f_y={f_y}, c_x={c_x}, c_y={c_y}")

        # Using the average focal length from the K360 calibration
        focal_k360 = (f_x + f_y) / 2.0
        depth_scale_factor = focal_k360 / focal_kitti

        path_to_imgs = os.path.join(sequence_path, 'image_00', 'data_rect')
        if not os.path.isdir(path_to_imgs):
            print(f"Skipping sequence '{folder}' because image directory was not found.")
            continue
            
        image_paths = shuffle_with_seed(sorted(glob.glob(os.path.join(path_to_imgs, '*.png'))))

        output_path_pcds = os.path.join(output_sequence_path, 'pcds')
        if not os.path.exists(output_path_pcds):
            os.makedirs(output_path_pcds)

        for image_path in tqdm(image_paths, desc=f"Processing Images in {folder}"):
            # Get the original file name without extension
            file_name = os.path.basename(image_path).split('.')[0]
            output_file_path = os.path.join(output_path_pcds, file_name + '.npz')
            if os.path.exists(output_file_path):
                # This can be verbose if many files already exist, but it fulfills the request.
                # You can comment out the next line if you don't want to see a message for every skipped file.
                # print(f"Skipping frame '{file_name}' in sequence '{folder}' because output already exists.")
                continue
            color_image = Image.open(image_path).convert('RGB')
            width, height = color_image.size
            depth_numpy = zoe.infer_pil(color_image, pad_input=False)

            # Scale the depth prediction
            depth_numpy_scaled = depth_numpy * depth_scale_factor

            x, y = np.meshgrid(np.arange(width), np.arange(height))
            x = (x - c_x) / focal_k360
            y = (y - c_y) / focal_k360
            z = np.array(depth_numpy_scaled)
            pseudo_lidar = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
            
            np.savez_compressed(output_file_path, array1=pseudo_lidar)
