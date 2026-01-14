import glob
import numpy as np
import os
from PIL import Image
import torch
import time
import random
from tqdm import tqdm
import cv2

from depth_anything_v2.dpt import DepthAnythingV2

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
    focal_kitti = 725.0087 
    encoder = 'vitl' # or 'vits', 'vitb'
    dataset = 'vkitti' # 'hypersim' for indoor model, 'vkitti' for outdoor model
    max_depth = 80 # 20 for indoor model, 80 for outdoor model
    checkpoint_path = "checkpoints/depth_anything_v2_metric_vkitti_vitl.pth"

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model = model.to(DEVICE).eval()

    path_to_inputs = "/path/to/KITTI/" # IMPORTANT: Update this path
    output_path_frames = "/path/to/output/k360_dany/"

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

        path_to_imgs = os.path.join(sequence_path, 'image_00', 'data_rect')
        if not os.path.isdir(path_to_imgs):
            print(f"Skipping sequence '{folder}' because image directory was not found.")
            continue

        # Using the average focal length from the K360 calibration
        focal_k360 = (f_x + f_y) / 2.0
        depth_scale_factor = focal_k360 / focal_kitti
            
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
            
            image = cv2.imread(image_path)
            height, width, _ = image.shape
            
            depth_numpy = model.infer_image(image)
            depth_numpy = np.array(Image.fromarray(depth_numpy).resize((width, height), Image.NEAREST))
            # Scale the depth prediction
            depth_numpy_scaled = depth_numpy * depth_scale_factor

            x, y = np.meshgrid(np.arange(width), np.arange(height))
            x = (x - c_x) / f_x
            y = (y - c_y) / f_y
            z = np.array(depth_numpy_scaled)
            pseudo_lidar = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
            
            np.savez_compressed(output_file_path, array1=pseudo_lidar)
