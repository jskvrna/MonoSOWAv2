import argparse
import cv2
import glob
import numpy as np
import os
from PIL import Image
import torch
import time
import random
from tqdm import tqdm
import sys

from unidepth.models import UniDepthV2
from unidepth.utils.camera import Pinhole


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

    # Find the projection matrix
    # We look for "P2" and "P_rect_02" in the calibration file

    for line in lines:
        if 'P2' in line:
            P2 = np.array([float(value) for value in line.split()[1:]]).reshape(3, 4)
            return P2
        if 'P_rect_02' in line:
            P2 = np.array([float(value) for value in line.split()[1:]]).reshape(3, 4)
            return P2
    return None


if __name__ == "__main__":
    model = UniDepthV2.from_pretrained(f"lpiccinelli/unidepth-v2-vitl14")
    model.resolution_level = 9
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    

    path_to_inputs = "/path/to/datasets/KITTI/complete_sequences/"
    output_path_frames = "/path/to/frames/frames_uni/"

    for folder in shuffle_with_seed(os.listdir(path_to_inputs)):
        if not os.path.isdir(os.path.join(path_to_inputs, folder)):
            continue
        print("Current folder for pseudo_lidar: ", folder)
        output = os.path.join(output_path_frames, "lidar_raw/")
        if not os.path.exists(os.path.join(output, folder)):
            os.makedirs(os.path.join(output, folder))

        # load calibration
        calibration_file_path = os.path.join(path_to_inputs, folder, 'calib_cam_to_cam.txt')
        P2 = load_calibration(calibration_file_path)

        f_x, f_y = P2[0, 0], P2[1, 1]
        c_x, c_y = P2[0, 2], P2[1, 2]

        K = torch.tensor(P2[:3, :3])

        for subfolder in shuffle_with_seed(os.listdir(os.path.join(path_to_inputs, folder))):
            if not os.path.isdir(os.path.join(path_to_inputs, folder, subfolder)):
                continue
            print("Current subfolder for pseudo_lidar: ", subfolder)
            tmp_folder_path = os.path.join(path_to_inputs, folder, subfolder)
            path_to_imgs = os.path.join(tmp_folder_path, 'image_02/data/')
            image_paths = shuffle_with_seed(sorted(glob.glob(os.path.join(path_to_imgs, '*.png'))))

            output_path = os.path.join(output, folder, subfolder)

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            if not os.path.exists(os.path.join(output_path, 'pcds')):
                os.makedirs(os.path.join(output_path, 'pcds'))

            for image_path in tqdm(image_paths, desc="Processing Images"):
                # Get the original file name without extension
                file_name = os.path.basename(image_path).split('.')[0]
                if os.path.exists(os.path.join(output_path, 'pcds', file_name + '.npz')):
                    continue
                img = Image.open(image_path)
                width, height = img.size
                camera = Pinhole(K=K)
                
                rgb = torch.from_numpy(np.array(img)).permute(2, 0, 1) # C, H, W
 
                predictions = model.infer(rgb, camera)

                depth = predictions["depth"].cpu()

                x, y = np.meshgrid(np.arange(width), np.arange(height))
                x = (x - c_x) / f_x
                y = (y - c_y) / f_x
                z = np.array(depth)
                pseudo_lidar = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
                
                np.savez_compressed(os.path.join(output_path, 'pcds', file_name + '.npz'), array1=pseudo_lidar)
