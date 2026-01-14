import argparse
import cv2
import glob
import numpy as np
import open3d as o3d
import os
from PIL import Image
import torch
import time
import random
from tqdm import tqdm
import sys
from SQLdepth import MonodepthOptions, SQLdepth
from torchvision import transforms, datasets
import networks

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

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)

def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

if __name__ == "__main__":
    options = MonodepthOptions()
    options.parser.convert_arg_line_to_args = convert_arg_line_to_args
    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        opt = options.parser.parse_args([arg_filename_with_prefix])
    else:
        opt = options.parser.parse_args()

    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)

    if opt.backbone in ["resnet", "resnet_lite"]:
            encoder = networks.ResnetEncoderDecoder(num_layers=opt.num_layers, num_features=opt.num_features, model_dim=opt.model_dim)
    elif opt.backbone == "resnet18_lite":
        encoder = networks.LiteResnetEncoderDecoder(model_dim=opt.model_dim)
    elif opt.backbone == "eff_b5":
        encoder = networks.BaseEncoder.build(num_features=opt.num_features, model_dim=opt.model_dim)
    else: 
        encoder = networks.Unet(pretrained=(not opt.load_pretrained_model), backbone=opt.backbone, in_channels=3, num_classes=opt.model_dim, decoder_channels=opt.dec_channels)

    if opt.backbone.endswith("_lite"):
        depth_decoder = networks.Lite_Depth_Decoder_QueryTr(in_channels=opt.model_dim, patch_size=opt.patch_size, dim_out=opt.dim_out, embedding_dim=opt.model_dim, 
                                                    query_nums=opt.query_nums, num_heads=4, min_val=opt.min_depth, max_val=opt.max_depth)
    else:
        depth_decoder = networks.Depth_Decoder_QueryTr(in_channels=opt.model_dim, patch_size=opt.patch_size, dim_out=opt.dim_out, embedding_dim=opt.model_dim, 
                                                query_nums=opt.query_nums, num_heads=4, min_val=opt.min_depth, max_val=opt.max_depth)
    
    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda()
    encoder = torch.nn.DataParallel(encoder)
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder = torch.nn.DataParallel(depth_decoder)
    depth_decoder.eval()

    path_to_inputs = "/path/to/datasets/KITTI/complete_sequences/"
    output_path_frames = "/path/to/frames/frames_spi2/"

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

                with torch.no_grad():
                    input_image = Image.open(image_path).convert('RGB')
                    original_width, original_height = input_image.size
                    input_image = input_image.resize((encoder_dict['width'], encoder_dict['height']), Image.LANCZOS)
                    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
                    
                    input_image = input_image.to('cuda')
                    input_color = torch.cat((input_image, torch.flip(input_image, [3])), 0)

                    outputs = depth_decoder(encoder(input_color))

                    pred_disp = outputs[("disp", 0)]
                    # pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                    pred_disp = pred_disp.cpu()[:, 0].numpy()

                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                    disp = cv2.resize(pred_disp[0], (original_width, original_height), interpolation=cv2.INTER_LINEAR)
                    #print(disp.shape, disp_resized.shape)
                    #disp_resized = 1 / disp_resized

                depth_numpy = disp
                x, y = np.meshgrid(np.arange(original_width), np.arange(original_height))
                x = (x - c_x) / f_x
                y = (y - c_y) / f_y
                z = np.array(depth_numpy)
                pseudo_lidar = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
                
                np.savez_compressed(os.path.join(output_path, 'pcds', file_name + '.npz'), array1=pseudo_lidar)
