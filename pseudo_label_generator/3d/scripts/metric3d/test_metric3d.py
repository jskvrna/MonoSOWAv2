import torch
import glob
import os
from tqdm import tqdm
import cv2
import numpy as np
import open3d as o3d

model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_giant2', pretrain=True)
model.eval().cuda()
intrinsic = [7.215377000000e+02, 7.215377000000e+02, 6.095593000000e+02, 1.728540000000e+02]

def get_pcd_base(H, W, u0, v0, fx, fy):
    x_row = np.arange(0, W)
    x = np.tile(x_row, (H, 1))
    x = x.astype(np.float32)
    u_m_u0 = x - u0

    y_col = np.arange(0, H)  # y_col = np.arange(0, height)
    y = np.tile(y_col, (W, 1)).T
    y = y.astype(np.float32)
    v_m_v0 = y - v0

    x = u_m_u0 / fx
    y = v_m_v0 / fy
    z = np.ones_like(x)
    pw = np.stack([x, y, z], axis=2)  # [h, w, c]
    return pw


def reconstruct_pcd(depth, fx, fy, u0, v0, pcd_base=None, mask=None):
    if type(depth) == torch.__name__:
        depth = depth.cpu().numpy().squeeze()
    depth = cv2.medianBlur(depth, 5)
    if pcd_base is None:
        H, W = depth.shape
        pcd_base = get_pcd_base(H, W, u0, v0, fx, fy)
    pcd = depth[:, :, None] * pcd_base
    if mask:
        pcd[mask] = 0
    return pcd

def load_calibration(calibration_file):
    with open(calibration_file, 'r') as f:
        lines = f.readlines()

    # Find the transformation matrix between LiDAR and camera2
    # We look for "Tr_velo_to_cam" and "P2" in the calibration file

    for line in lines:
        if 'Tr_velo_to_cam' in line:
            Tr_velo_to_cam = np.array([float(value) for value in line.split()[1:]]).reshape(3, 4)
            # The projection matrix is not needed for this transformation; we only need Tr_velo_to_cam
        if 'P2' in line:
            P2 = np.array([float(value) for value in line.split()[1:]]).reshape(3, 4)
            return P2
        if 'P_rect_02' in line:
            P2 = np.array([float(value) for value in line.split()[1:]]).reshape(3, 4)
            return P2
    return None

path_to_kitti = '/path/to/datasets/KITTI/complete_sequences/'
output = '/path/to/output/metric3d/'
#iterate through folders in path_to_kitti
for folder in os.listdir(path_to_kitti):
    if not os.path.isdir(os.path.join(path_to_kitti, folder)):
        continue
    print(folder)
    if not os.path.exists(os.path.join(output, folder)):
        os.makedirs(os.path.join(output, folder))

    # load calibration
    calibration_file_path = os.path.join(path_to_kitti, folder, 'calib_cam_to_cam.txt')
    P2 = load_calibration(calibration_file_path)

    for subfolder in os.listdir(os.path.join(path_to_kitti, folder)):
        if not os.path.isdir(os.path.join(path_to_kitti, folder, subfolder)):
            continue
        print(subfolder)
        tmp_folder_path = os.path.join(path_to_kitti, folder, subfolder)
        path_to_imgs = os.path.join(tmp_folder_path, 'image_02/data/')
        image_paths = sorted(glob.glob(os.path.join(path_to_imgs, '*.png')))
        index = 0

        output_path = os.path.join(output, folder, subfolder)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if not os.path.exists(os.path.join(output_path, 'pngs')):
            os.makedirs(os.path.join(output_path, 'pngs'))
        if not os.path.exists(os.path.join(output_path, 'pcds')):
            os.makedirs(os.path.join(output_path, 'pcds'))

        for image_path in tqdm(image_paths, desc="Processing Images"):
            rgb_origin = cv2.imread(image_path)[:, :, ::-1]

            intrinsic = [P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]]
            #### ajust input size to fit pretrained model
            # keep ratio resize
            input_size = (616, 1064)  # for vit model
            # input_size = (544, 1216) # for convnext model
            h, w = rgb_origin.shape[:2]
            scale = min(input_size[0] / h, input_size[1] / w)
            rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
            # remember to scale intrinsic, hold depth
            intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
            # padding to input_size
            padding = [123.675, 116.28, 103.53]
            h, w = rgb.shape[:2]
            pad_h = input_size[0] - h
            pad_w = input_size[1] - w
            pad_h_half = pad_h // 2
            pad_w_half = pad_w // 2
            rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT,value=padding)
            pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

            mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
            std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
            rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
            rgb = torch.div((rgb - mean), std)
            rgb = rgb[None, :, :, :].cuda()

            with torch.no_grad():
                pred_depth, confidence, output_dict = model.inference({'input': rgb})

            pred_depth = pred_depth.squeeze()
            pred_depth = pred_depth[pad_info[0]: pred_depth.shape[0] - pad_info[1],
                         pad_info[2]: pred_depth.shape[1] - pad_info[3]]

            # upsample to original size
            pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], rgb_origin.shape[:2],
                                                         mode='bilinear').squeeze()
            ###################### canonical camera space ######################

            #### de-canonical transform
            canonical_to_real_scale = intrinsic[0] / 1000.0  # 1000.0 is the focal length of canonical camera
            pred_depth = pred_depth * canonical_to_real_scale  # now the depth is metric
            pred_depth = torch.clamp(pred_depth, 0, 300).cpu()

            #pcd = reconstruct_pcd(pred_depth.numpy(), intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3])
            #pcd = pcd.reshape(h * w, 3)
            #pcd_to_save = o3d.geometry.PointCloud()
            #pcd_to_save.points = o3d.utility.Vector3dVector([pcd])
            #o3d.io.write_point_cloud('output/' + image_path.replace('.png', '2.pcd'), pcd_to_save)

            #save depth map
            pred_depth_np = pred_depth.cpu().numpy()
            pred_depth_normalized = cv2.normalize(pred_depth_np, None, 0, 255, cv2.NORM_MINMAX)
            pred_depth_uint8 = pred_depth_normalized.astype(np.uint8)
            cv2.imwrite(os.path.join(output_path, 'pngs/') + str(index).zfill(6) + '.png', pred_depth_uint8)

            out_height, out_width = pred_depth.shape
            intrinsic = [P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]]
            x, y = np.meshgrid(np.arange(out_width), np.arange(out_height))
            x = (x - intrinsic[2]) / intrinsic[0]
            y = (y - intrinsic[3]) / intrinsic[1]
            z = np.array(pred_depth)
            points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)

            #Write the points as open3d pcd file
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            o3d.io.write_point_cloud(os.path.join(output_path, 'pcds/') + str(index).zfill(6) + '.pcd', pcd)

            index+=1