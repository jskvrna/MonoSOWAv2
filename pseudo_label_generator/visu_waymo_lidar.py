import os
import glob
import random
import numpy as np
import open3d as o3d

waymo_lidar_path = "/path/to/waymo_to_kitti/training/"
waymo_pseudolidar_path = "/path/to/output/frames_waymo_converted/lidar_raw/"

all_folders = sorted(os.listdir(waymo_lidar_path))
random.shuffle(all_folders)

for folder in all_folders:
    cur_path = os.path.join(waymo_lidar_path, folder)
    cur_pseudo_path = os.path.join(waymo_pseudolidar_path, folder)

    all_files = sorted(glob.glob(os.path.join(cur_path, "velodyne", "*.npz")))
    random_index = random.randint(0, len(all_files))
    all_files = [all_files[random_index]]

    for file in all_files:
        file_name = os.path.basename(file)
        pseudo_file = os.path.join(cur_pseudo_path, "pcds", file_name)

        calib = os.path.join(cur_path, "calib", file_name.replace(".npz", ".txt"))
        with open(calib, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "Tr_velo_to_cam" in line:
                    velo_to_cam = line.split(" ")[1:]
                    velo_to_cam = [float(x) for x in velo_to_cam]
                    velo_to_cam = np.array(velo_to_cam).reshape(3, 4)
                    velo_to_cam = np.vstack((velo_to_cam, [0, 0, 0, 1]))
                    break

        lidar = np.load(file)['arr_0'][:, :4]
        lidar[:, 3] = 1.
        lidar = lidar.T
        lidar = velo_to_cam @ lidar
        lidar = lidar[:3].T

        pseudo_lidar = np.load(pseudo_file)["array1"]

        lidar_pcloud = o3d.geometry.PointCloud()
        lidar_pcloud.points = o3d.utility.Vector3dVector(lidar)
        colors = np.zeros_like(lidar)
        colors[:, 0] = 1
        lidar_pcloud.colors = o3d.utility.Vector3dVector(colors)

        pseudo_pcloud = o3d.geometry.PointCloud()
        pseudo_pcloud.points = o3d.utility.Vector3dVector(pseudo_lidar)
        colors = np.ones_like(pseudo_lidar)
        pseudo_pcloud.colors = o3d.utility.Vector3dVector(colors)

        visu_things = [lidar_pcloud, pseudo_pcloud]

        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window()
        for k in range(len(visu_things)):
            visualizer.add_geometry(visu_things[k])
        # visualizer.get_render_option().point_size = 5  # Adjust the point size if necessary
        visualizer.get_render_option().background_color = np.asarray([0, 0, 0])  # Set background to black
        visualizer.get_view_control().set_front([0, -0.3, -0.5])
        visualizer.get_view_control().set_lookat([0, 0, 1])
        visualizer.get_view_control().set_zoom(0.05)
        visualizer.get_view_control().set_up([0, -1, 0])
        visualizer.get_view_control().camera_local_translate(5., 0., 8.)
        visualizer.run()
        visualizer.destroy_window()
