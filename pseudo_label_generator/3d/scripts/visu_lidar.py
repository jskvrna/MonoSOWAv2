import numpy as np
import zstd
import dill
import open3d
# Load LiDAR data from npz file
lidar_data = np.load('/path/to/output/frames_all_noicp_newm/depth/000032.npz')  # (redacted)

point_cloud = open3d.geometry.PointCloud()
point_cloud.points = open3d.utility.Vector3dVector(lidar_data['array1'])
color = np.ones((point_cloud.points.__len__(), 3))
point_cloud.colors = open3d.utility.Vector3dVector(color)
point_cloud = point_cloud.voxel_down_sample(voxel_size=0.05)

visu_things = [point_cloud]
#open3d.visualization.draw_geometries(visu_things, zoom=0.1, lookat=[0, 0, 1], front=[0, -0.2, -0.5],
#                                     up=[0, -1, 0])
visualizer = open3d.visualization.Visualizer()
visualizer.create_window()
for k in range(len(visu_things)):
    visualizer.add_geometry(visu_things[k])
#visualizer.get_render_option().point_size = 5  # Adjust the point size if necessary
visualizer.get_render_option().background_color = np.asarray([0, 0, 0])  # Set background to black
visualizer.get_view_control().set_front([0, -0.3, -0.5])
visualizer.get_view_control().set_lookat([0, 0, 1])
visualizer.get_view_control().set_zoom(0.05)
visualizer.get_view_control().set_up([0, -1, 0])
visualizer.get_view_control().camera_local_translate(5.,0.,8.)
visualizer.run()
visualizer.destroy_window()
