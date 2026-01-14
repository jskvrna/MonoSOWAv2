import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import time

mesh = o3d.io.read_triangle_mesh("../../data/fiat2.gltf")
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.)

T = np.eye(4)
T[:3, :3] = o3d.geometry.get_rotation_matrix_from_zxy((0, -np.pi/2, 0)) #X rotation
mesh.transform(T)
T = np.eye(4)
T[:3, :3] = o3d.geometry.get_rotation_matrix_from_zxy((0, 0, np.pi)) #Y rotation
mesh.transform(T)
T = np.eye(4)
T[:3, 3] = (0, 1, 5)
mesh.transform(T)
mesh.compute_vertex_normals()

height_kittiavg = 1.52608343
width_kittiavg = 1.62858987
length_kittiavg = 3.88395449

bbox = mesh.get_minimal_oriented_bounding_box()
print(bbox)

scale_height = height_kittiavg/bbox.extent[2]
scale_width = width_kittiavg/bbox.extent[1]
scale_kittiavg = length_kittiavg/bbox.extent[0]

optim_scale = np.mean([scale_height, scale_width, scale_kittiavg])

mesh.scale(optim_scale,bbox.center)

bbox = mesh.get_minimal_oriented_bounding_box()
print(bbox)

#o3d.visualization.draw_geometries([mesh, coord_frame, bbox])

mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

start = time.time_ns()
scene = o3d.t.geometry.RaycastingScene()
temp_id = scene.add_triangles(mesh_t)

rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=70,
        center=[0, 0, 1],
        eye=[0, 0, 0],
        up=[0, -1, 0],
        width_px=1382,
        height_px=512,
    )
#print(rays.shape)
ans = scene.cast_rays(rays)

#print(ans['t_hit'].shape)

mask = ans['t_hit'] < 1000
rays_dist = ans['t_hit'][mask]
#print(np.array(rays_dist, dtype=float).shape)
#print(np.array(rays_dist).mean())
rays = rays[mask]

#print(rays_dist.shape, rays.shape)

temp_x = rays[:,3] * rays_dist
temp_y = rays[:,4] * rays_dist
temp_z = rays[:,5] * rays_dist

pcloud = np.zeros((temp_x.shape[0], 3))

pcloud[:,0] = np.array(temp_x.numpy())
pcloud[:,1] = np.array(temp_y.numpy())
pcloud[:,2] = np.array(temp_z.numpy())

#print(pcloud.shape)

point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(pcloud)

point_cloud = point_cloud.voxel_down_sample(voxel_size=0.1)
point_cloud = point_cloud.remove_statistical_outlier(100, 0.5)[0]


#print(len(point_cloud.points))
#o3d.io.write_point_cloud("copy_of_fragment.pcd", point_cloud)
o3d.visualization.draw_geometries([point_cloud,mesh], zoom=0.5, lookat=[0, 0, 1], front=[0, -0.2, -0.5],
                                 up=[0, -1, 0])

#plt.imshow(ans['t_hit'].numpy())
#plt.show()
#plt.imshow(np.abs(ans['primitive_normals'].numpy()))
#plt.show()
#plt.imshow(np.abs(ans['geometry_ids'].numpy()), vmax=3)
#plt.show()

