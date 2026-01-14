import open3d as o3d
import numpy as np

degrees = 0
nclouds = 360
with_windows = False

for angle_rad in np.linspace(0., 2*np.pi - (2*np.pi)/nclouds , nclouds): #Per 10 degrees
    print(np.rad2deg(angle_rad))
    if with_windows:
        mesh = o3d.io.read_triangle_mesh("../../data/fiat2.gltf") #Read mesh of fiat uno converted via blender
    else:
        mesh = o3d.io.read_triangle_mesh("../../data/fiat_nowindows.gltf")  # Read mesh of fiat uno converted via blender
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.) #Only for visu purpose

    # Avg height, width and length to be used to correspond with the KITTI dataset.
    height_kittiavg = 1.52608343
    width_kittiavg = 1.62858987
    length_kittiavg = 3.88395449

    # Get the minimal bbox to get the dimension
    bbox = mesh.get_minimal_oriented_bounding_box()
    print(bbox)

    T = np.eye(4)
    T[:3, 3] = (-bbox.center[0], -bbox.center[1] , -bbox.center[2])  # Move it a little bit in front of the camera. The Y or the height was chosen empirically to not get many points from the roof, since in real life, we usually lack those points
    mesh.transform(T)

    # Compute the scales in each directions to be used in the point cloud.
    scale_height = height_kittiavg / bbox.extent[2]
    scale_width = width_kittiavg / bbox.extent[1]
    scale_lenght = length_kittiavg / bbox.extent[0]

    # Get mesh vertices as a numpy array
    vertices = np.asarray(mesh.vertices)

    # Scale vertices in each dimension separately, works, eventhough it looks wrong
    vertices[:, 2] *= scale_height
    vertices[:, 1] *= scale_lenght
    vertices[:, 0] *= scale_width

    # Update mesh vertices with scaled vertices
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    T = np.eye(4)
    T[:3, :3] = o3d.geometry.get_rotation_matrix_from_zxy((0, -np.pi/2, 0)) #X rotation to make the car as in the dataset and pointing forward as the reference car
    if with_windows:
        mesh.transform(T)

    T = np.eye(4)
    T[:3, :3] = o3d.geometry.get_rotation_matrix_from_zxy((0, 0, np.pi + angle_rad)) #Y rotation -||-
    mesh.transform(T)

    T = np.eye(4)
    T[:3, 3] = (0, 0.5, 5) #Move it a little bit in front of the camera. The Y or the height was chosen empirically to not get many points from the roof, since in real life, we usually lack those points
    mesh.transform(T)

    mesh.compute_vertex_normals() #Looks better in visu with normals

    # Get the minimal bbox to get the dimension to check, that the scaling is correct
    bbox = mesh.get_minimal_oriented_bounding_box()
    bbox.color = np.array([1,0,0])
    print(bbox)

    #o3d.visualization.draw_geometries([mesh, coord_frame, bbox])

    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    #Creates the raycasting scene and adds the template car mesh
    scene = o3d.t.geometry.RaycastingScene()
    temp_id = scene.add_triangles(mesh_t)

    #Specify the camera of the KITTI dataset
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            fov_deg=70,
            center=[0, 0, 1],
            eye=[0, 0, 0],
            up=[0, -1, 0],
            width_px=1382,
            height_px=512,
        )

    #Do the raycasting
    ans = scene.cast_rays(rays)

    #Filter points, that didnt hit anything
    mask = ans['t_hit'] < 1000
    rays_dist = ans['t_hit'][mask]
    rays = rays[mask]

    #Compute the intersection of the points, that hit the mesh
    temp_x = rays[:,3] * rays_dist
    temp_y = rays[:,4] * rays_dist
    temp_z = rays[:,5] * rays_dist

    pcloud = np.zeros((temp_x.shape[0], 3))

    pcloud[:,0] = np.array(temp_x.numpy())
    pcloud[:,1] = np.array(temp_y.numpy())
    pcloud[:,2] = np.array(temp_z.numpy())

    #Create pointcloud from the intersections
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pcloud)

    #There is many many points, thus we want to down sample and this works the best
    point_cloud = point_cloud.voxel_down_sample(voxel_size=0.025)
    #Because the mesh is not completely "waterproof" especially around wheels, this method removes points, that are seen but would make mess in the optimization
    #point_cloud = point_cloud.remove_statistical_outlier(100, 0.5)[0]
    #Transpose to the center of the coordinate system
    point_cloud = point_cloud.translate([0, -0.5, -5])
    #saves the pointcloud
    if with_windows:
        file_name = "../data/pcloud_filtered/" + str(int(degrees)) + ".pcd"
    else:
        file_name = "../data/pcloud_filtered_nw/" + str(int(degrees)) + ".pcd"
    o3d.io.write_point_cloud(file_name, point_cloud)

    T = np.eye(4)
    T[:3, 3] = (0, -0.5, -5)  # Move it a little bit in front of the camera. The Y or the height was chosen empirically to not get many points from the roof, since in real life, we usually lack those points
    mesh.transform(T)
    #o3d.visualization.draw_geometries([point_cloud, mesh, coord_frame], zoom=0.5, lookat=[0, 0, 1], front=[0, -0.2, -0.5], up=[0, -1, 0])

    degrees += 360/nclouds


#Now we want to just generate simple pcloud without any raycasting

if with_windows:
    mesh = o3d.io.read_triangle_mesh("../../data/fiat2.gltf")  # Read mesh of fiat uno converted via blender
else:
    mesh = o3d.io.read_triangle_mesh("../../data/fiat_nowindows.gltf")  # Read mesh of fiat uno converted via blender
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.) #Only for visu purpose

# Avg height, width and length to be used to correspond with the KITTI dataset.
height_kittiavg = 1.52608343
width_kittiavg = 1.62858987
length_kittiavg = 3.88395449

# Get the minimal bbox to get the dimension
bbox = mesh.get_minimal_oriented_bounding_box()
print(bbox)

T = np.eye(4)
T[:3, 3] = (-bbox.center[0], -bbox.center[1] , -bbox.center[2])  # Move it a little bit in front of the camera. The Y or the height was chosen empirically to not get many points from the roof, since in real life, we usually lack those points
mesh.transform(T)

# Compute the scales in each directions to be used in the point cloud.
scale_height = height_kittiavg / bbox.extent[2]
scale_width = width_kittiavg / bbox.extent[1]
scale_lenght = length_kittiavg / bbox.extent[0]

# Get mesh vertices as a numpy array
vertices = np.asarray(mesh.vertices)

# Scale vertices in each dimension separately, works, eventhough it looks wrong
vertices[:, 2] *= scale_height
vertices[:, 1] *= scale_lenght
vertices[:, 0] *= scale_width

# Update mesh vertices with scaled vertices
mesh.vertices = o3d.utility.Vector3dVector(vertices)

T = np.eye(4)
T[:3, :3] = o3d.geometry.get_rotation_matrix_from_zxy((0, -np.pi/2, 0)) #X rotation to make the car as in the dataset and pointing forward as the reference car
if with_windows:
    mesh.transform(T)

T = np.eye(4)
T[:3, :3] = o3d.geometry.get_rotation_matrix_from_zxy((0, 0, np.pi)) #Y rotation -||-
mesh.transform(T)

mesh.compute_vertex_normals() #Looks better in visu with normals

# Get the minimal bbox to get the dimension to check, that the scaling is correct
bbox = mesh.get_minimal_oriented_bounding_box()
bbox.color = np.array([1,0,0])
print(bbox)

#o3d.visualization.draw_geometries([mesh, coord_frame, bbox])

mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

pcd = mesh.sample_points_uniformly(number_of_points=10000)
#Because the mesh is not completely "waterproof" especially around wheels, this method removes points, that are seen but would make mess in the optimization
#point_cloud = pcd.remove_statistical_outlier(100, 0.5)[0]
point_cloud = pcd
#saves the pointcloud
if with_windows:
    file_name = "../data/pcloud_filtered/" + "999" + ".pcd"
else:
    file_name = "../data/pcloud_filtered_nw/" + "999" + ".pcd"
o3d.io.write_point_cloud(file_name, point_cloud)
o3d.visualization.draw_geometries([point_cloud, mesh, coord_frame], zoom=0.5, lookat=[0, 0, 1], front=[0, -0.2, -0.5], up=[0, -1, 0])


