import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools
from mayavi import mlab

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def show_camera_image(camera_image, camera_labels, layout, cmap=None):
  """Show a camera image and the given camera labels."""

  ax = plt.subplot(*layout)

  # Draw the camera labels.
  for camera_labels in frame.camera_labels:
    # Ignore camera labels that do not correspond to this camera.
    if camera_labels.name != camera_image.name:
      continue

    # Iterate over the individual labels.
    for label in camera_labels.labels:
      # Draw the object bounding box.
      ax.add_patch(patches.Rectangle(
        xy=(label.box.center_x - 0.5 * label.box.length,
            label.box.center_y - 0.5 * label.box.width),
        width=label.box.length,
        height=label.box.width,
        linewidth=1,
        edgecolor='red',
        facecolor='none'))

  # Show the camera image.
  plt.imshow(tf.image.decode_jpeg(camera_image.image), cmap=cmap)
  plt.title(open_dataset.CameraName.Name.Name(camera_image.name))
  plt.grid(False)
  plt.axis('off')
  plt.show()

def draw_pcloud(point_cloud):
    """
    Visualize a point cloud using Mayavi.

    Parameters:
    - point_cloud: NumPy array of shape (n, 3) representing the point cloud.
    """
    x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
    mlab.points3d(x, y, z, mode="point")

    # Customize the visualization (optional)
    mlab.axes()
    mlab.xlabel("X")
    mlab.ylabel("Y")
    mlab.zlabel("Z")
    mlab.show()

FILENAME = '/path/to/waymo_extracted/segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord'
dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')

for data in dataset:
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))

    for index, image in enumerate(frame.images):
        show_camera_image(image, frame.camera_labels, [1, 1, 1])
        break

    calibration = [cc for cc in frame.context.camera_calibrations]
    break

    (range_images, camera_projections, _, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose)

    points_all = np.concatenate(points, axis=0)
    cp_points_all = np.concatenate(cp_points, axis=0)

    draw_pcloud(points_all)

    images = sorted(frame.images, key=lambda i: i.name)
    cp_points_all_concat = np.concatenate([cp_points_all, points_all], axis=-1)
    cp_points_all_concat_tensor = tf.constant(cp_points_all_concat)

    # The distance between lidar points and vehicle frame origin.
    points_all_tensor = tf.norm(points_all, axis=-1, keepdims=True)
    cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)

    mask = tf.equal(cp_points_all_tensor[..., 0], images[0].name)

    cp_points_all_tensor = tf.cast(tf.gather_nd(
        cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)
    points_all_tensor = tf.gather_nd(points_all_tensor, tf.where(mask))

    projected_points_all_from_raw_data = tf.concat(
        [cp_points_all_tensor[..., 1:3], points_all_tensor], axis=-1).numpy()

    break
