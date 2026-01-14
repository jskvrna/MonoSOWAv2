import os
import cv2
import random
import numpy as np

# Paths to your label and image directories
#label_dir = "/path/to/waymo_kitti_monodetr/training/label_2"  # (redacted)
label_dir = "/path/to/monodetr_outputs/waymo_pseudo3_newlabels/outputs/data/"  # (redacted)
#label_dir = "/path/to/monodetr_outputs/waymo_orig3_newlabels/outputs/data/"  # (redacted)
#label_dir = "/path/to/MonoDETR/data/waymo_orig/training/label_2"  # (redacted)
#label_dir = "/path/to/monodetr_outputs/waymo_orig_wrong/outputs/data"  # (redacted)
image_dir = "/path/to/MonoDETR/data/waymo_orig/training/image_2"  # (redacted)
calib_dir = "/path/to/MonoDETR/data/waymo_orig/training/calib"  # (redacted)
seed = 42

visu_2D = True
visu_3D = True

def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line == '':
                continue
            key, value = line.split(':', 1)
            data[key] = np.array([float(x) for x in value.strip().split()])
        # Reshape matrices
        if 'P2' in data:
            data['P2'] = data['P2'].reshape(3, 4)
        if 'R0_rect' in data:
            data['R0_rect'] = data['R0_rect'].reshape(3, 3)
        if 'Tr_velo_to_cam' in data:
            data['Tr_velo_to_cam'] = data['Tr_velo_to_cam'].reshape(3, 4)
        elif 'Tr_velo_to_cam0' in data:
            data['Tr_velo_to_cam'] = data['Tr_velo_to_cam0'].reshape(3, 4)
        return data

def create_3d_bbox(x, y, z, h, w, l, ry):
    # Create the 8 corners of the bounding box in object coordinate system
    z_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    x_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

    # Rotate the corners
    corners = np.array([x_corners, y_corners, z_corners])
    # Rotation matrix around y-axis
    R = np.array([[np.cos(ry), 0, np.sin(ry)],
                  [0, 1, 0],
                  [-np.sin(ry), 0, np.cos(ry)]])
    corners_3d = R @ corners

    # Translate the corners to the location
    corners_3d += np.array([[x], [y], [z]])

    # Return the corners
    return corners_3d.T  # 8x3

def project_to_image(points_3d, P):
    points_3d_hom = np.vstack((points_3d, np.ones((1, points_3d.shape[1]))))
    points_2d = np.dot(P, points_3d_hom)
    points_2d /= points_2d[2]
    return points_2d[:2]

def draw_bounding_box(image, corners_2d, color):
    corners_2d = corners_2d.astype(np.int32).T

    # Define the faces of the 3D bounding box
    faces = [
        [0, 1, 2, 3],  # Bottom face
        [4, 5, 6, 7],  # Top face
        [0, 1, 5, 4],  # Side faces
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7]
    ]

    # Create an overlay for transparent coloring
    overlay = image.copy()

    # Draw the faces with transparent fill
    for face in faces:
        pts = corners_2d[face]
        cv2.fillPoly(overlay, [pts], color=color)

    # Add transparency (alpha blending)
    alpha = 0.4
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Draw the edges of the bounding box
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face edges
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face edges
        (0, 4), (1, 5), (2, 6), (3, 7)   # Side edges
    ]
    for edge in edges:
        pt1 = tuple(corners_2d[edge[0]])
        pt2 = tuple(corners_2d[edge[1]])
        cv2.line(image, pt1, pt2, color=(255, 255, 255), thickness=1)
    return image

# Iterate over all label files
all_labels = os.listdir(label_dir)
all_labels.sort()
random.seed(seed)
shuffled_array = random.shuffle(all_labels)

for label_filename in all_labels:
    if not label_filename.endswith(".txt"):
        continue

    print(f"Processing {label_filename}...")

    # Construct the corresponding image filename
    base_name = os.path.splitext(label_filename)[0]
    # Try common image extensions
    possible_extensions = [".png", ".jpg", ".jpeg"]
    image_path = None
    for ext in possible_extensions:
        candidate = os.path.join(image_dir, base_name + ext)
        if os.path.exists(candidate):
            image_path = candidate
            break

    if image_path is None:
        print(f"No corresponding image found for {label_filename}. Skipping.")
        continue

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image at {image_path}.")
        continue

    # Prase the calibration file
    calib_filename = label_filename.replace("txt", "txt")
    calib_path = os.path.join(calib_dir, calib_filename)
    calib = read_calib_file(calib_path)

    # Parse the label file
    with open(os.path.join(label_dir, label_filename), "r") as f:
        lines = f.readlines()

    # Draw bounding boxes and class labels + 3D coordinates
    for line in lines:
        print(line)
        # KITTI format:
        # type, truncated, occluded, alpha, bbox_left, bbox_top, bbox_right, bbox_bottom,
        # height, width, length, x, y, z, rotation_y
        parts = line.strip().split(" ")
        if len(parts) < 15:
            # Not a valid KITTI label line
            continue

        object_type = parts[0]
        bbox_left = float(parts[4])
        bbox_top = float(parts[5])
        bbox_right = float(parts[6])
        bbox_bottom = float(parts[7])

        # Extract predicted 3D coordinates (x, y, z)
        x_3d = float(parts[11])
        y_3d = float(parts[12])
        z_3d = float(parts[13])

        if visu_2D:
            # Draw a rectangle around the object in green
            cv2.rectangle(image, (int(bbox_left), int(bbox_top)), (int(bbox_right), int(bbox_bottom)),
                          (0, 255, 0), 2)

            # Prepare the text to be displayed: object type and 3D coordinates
            label_text = f"{object_type} (X:{x_3d:.2f}, Y:{y_3d:.2f}, Z:{z_3d:.2f})"

            # Put the text above the bounding box
            cv2.putText(image, label_text, (int(bbox_left), int(bbox_top)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if visu_3D:
            h = float(parts[8])
            w = float(parts[9])
            l = float(parts[10])
            ry = float(parts[14])

            bbox_3d = create_3d_bbox(x_3d, y_3d, z_3d, h, w, l, ry)
            bbox_2d = project_to_image(bbox_3d.T, calib['P2'])

            # Draw the 3D bounding box
            image = draw_bounding_box(image, bbox_2d, color=(0, 255, 0))


    # Display the image
    cv2.imshow("Labeled Image", image)
    key = cv2.waitKey(0)

cv2.destroyAllWindows()

