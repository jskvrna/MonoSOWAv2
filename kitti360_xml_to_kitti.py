import os
import sys
import glob
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from collections import defaultdict

# User configuration
DATASET_ROOT = "/path/to/KITTI360/"
OUTPUT_DIR_NAME = "label_xml_converted"
TARGET_CLASSES = ["Car", "Pedestrian", "Cyclist"]

# KITTI-360 label names are typically lowercase in XML. Map them to KITTI label_2 types.
LABEL_TO_KITTI_TYPE = {
    "car": "Car",
    "pedestrian": "Pedestrian",
    "cyclist": "Cyclist",
    "bicycle": "Cyclist",
    "rider": "Cyclist",
}


def _parse_mat_from_line(prefix: str, line: str, rows: int, cols: int):
    if not line.startswith(prefix):
        return None
    vals = [float(x) for x in line.split()[1:]]
    if len(vals) != rows * cols:
        return None
    return np.array(vals, dtype=np.float32).reshape(rows, cols)


def load_perspective_calibration(calib_dir: str):
    """Load KITTI-360 perspective intrinsics/rectification from calibration/perspective.txt.

    Returns dict with keys like:
      - S_rect_00 -> (W, H)
      - R_rect_00 -> 3x3
      - P_rect_00 -> 3x4
    """
    path = os.path.join(calib_dir, 'perspective.txt')
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    calib = {}
    with open(path, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith('S_rect_'):
                key = line.split(':', 1)[0]
                vals = [float(x) for x in line.split()[1:]]
                if len(vals) == 2:
                    calib[key] = (int(vals[0]), int(vals[1]))
            elif line.startswith('R_rect_'):
                key = line.split(':', 1)[0]
                mat = _parse_mat_from_line(f'{key}:', line, 3, 3)
                if mat is not None:
                    calib[key] = mat
            elif line.startswith('P_rect_'):
                key = line.split(':', 1)[0]
                mat = _parse_mat_from_line(f'{key}:', line, 3, 4)
                if mat is not None:
                    calib[key] = mat

    return calib


def load_calib_cam_to_pose(calib_dir: str):
    """Load KITTI-360 calibration/calib_cam_to_pose.txt.

    Each line: image_00: r11 r12 r13 tx r21 ... tz
    Interpreted as T_pose_cam (pose <- cam).
    """
    path = os.path.join(calib_dir, 'calib_cam_to_pose.txt')
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    out = {}
    with open(path, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line or ':' not in line:
                continue
            name, rest = line.split(':', 1)
            name = name.strip()
            vals = [float(x) for x in rest.split()]
            if len(vals) != 12:
                continue
            T = np.eye(4, dtype=np.float32)
            T[:3, :] = np.array(vals, dtype=np.float32).reshape(3, 4)
            out[name] = T
    return out


def load_poses(poses_txt: str):
    """Load per-frame poses.txt.

    Expected format: frame_id r11 r12 r13 tx r21 ... tz
    Returns dict[int -> 4x4] of T_world_pose (pose/camera-rig pose in world).
    """
    poses = {}
    with open(poses_txt, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 13:
                continue
            frame = int(parts[0])
            vals = [float(x) for x in parts[1:]]
            T = np.eye(4, dtype=np.float32)
            T[:3, :] = np.array(vals, dtype=np.float32).reshape(3, 4)
            poses[frame] = T
    return poses


def normalize_angle(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def project_points(P_rect: np.ndarray, pts_rect: np.ndarray):
    """Project Nx3 rectified camera points with 3x4 P matrix -> Nx2 pixels + depth.

    Returns (uv Nx2, depth N)
    """
    ones = np.ones((pts_rect.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts_rect.astype(np.float32), ones], axis=1)
    proj = (P_rect @ pts_h.T).T
    depth = proj[:, 2].copy()
    depth[depth == 0] = 1e-6
    uv = proj[:, :2] / depth[:, None]
    return uv, depth


def _bbox_floor_ceil(x1: float, y1: float, x2: float, y2: float):
    """Match typical KITTI rounding: min->floor, max->ceil."""
    return (
        float(np.floor(x1)),
        float(np.floor(y1)),
        float(np.ceil(x2)),
        float(np.ceil(y2)),
    )


def _env_flag(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v not in ('0', 'false', 'False', 'no', 'No')


class Kitti360Object(object):
    def __init__(self):
        self.kitti_type = None
        self.start_frame = 0
        self.end_frame = 0
        self.timestamp = None  # int frame index when provided (dynamic objects)
        self.instance_id = None
        self.transform = np.eye(4, dtype=np.float32)  # 4x4
        self.vertices = None  # 8x3 in local coords

    def compute_dims_loc_yaw(self):
        """Return (h, w, l, x, y, z, ry). Calibration is ignored.

        - Uses transform translation as (x,y,z)
        - Uses scale norms of transform axes for (l,w,h)
        - Uses 0 for rotation_y (ry) because mapping to camera coords needs calibration.
        """
        t = self.transform
        # These are in world/map frame; proper camera coords are computed per-frame in main.
        x, y, z = float(t[0, 3]), float(t[1, 3]), float(t[2, 3])

        # Approx dims from column norms (handles embedded scale).
        sx = float(np.linalg.norm(t[:3, 0]))
        sy = float(np.linalg.norm(t[:3, 1]))
        sz = float(np.linalg.norm(t[:3, 2]))

        # Heuristic mapping to KITTI (h,w,l).
        # Many KITTI-360 bboxes are defined with local unit cube scaled by (sx,sy,sz).
        l = sx
        w = sy
        h = sz

        ry = 0.0
        return h, w, l, x, y, z, ry

class Tracklet(object):
    def __init__(self):
        self.objectType = None
        self.h = 0
        self.w = 0
        self.l = 0
        self.first_frame = -1
        self.poses = []  # list of (tx, ty, tz, rx, ry, rz)
        self.finished = False

    def __str__(self):
        return "[Tracklet] type={}, h={}, w={}, l={}, first_frame={}, poses={}".format(
            self.objectType, self.h, self.w, self.l, self.first_frame, len(self.poses))

def parseXML(xml_file):
    """Parse KITTI-360 3D bbox XML (opencv_storage/objectN).

    Returns a list of Kitti360Object instances.

    Note: These XMLs are not KITTI raw 'tracklets'; they typically contain:
      - <label>car/pedestrian/...</label>
      - <transform> (4x4)
      - <vertices> (8x3 unit cube)
      - <start_frame>, <end_frame>
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    if root.tag != 'opencv_storage':
        # Fallback: older style could exist, but your dataset is opencv_storage.
        print(f"Warning: Unexpected root tag '{root.tag}' in {xml_file}")

    objects = []
    for obj_elem in list(root):
        if not obj_elem.tag.startswith('object'):
            continue

        raw_label = (obj_elem.findtext('label') or '').strip().lower()
        kitti_type = LABEL_TO_KITTI_TYPE.get(raw_label)
        if not kitti_type:
            continue

        start_frame = int(obj_elem.findtext('start_frame') or 0)
        end_frame = int(obj_elem.findtext('end_frame') or start_frame)

        # KITTI-360 dynamic objects are often stored as multiple entries with a per-entry timestamp.
        # timestamp == -1 typically means "static" entry spanning [start_frame, end_frame].
        ts_text = obj_elem.findtext('timestamp')
        timestamp = None
        if ts_text is not None:
            try:
                ts_val = int(float(ts_text))
                if ts_val >= 0:
                    timestamp = ts_val
            except ValueError:
                timestamp = None

        inst_text = obj_elem.findtext('instanceId')
        instance_id = None
        if inst_text is not None:
            try:
                instance_id = int(inst_text)
            except ValueError:
                instance_id = None

        # Parse 4x4 transform
        T = np.eye(4, dtype=np.float32)
        tr = obj_elem.find('transform')
        if tr is not None:
            data_text = (tr.findtext('data') or '').strip()
            vals = [float(x) for x in data_text.split() if x]
            if len(vals) >= 16:
                T = np.array(vals[:16], dtype=np.float32).reshape(4, 4)

        # Parse vertices (8x3)
        verts = None
        ve = obj_elem.find('vertices')
        if ve is not None:
            data_text = (ve.findtext('data') or '').strip()
            vals = [float(x) for x in data_text.split() if x]
            if len(vals) >= 24:
                verts = np.array(vals[:24], dtype=np.float32).reshape(8, 3)

        obj = Kitti360Object()
        obj.kitti_type = kitti_type
        obj.start_frame = start_frame
        obj.end_frame = end_frame
        obj.timestamp = timestamp
        obj.instance_id = instance_id
        obj.transform = T
        obj.vertices = verts
        objects.append(obj)

    return objects

def read_kitti360_calibration(dataset_root, cam_idx='00'):
    """
    Reads KITTI-360 calibration files.
    Expects calibration/calib_cam_to_velo.txt and calibration/calib_intrinsic.txt
    Returns Tr_velo_to_cam, R0_rect, P_rect
    """
    calib_dir = os.path.join(dataset_root, 'calibration')
    
    # 1. Read Camera to Velo
    # calib_cam_to_velo.txt
    cam2velo_file = os.path.join(calib_dir, 'calib_cam_to_velo.txt')
    Tr_cam_to_velo = np.eye(4)
    
    if os.path.exists(cam2velo_file):
        with open(cam2velo_file, 'r') as f:
            lines = f.readlines()
            vals = []
            for line in lines:
                vals.extend([float(x) for x in line.split()])
            
            if len(vals) >= 12:
                Tr_cam_to_velo[:3, :] = np.array(vals[:12]).reshape(3, 4)
    else:
        print(f"Warning: {cam2velo_file} not found.")

    # Invert to get Velo to Cam
    # Tr_cam_to_velo is 4x4 (bottom row 0 0 0 1)
    Tr_velo_to_cam = np.linalg.inv(Tr_cam_to_velo)
    
    # 2. Read Intrinsics
    # calib_intrinsic.txt
    intrinsic_file = os.path.join(calib_dir, 'calib_intrinsic.txt')
    P_rect = np.eye(4)
    R_rect_00 = np.eye(4) 
    
    if os.path.exists(intrinsic_file):
        with open(intrinsic_file, 'r') as f:
            for line in f:
                if line.startswith(f'P_rect_{cam_idx}:'):
                    vals = [float(x) for x in line.split()[1:]]
                    P_rect[:3, :4] = np.array(vals).reshape(3, 4)
                elif line.startswith('R_rect_00:'): 
                    vals = [float(x) for x in line.split()[1:]]
                    R_rect_00[:3, :3] = np.array(vals).reshape(3, 3)
    else:
        print(f"Warning: {intrinsic_file} not found.")
                    
    return Tr_velo_to_cam, R_rect_00, P_rect

def read_calibration(calib_dir, cam_idx='00'):
    """
    Reads calibration files from calib_dir.
    Expects calib_velo_to_cam.txt and calib_cam_to_cam.txt
    Returns Tr_velo_to_cam, R0_rect, P_rect (for specified cam_idx)
    """
    # Initialize with identity in case files are missing (should handle error better)
    Tr_velo_to_cam = np.eye(4)
    R0_rect = np.eye(4)
    P_rect = np.eye(4)
    
    # Check if this is KITTI-360 style calibration (in parent 'calibration' folder)
    # We assume if we are calling this, we might be in a sequence folder.
    # But let's check if the passed calib_dir has the files.
    
    if os.path.exists(os.path.join(calib_dir, 'calib_cam_to_velo.txt')):
         return read_kitti360_calibration(os.path.dirname(calib_dir), cam_idx)
         
    # Also check if calib_dir IS the calibration folder
    if os.path.basename(calib_dir) == 'calibration' and os.path.exists(os.path.join(calib_dir, 'calib_cam_to_velo.txt')):
         return read_kitti360_calibration(os.path.dirname(calib_dir), cam_idx)

    # Read Tr_velo_to_cam
    v2c_file = os.path.join(calib_dir, 'calib_velo_to_cam.txt')
    if os.path.exists(v2c_file):
        with open(v2c_file, 'r') as f:
            for line in f:
                if line.startswith('R:'):
                    R = np.array([float(x) for x in line.split()[1:]]).reshape(3, 3)
                elif line.startswith('T:'):
                    T = np.array([float(x) for x in line.split()[1:]]).reshape(3, 1)
        
        # Tr_velo_to_cam = [R T; 0 0 0 1]
        Tr_velo_to_cam[:3, :3] = R
        Tr_velo_to_cam[:3, 3] = T.flatten()
    else:
        # Try KITTI-360 format if different? 
        # KITTI-360 usually has calib_cam_to_velo.txt
        pass

    # Read calib_cam_to_cam
    c2c_file = os.path.join(calib_dir, 'calib_cam_to_cam.txt')
    if os.path.exists(c2c_file):
        with open(c2c_file, 'r') as f:
            for line in f:
                if line.startswith('R_rect_00:'):
                    R_rect_00 = np.array([float(x) for x in line.split()[1:]]).reshape(3, 3)
                    R0_rect[:3, :3] = R_rect_00
                elif line.startswith(f'P_rect_{cam_idx}:'):
                    P_rect_val = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
                    P_rect = np.zeros((4, 4))
                    P_rect[:3, :4] = P_rect_val
                    P_rect[3, 3] = 1
                    
    return Tr_velo_to_cam, R0_rect, P_rect

def project_to_image(pts_3d, P):
    """
    Project 3D points to image plane.
    pts_3d: Nx4 (homogeneous)
    P: 3x4 projection matrix
    """
    pts_2d = np.dot(P, pts_3d.T).T
    
    # Avoid division by zero
    depth = pts_2d[:, 2]
    depth[depth == 0] = 1e-5
    
    pts_2d[:, 0] /= depth
    pts_2d[:, 1] /= depth
    return pts_2d[:, :2]

def compute_box_3d(tracklet, pose_idx):
    """
    Compute 3D bounding box corners in Velodyne coordinates.
    """
    h = tracklet.h
    w = tracklet.w
    l = tracklet.l
    tx, ty, tz, rx, ry, rz = tracklet.poses[pose_idx]
    
    # Rotation matrix around Z axis (in Velodyne coords)
    # KITTI tracklets rotation is around Z axis
    c = np.cos(rz)
    s = np.sin(rz)
    R = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])
    
    # 3D bounding box corners relative to center
    # KITTI format: l is x-size, w is y-size, h is z-size in Velo?
    # Wait, in Velo: x-forward, y-left, z-up.
    # Tracklet dims: h (height), w (width), l (length).
    # Usually l is along heading (x), w is along y, h is along z.
    
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [0, 0, 0, 0, h, h, h, h] # Assuming center is at bottom?
    # KITTI Tracklet README says: "The object center is the geometric center of the 3D bounding box."
    # So z should be from -h/2 to h/2
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    
    # Rotate and translate
    corners_3d = np.dot(R, corners_3d)
    corners_3d[0, :] += tx
    corners_3d[1, :] += ty
    corners_3d[2, :] += tz
    
    return corners_3d

def main():
    print(f"Searching for XML files in {DATASET_ROOT}/data_3d_bboxes/train/...")
    xml_search_path = os.path.join(DATASET_ROOT, "data_3d_bboxes", "train", "*.xml")
    xml_files = glob.glob(xml_search_path)
    
    if not xml_files:
        print(f"No XML files found in {xml_search_path}.")
        # Fallback to recursive search if not found in specific folder
        print("Falling back to recursive search for tracklet_labels.xml...")
        xml_files = glob.glob(os.path.join(DATASET_ROOT, "**", "tracklet_labels.xml"), recursive=True)
        
    if not xml_files:
        print("No XML files found.")
        return

    print(f"Found {len(xml_files)} XML files.")

    only_seq = os.getenv('K360_SEQ')
    if only_seq:
        only_seq = only_seq.strip()
        xml_files = [p for p in xml_files if os.path.splitext(os.path.basename(p))[0] == only_seq]
        print(f"Filtering to sequence '{only_seq}': {len(xml_files)} XML files")
        if not xml_files:
            return

    xml_limit = os.getenv('K360_XML_LIMIT')
    if xml_limit:
        try:
            xml_files = xml_files[: max(0, int(xml_limit))]
            print(f"Limiting XML files to first {len(xml_files)} due to K360_XML_LIMIT")
        except ValueError:
            pass
    
    for xml_file in xml_files:
        print(f"Processing {xml_file}...")
        
        # Determine sequence directory
        # Case 1: XML is .../data_3d_bboxes/train/SEQUENCE_NAME.xml
        filename = os.path.basename(xml_file)
        seq_name = os.path.splitext(filename)[0]
        
        # Case 2: XML is .../SEQUENCE_NAME/tracklet_labels.xml
        if filename == "tracklet_labels.xml":
             seq_dir_candidate = os.path.dirname(xml_file)
             seq_name = os.path.basename(seq_dir_candidate)
        
        # Look for raw data directory
        # KITTI-360 structure: data_2d_raw/SEQUENCE_NAME
        # KITTI Raw structure: DATE/SEQUENCE_NAME
        
        seq_dir = None
        possible_dirs = [
            os.path.join(DATASET_ROOT, "data_2d_raw", seq_name),
            os.path.join(DATASET_ROOT, seq_name),
            os.path.dirname(xml_file) # If xml is inside the sequence dir
        ]
        
        # Also try to find by date if it's KITTI Raw style (2011_09_26_drive_0001_sync -> 2011_09_26/...)
        if len(seq_name.split('_')) >= 3:
            date_part = "_".join(seq_name.split('_')[:3])
            possible_dirs.append(os.path.join(DATASET_ROOT, date_part, seq_name))
            
        for d in possible_dirs:
            if os.path.exists(d) and os.path.isdir(d):
                # Check if it looks like a data dir (has image_00 or similar)
                if os.path.exists(os.path.join(d, "image_00")) or os.path.exists(os.path.join(d, "image_02")):
                    seq_dir = d
                    break
        
        if not seq_dir:
            print(f"Warning: Could not find raw data directory for sequence {seq_name}. Skipping.")
            continue
            
        print(f"Found sequence directory: {seq_dir}")

        # Load KITTI-360 calibration + poses for this sequence
        calib_root = os.path.join(DATASET_ROOT, 'calibration')
        poses_file = os.path.join(DATASET_ROOT, 'data_poses', seq_name, 'poses.txt')

        if not os.path.exists(calib_root) or not os.path.exists(poses_file):
            print(f"Error: Missing calibration or poses for {seq_name}.")
            print(f"Expected calibration at: {calib_root}")
            print(f"Expected poses at: {poses_file}")
            continue

        try:
            persp = load_perspective_calibration(calib_root)
            cam_to_pose = load_calib_cam_to_pose(calib_root)  # T_pose_cam
            poses = load_poses(poses_file)  # T_world_pose
        except Exception as e:
            print(f"Error loading calibration/poses for {seq_name}: {e}")
            continue

        # Choose camera consistent with KITTI-360 image_00
        cam_name = 'image_00'
        cam_idx = '00'
        if cam_name not in cam_to_pose:
            # fallback to image_02 if needed
            if 'image_02' in cam_to_pose:
                cam_name = 'image_02'
                cam_idx = '02'
        if cam_name not in cam_to_pose:
            print(f"Error: calib_cam_to_pose.txt missing {cam_name}")
            continue

        R_rect = persp.get(f'R_rect_{cam_idx}')
        P_rect = persp.get(f'P_rect_{cam_idx}')
        img_size = persp.get(f'S_rect_{cam_idx}')
        if R_rect is None or P_rect is None:
            print(f"Error: perspective.txt missing R_rect_{cam_idx} or P_rect_{cam_idx}")
            continue

        if img_size is None:
            # fallback size
            img_w, img_h = 1408, 376
        else:
            img_w, img_h = img_size

        # KITTI-360 transform chain:
        #   poses.txt provides T_world_pose (world <- pose)
        #   calib_cam_to_pose provides T_pose_cam (pose <- cam)
        # so T_world_cam = T_world_pose @ T_pose_cam and T_cam_world = inv(T_world_cam)

        T_pose_cam = cam_to_pose[cam_name]  # pose <- cam
        
        # Parse KITTI-360 objects
        objects = parseXML(xml_file)
        print(f"Parsed {len(objects)} relevant objects (Car/Pedestrian/Cyclist) from {xml_file}")

        objects_by_frame = defaultdict(list)
        max_frame_from_labels = -1
        for obj in objects:
            if obj.timestamp is not None:
                max_frame_from_labels = max(max_frame_from_labels, obj.timestamp)
                objects_by_frame[obj.timestamp].append(obj)
            else:
                max_frame_from_labels = max(max_frame_from_labels, obj.end_frame)
                for frame_idx in range(obj.start_frame, obj.end_frame + 1):
                    objects_by_frame[frame_idx].append(obj)
        
        # If no objects found, we still might want to generate empty label files for all frames in the sequence
        # To do this, we need to know the number of frames in the sequence.
        # We can check the image directory.
        
        image_dir = None
        possible_img_subdirs = ["data_rect", "data_rgb", "data"]
        base_img_dir = os.path.join(seq_dir, cam_name)
        
        if os.path.exists(base_img_dir):
            for subdir in possible_img_subdirs:
                d = os.path.join(base_img_dir, subdir)
                if os.path.exists(d):
                    image_dir = d
                    break
        
        images = []
        num_frames_from_images = 0
        num_digits = 10
        if image_dir and os.path.exists(image_dir):
            images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.png')])
            num_frames_from_images = len(images)
            if images:
                num_digits = len(images[0].split('.')[0])
             
        # Use the larger of max_frame from tracklets or num_frames from images
        # This ensures we cover all frames even if the last few have no objects,
        # or if there are no objects at all but we have images.
        final_max_frame = max(max_frame_from_labels, num_frames_from_images - 1)

        frame_limit = os.getenv('K360_FRAME_LIMIT')
        if frame_limit:
            try:
                final_max_frame = min(final_max_frame, max(0, int(frame_limit) - 1))
                print(f"Limiting frames to {final_max_frame + 1} due to K360_FRAME_LIMIT")
            except ValueError:
                pass
        
        if final_max_frame < 0:
            print(f"Warning: No frames found for {seq_name} (no tracklets and no images found). Skipping generation.")
            continue

        print(f"Generating labels for {final_max_frame + 1} frames (Labels max: {max_frame_from_labels}, Images: {num_frames_from_images})")

        # Create output directory
        output_dir = os.path.join(seq_dir, OUTPUT_DIR_NAME)
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate label files
        for frame_idx in tqdm(range(final_max_frame + 1), desc="Generating labels"):
            label_file = os.path.join(output_dir, f"{frame_idx:0{num_digits}d}.txt")

            with open(label_file, 'w') as f:
                # Need pose for this frame to compute camera coords.
                if frame_idx not in poses:
                    continue

                T_world_pose = poses[frame_idx]  # world <- pose
                T_world_cam = T_world_pose @ T_pose_cam  # world <- cam
                T_cam_world = np.linalg.inv(T_world_cam)  # cam <- world

                R_cw = T_cam_world[:3, :3]

                for obj in objects_by_frame.get(frame_idx, []):
                    # Dimensions from object transform scale
                    t = obj.transform
                    sx = float(np.linalg.norm(t[:3, 0])); sy = float(np.linalg.norm(t[:3, 1])); sz = float(np.linalg.norm(t[:3, 2]))
                    l = sx; w = sy; h = sz

                    # Object center in world
                    center_world = t @ np.array([0, 0, 0, 1], dtype=np.float32)
                    center_cam = T_cam_world @ center_world
                    center_rect = (R_rect @ center_cam[:3]).astype(np.float32)

                    loc_x = float(center_rect[0])
                    loc_y = float(center_rect[1] + h / 2.0)  # bottom center approx
                    loc_z = float(center_rect[2])

                    # Filter by max depth (KITTI usually < 80m)
                    max_depth = float(os.getenv('K360_MAX_DEPTH', '80.0'))
                    if loc_z > max_depth:
                        continue

                    # Yaw: KITTI-360 object forward axis is local +Y (not +X).
                    R_obj_world = t[:3, :3].astype(np.float32)
                    # remove scale
                    R_obj_world = R_obj_world / (np.linalg.norm(R_obj_world, axis=0, keepdims=True) + 1e-9)
                    dir_world = R_obj_world @ np.array([0, 1, 0], dtype=np.float32)
                    dir_cam = R_cw @ dir_world
                    dir_rect = R_rect @ dir_cam
                    ry = float(np.arctan2(dir_rect[0], dir_rect[2]))
                    ry = normalize_angle(ry)

                    theta = float(np.arctan2(loc_x, loc_z)) if loc_z != 0 else 0.0
                    alpha = normalize_angle(ry - theta)

                    # 3D bbox corners for 2D projection
                    if obj.vertices is None:
                        continue
                    verts = obj.vertices.astype(np.float32)
                    verts_h = np.concatenate([verts, np.ones((verts.shape[0], 1), dtype=np.float32)], axis=1)
                    corners_world = (t @ verts_h.T).T
                    corners_cam = (T_cam_world @ corners_world.T).T[:, :3]
                    corners_rect = (R_rect @ corners_cam.T).T

                    uv, depth = project_points(P_rect, corners_rect)
                    # Defaults chosen to match typical older converters:
                    # - require all corners in front of camera
                    # - skip truncated boxes that would extend outside the image
                    allow_partial_z = _env_flag('K360_ALLOW_PARTIAL_Z', default=False)
                    allow_truncated = _env_flag('K360_ALLOW_TRUNCATED', default=False)

                    if allow_partial_z:
                        keep = depth > 0.1
                        if not np.any(keep):
                            continue
                        uv_use = uv[keep]
                    else:
                        if not np.all(depth > 0.1):
                            continue
                        uv_use = uv

                    x1 = float(np.min(uv_use[:, 0])); y1 = float(np.min(uv_use[:, 1]))
                    x2 = float(np.max(uv_use[:, 0])); y2 = float(np.max(uv_use[:, 1]))

                    if not allow_truncated:
                        # Require bbox fully inside image; do not clip.
                        if x1 < 0.0 or y1 < 0.0 or x2 > (img_w - 1.0) or y2 > (img_h - 1.0):
                            continue
                    else:
                        # Clip to image
                        x1 = max(0.0, min(x1, img_w - 1.0))
                        x2 = max(0.0, min(x2, img_w - 1.0))
                        y1 = max(0.0, min(y1, img_h - 1.0))
                        y2 = max(0.0, min(y2, img_h - 1.0))

                    x1, y1, x2, y2 = _bbox_floor_ceil(x1, y1, x2, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    truncated = 0.0
                    occluded = 0

                    add_score = os.getenv('K360_ADD_SCORE')
                    if add_score and add_score not in ('0', 'false', 'False'):
                        line = (
                            f"{obj.kitti_type} {truncated:.1f} {occluded} {alpha:.6f} "
                            f"{x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} "
                            f"{h:.6f} {w:.6f} {l:.6f} "
                            f"{loc_x:.6f} {loc_y:.6f} {loc_z:.6f} {ry:.6f} 1.0\n"
                        )
                    else:
                        line = (
                            f"{obj.kitti_type} {truncated:.2f} {occluded} {alpha:.2f} "
                            f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
                            f"{h:.2f} {w:.2f} {l:.2f} "
                            f"{loc_x:.2f} {loc_y:.2f} {loc_z:.2f} {ry:.2f}\n"
                        )
                    f.write(line)

if __name__ == "__main__":
    main()
