import cv2
import os

def draw_2d_bboxes(image_dir, labels_dir):
    # Get list of all image files in the directory and sort them
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')])

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        label_file = image_file.replace('.png', '.txt').replace('.jpg', '.txt')
        labels_path = os.path.join(labels_dir, label_file)

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Failed to load image '{image_path}'.")
            continue

        # Read the labels
        if not os.path.exists(labels_path):
            print(f"Error: Labels file '{labels_path}' not found.")
            continue

        with open(labels_path, 'r') as f:
            lines = f.readlines()

        # Parse the labels and draw the bounding boxes
        for line in lines:
            parts = line.strip().split(' ')
            if parts[0] == 'Car' or parts[0] == 'car':
                x1 = int(float(parts[4]))
                y1 = int(float(parts[5]))
                x2 = int(float(parts[6]))
                y2 = int(float(parts[7]))
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Show the image with bounding boxes
        cv2.imshow('Image with 2D Bounding Boxes', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Example usage
image_dir = "/path/to/datasets/KITTI/object_detection/training/image_2"
labels_dir = "/path/to/output/labels_kitti_pseudo_lidar_woscale"
draw_2d_bboxes(image_dir, labels_dir)