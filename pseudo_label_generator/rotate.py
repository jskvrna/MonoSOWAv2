import glob
import numpy as np

path = "/path/to/KITTI/label_pseudo_rotated/"

for file in sorted(glob.glob(path + "*.txt")):
    with open(file, "r") as f:
        lines = f.readlines()
    modified_lines = []
    for line in lines:
        split = line.split(" ")
        if split[3] == "-10":
            angle = float(split[-3])
            angle -= np.pi/2
            if angle > np.pi:
                angle -= 2 * np.pi
            elif angle < -np.pi:
                angle += 2 * np.pi
            split[-3] = str(angle)
        else:
            print("Skipping invalid line:", split)

        modified_lines.append(" ".join(split))

    with open(file, "w") as f:
        f.writelines(modified_lines)