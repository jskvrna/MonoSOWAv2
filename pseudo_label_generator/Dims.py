import glob
import numpy as np

path = "/path/to/MonoDETR/data/k360/training/label_2/"

length = []
width = []
height = []
for file in sorted(glob.glob(path + "*.txt")):
    with open(file, "r") as f:
        lines = f.readlines()
    for line in lines:
        split = line.split(" ")
        length.append(float(split[10]))
        width.append(float(split[9]))
        height.append(float(split[8]))

length = np.array(length)
width = np.array(width)
height = np.array(height)

print("Length: ", np.mean(length), np.std(length), np.median(length))
print("Width: ", np.mean(width), np.std(width), np.median(width))
print("Height: ", np.mean(height), np.std(height), np.median(height))