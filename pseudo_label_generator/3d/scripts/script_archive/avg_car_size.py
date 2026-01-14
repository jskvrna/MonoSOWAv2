import os, glob
import numpy as np

#Results: [1.52608343 1.62858987 3.88395449], height, width, length (in meters)
data = []
avg_y = []
for pic in glob.glob('/path/to/datasets/kitti/training/label_2/*.txt'):
    # https://open3dmodel.com/3d-models/fiat-uno-car-lowpoly_477976.html
    # Blender used to cut off some planes, the downside of the car
    # Cloud compare used to sample it with points to generate points cloud

    # Open the file and read in the data
    with open(pic, 'r') as file:
        for line in file:
            # Remove any leading/trailing white space from the line
            line = line.strip()

            # Split the line into three values using whitespace as the delimiter
            values = line.split()

            # Convert each value to a float and append it to the data list
            # 3D object dimensions: height, width, length (in meters)
            if str(values[0]) == 'car' or str(values[0]) == 'Car':
                data.append([float(values[8]), float(values[9]), float(values[10])])
                avg_y.append(float(values[12]))


data_array = np.array(data)
sums = np.mean(data,axis=0)
print(sums)

print(np.mean(avg_y))
