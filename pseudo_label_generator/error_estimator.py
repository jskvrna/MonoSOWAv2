import os
import numpy as np
import copy
from scipy.spatial.distance import cdist

# Define the paths to the directories
#directory1 = "/path/to/final_result/data"  # (redacted)
directory1 = "/path/to/final_result/data"  # (redacted)
directory2 = "/path/to/kitti/training/label_2"  # (redacted)

np.set_printoptions(precision=3)

errors = []
# Iterate over the files
for i in range(7482):
    filename = f"{i:06d}.txt"
    file_path1 = os.path.join(directory1, filename)
    file_path2 = os.path.join(directory2, filename)
    if os.path.exists(file_path1) and os.path.exists(file_path2):
        with open(file_path1, 'r') as file1, open(file_path2, 'r') as file2:
            # Do something with the files
            pseudoGT = file1.read().splitlines()
            kittiGT = file2.read().splitlines()

            pseudoGT_arr = []
            for k in range(len(pseudoGT)):
                tmp_data = pseudoGT[k].split(" ")
                if tmp_data[0] == "car" or tmp_data[0] == "Car" or tmp_data[0] == "Van" or tmp_data[0] == "Van":
                    tmp_detection = np.zeros(7)
                    for z in range(7):
                        tmp_detection[z] = float(tmp_data[z + 8])
                    pseudoGT_arr.append(tmp_detection)

            kittiGT_arr = []
            for k in range(len(kittiGT)):
                tmp_data = kittiGT[k].split(" ")
                if tmp_data[0] == "car" or tmp_data[0] == "Car" or tmp_data[0] == "Van" or tmp_data[0] == "Van":
                    tmp_detection = np.zeros(7)
                    for z in range(7):
                        tmp_detection[z] = float(tmp_data[z + 8])
                    kittiGT_arr.append(tmp_detection)

            if len(pseudoGT_arr) > 0 and len(kittiGT_arr) > 0:
                #Now lets perform matching
                pseudoGT_arr = np.array(pseudoGT_arr)
                kittiGT_arr = np.array(kittiGT_arr)

                dists = cdist(pseudoGT_arr[:, 3:6], kittiGT_arr[:, 3:6])

                #for each pseudoGT
                for k in range(dists.shape[0]):
                    closest = np.argmin(dists[k, :])
                    closest_from_closest = np.argmin(dists[:, closest])

                    if closest_from_closest == k and dists[k, closest] < 3.:
                        #We got a match
                        errors.append(pseudoGT_arr[k] - kittiGT_arr[closest])
                    else:
                        print("outlier")

errors = np.array(errors)

mean_errors = np.mean(errors, axis=0)
L1_errors = np.mean(np.abs(errors), axis=0)
L2_errors = np.mean(np.power(errors, 2), axis=0)
median_errors = np.median(errors, axis=0)

mean_errors = [f'{x:.3f}' for x in mean_errors]
median_errors = [f'{x:.3f}' for x in median_errors]

print("Mean of errors: ", mean_errors)
print("Median of errors: ", median_errors)
print("Mean of L1 errors: ", L1_errors)
print("Mean of L2 errors: ", L2_errors)






