import os, random

path_to_kitti_complete = "/path/to/KITTI_complete"

num_of_frames = 0 #47885
stride = 1

with open("mapping.txt", "w") as map_file:
    for parent_folder in sorted(os.listdir(path_to_kitti_complete)):
        parent_path = os.path.join(path_to_kitti_complete, parent_folder)
        if os.path.isdir(parent_path):
            for child_folder in sorted(os.listdir(parent_path)):
                child_path = os.path.join(parent_path, child_folder)
                child_path = os.path.join(child_path, "velodyne_points/data")
                if os.path.isdir(child_path):
                    for filename in sorted(os.listdir(child_path)):
                        file_path = os.path.join(child_path, filename)
                        if os.path.isfile(file_path):
                            num_of_frames += 1
                            val = file_path.split('/')
                            file_name = val[-1].split('.')[0]
                            map_file.write(val[4] + " " + val[5] + " " + file_name + '\n')

with open("rand.txt", "w") as rand_file:
    rng = list(range(0, num_of_frames, stride))
    random.shuffle(rng)
    rng = [str(num) for num in rng]

    to_write = ','.join(rng)

    rand_file.write(to_write)





