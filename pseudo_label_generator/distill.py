import os.path

with open("/path/to/KITTI-360/filenames/R50-N16-M128-B16/2013_05_28_drive_0010_sync/sampled_image_filenames.txt", 'r') as f:
    lines = f.readlines()
    sampled = []
    for line in lines:
        splited  = line.split(" ")
        for one_split in splited:
            if os.path.exists(one_split):
                sampled.append(one_split)

with open("sampled.txt", 'w') as f:
    for sample in sampled:
        sample = sample.split("/")[-1]
        f.write(sample + "\n")