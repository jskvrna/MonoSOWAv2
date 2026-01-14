import random

# Open the input and output files
with open('/path/to/OpenPCDet/data/kitti/ImageSets/train.txt', 'r') as infile, open('/path/to/OpenPCDet/data/kitti/ImageSets/train_sub.txt', 'w') as outfile:
    # Read all lines from the input file
    lines = infile.readlines()

    # Set the number of lines you want to subsample
    num_lines_to_subsample = 500

    # Get a random sample of lines
    random_lines = random.sample(lines, num_lines_to_subsample)

    # Sort the randomly selected lines
    random_lines.sort()

    # Write the subsampled and sorted lines to the output file
    outfile.writelines(random_lines)
