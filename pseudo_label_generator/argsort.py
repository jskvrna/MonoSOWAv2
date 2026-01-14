import numpy as np

# Read the .txt file and convert it into a 1D array
with open('3d/data/train_rand.txt', 'r') as f:
    content = f.read().strip()  # Assuming the numbers are separated by spaces
    numbers = np.array(content.split(','))

# Convert the strings to integers
numbers = numbers.astype(int)

# Use argsort() to get the indices that would sort the array
sorted_indices = np.argsort(numbers)

# Print the indices
print(sorted_indices)
print(numbers[sorted_indices])

#sorted_indices = sorted_indices + 1

with open('3d/data/train_mapping.txt', 'r') as f:
    lines = f.readlines()
    strings = np.array(lines)

with open('/path/to/OpenPCDet/data/kitti/ImageSets/train_orig.txt', 'r') as f:
    lines = f.readlines()
    train_orig = np.array(lines)

train_orig = train_orig.astype(int)

out = []
how_much = 371
cnter = 0

with open('train_sorted.txt', 'w') as f:
    for idx, index in enumerate(sorted_indices):
        if cnter < how_much:
            if index in train_orig:
                #out.append(f"{index:06}\n")
                f.write(f"{index:06}\n")
                cnter += 1
                # Print the array
                #print(strings[numbers[index] - 1])

#out = sorted(out, key=lambda x: int(x.split()[0]))
# Print the array
#print(out)

# Write the array to a new .txt file

#with open('train_sorted.txt', 'w') as f:
#    for item in out:
#        f.write(item)