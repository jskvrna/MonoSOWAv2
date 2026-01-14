import os
import numpy as np

path = "/path/to/Labels/labels_occlusion"  # replace with the path to your directory

for filename in os.listdir(path):
    if filename.endswith(".txt"):
        filepath = os.path.join(path, filename)
        with open(filepath, "r+") as file:
            lines = file.readlines()
            file.seek(0)
            for line in lines:
                line = line.strip()
                line += " " + "0.99" + "\n"
                file.write(line)
            file.truncate()