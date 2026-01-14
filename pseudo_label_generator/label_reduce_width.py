import os
import re

# Define the directory containing the .txt files
directory = '/path/to/labels/'  # Change this to the directory where your .txt files are located (redacted)

# Loop through each file
for i in range(7482):
    filename = str(i).zfill(6) + ".txt"
    filepath = os.path.join(directory, filename)

    # Check if the file exists
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            lines = file.readlines()

        # Process each line
        with open(filepath, 'w') as file:
            for line in lines:
                if re.match(r'^[Cc]ar', line):
                    values = line.split()
                    new_tenth_value = "{:.2f}".format(float(values[9]) - 0.05)
                    values[9] = new_tenth_value
                    updated_line = ' '.join(values) + '\n'
                    file.write(updated_line)
                else:
                    file.write(line)
    else:
        print(f"File {filename} does not exist.")
