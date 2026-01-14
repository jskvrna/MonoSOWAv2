import os
import sys

def find_missing_files(folder_path):
    start_number = 0
    end_number = 7481
    missing_files = []

    for number in range(start_number, end_number + 1):
        filename = f"{number:06d}.txt"
        file_path = os.path.join(folder_path, filename)

        if not os.path.exists(file_path):
            missing_files.append(filename)

    return missing_files

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py folder_path")
    else:
        folder_path = sys.argv[1]

        missing_files = find_missing_files(folder_path)

        if missing_files:
            print("Missing files:")
            for missing_file in missing_files:
                print(missing_file)
        else:
            print("No missing files.")
