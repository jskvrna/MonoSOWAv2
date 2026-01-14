import os
import pickle
import torch


def move_to_cpu(data):
    if isinstance(data, torch.Tensor):
        return data.cpu()
    elif isinstance(data, list):
        return [move_to_cpu(item) for item in data]
    elif isinstance(data, dict):
        return {key: move_to_cpu(value) for key, value in data.items()}
    return data


def convert_pickle_to_cpu(input_path, output_path):
    # Load the data
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    # Move all tensors to CPU
    data = move_to_cpu(data)

    # Save the data back
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"Converted {input_path} and saved to {output_path}")


def process_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            input_path = os.path.join(directory, filename)
            output_path = os.path.join(directory, filename.replace('.pkl', '_cpu.pkl'))
            convert_pickle_to_cpu(input_path, output_path)


# Example usage
directory = "/path/to/output/frames_waymo_mvit_2dtrack/homographies/"
process_directory(directory)
