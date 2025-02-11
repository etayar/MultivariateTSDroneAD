import os
from pathlib import Path
from typing import Dict
import pandas as pd


def parse_special_file(file_path: str) -> pd.DataFrame:
    """
    Parses the special file format where each line has:
    'start-end:sensor1,sensor2,...' and converts it to a DataFrame.
    """
    data = []

    with open(file_path, 'r') as file:
        for line in file:
            if ':' in line:  # Ensure correct format
                range_part, sensors_part = line.strip().split(':')
                sensors_list = sensors_part.split(',')
                data.append({'anomalies_range': range_part, 'sensors': sensors_list})

    return pd.DataFrame(data)


def convert_txt_to_csv(from_paths_d: Dict[str, list]):
    for dir_name, from_paths in from_paths_d.items():
        for from_path in from_paths:
            # Check if this file belongs to 'interpretation_label'
            is_special_format = 'interpretation_label' in from_path

            if is_special_format:
                df = parse_special_file(from_path)
            else:
                # Read the .txt file as a DataFrame (comma-separated)
                df = pd.read_csv(from_path, header=None)  # No header in the input file

            # Save the DataFrame as a CSV file
            tmp = from_path.split('/')
            save_dir = '/'.join(tmp[:-1]) + '_csv'
            to_path = save_dir + '/' + tmp[-1].split('.')[0] + '.' + 'csv'

            # Check and create directory
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            df.to_csv(to_path, index=False, header=False)

    print('Done')


def get_files_paths(directories: list):

    from_paths_dict = {}

    for directory in directories:
        dir_name = directory.name
        # Get all files matching the pattern "machine-*.txt"
        from_paths_dict[dir_name] = [str(file) for file in directory.glob("machine-*.txt")]

    return from_paths_dict


if __name__ == '__main__':
    # List of directories
    directories = [
        Path("/Users/etayar/PycharmProjects/MultivariateTSDroneAD/ServerMachineDataset/test_label"),
        Path("/Users/etayar/PycharmProjects/MultivariateTSDroneAD/ServerMachineDataset/test"),
        Path("/Users/etayar/PycharmProjects/MultivariateTSDroneAD/ServerMachineDataset/train"),
        Path("/Users/etayar/PycharmProjects/MultivariateTSDroneAD/ServerMachineDataset/interpretation_label")
    ]
    from_paths_dict = get_files_paths(directories)

    convert_txt_to_csv(from_paths_dict)

    exit()