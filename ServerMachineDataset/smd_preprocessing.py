import os
from pathlib import Path
from typing import Dict
import pandas as pd


def has_overlap(i, sample_size, lower_bound, high_bound):
    start1, end1 = i, i + sample_size  # First range
    start2, end2 = lower_bound, high_bound  # Second range

    return max(start1, start2) <= min(end1, end2)


def build_tagged_dataset(sensors_directories):
    anomalous_data_path = "/Users/etayar/PycharmProjects/MultivariateTSDroneAD/ServerMachineDataset/anomalous_data"
    normal_data_path = "/Users/etayar/PycharmProjects/MultivariateTSDroneAD/ServerMachineDataset/normal_data"

    tagged_dataset_directories = [anomalous_data_path, normal_data_path]
    for pth in tagged_dataset_directories:
        # Check and create directory
        if not os.path.exists(pth):
            os.makedirs(pth)

    sample_size = 10000

    appropriate_labels = {
        'train': Path("/Users/etayar/PycharmProjects/MultivariateTSDroneAD/ServerMachineDataset/interpretation_label"),
        'test': Path("/Users/etayar/PycharmProjects/MultivariateTSDroneAD/ServerMachineDataset/test_label")
    }

    for dir in sensors_directories:

        dir_name = dir.name
        labels_path = appropriate_labels[dir_name]

        # Get all files in the directory
        sensors_pths = list(dir.glob("*.txt"))  # This gets all files and directories
        labels_pths = list(labels_path.glob("*.txt"))

        for i, sensor_pths in enumerate(sensors_pths):
            machine = sensor_pths.name

            # Check if this file belongs to 'interpretation_label'
            is_special_format = 'interpretation_label' == labels_path.name

            if is_special_format:
                labels = parse_special_file(labels_pths[i])
                anomalies_ranges = labels['anomalies_range']
                ranges_list = [[int(x) for x in r.split('-')] for r in anomalies_ranges]
            else:
                # Read the .txt file as a DataFrame (comma-separated)
                labels = pd.read_csv(labels_pths[i], header=None)  # No header in the input file

            sensors = pd.read_csv(sensor_pths, header=None)

            residual_df = pd.DataFrame()  # Store residual rows

            idx = 0
            for i in range(0, len(sensors), sample_size):  # Step by sample_size (no overlap)
                sample_df = pd.concat(
                    [residual_df, sensors.iloc[i: i + sample_size]]).copy()  # Add residual rows to next sample

                if len(sample_df) < sample_size:
                    residual_df = sample_df  # Save remaining rows for next iteration
                    continue  # Skip incomplete samples for now

                residual_df = pd.DataFrame()  # Reset residuals after using them

                # Determine label (1 if any anomaly is present in the sample)
                if is_special_format:
                    lower_bound = ranges_list[idx][0]
                    high_bound = ranges_list[idx][1]
                    label = has_overlap(i, sample_size, lower_bound, high_bound)
                else:
                    # Read the .txt file as a DataFrame (comma-separated)
                    label = 1 if labels.iloc[i: i + sample_size].values.sum() > 0 else 0

                if label == 1:
                    file_pth = anomalous_data_path + '/' + machine.split('.')[0] + '.csv'
                    sample_df.to_csv(file_pth, index=False, header=False)
                else:
                    file_pth = normal_data_path + '/' + machine.split('.')[0] + '.csv'
                    sample_df.to_csv(file_pth, index=False, header=False)


if __name__ == '__main__':
    sensors_dirs = [
        Path("/Users/etayar/PycharmProjects/MultivariateTSDroneAD/ServerMachineDataset/test"),
        Path("/Users/etayar/PycharmProjects/MultivariateTSDroneAD/ServerMachineDataset/train")
    ]
    build_tagged_dataset(sensors_dirs)

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