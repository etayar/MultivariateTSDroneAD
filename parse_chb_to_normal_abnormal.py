import pandas as pd
import pyedflib
import numpy as np
import os
import re
from multiprocessing import Pool
import glob


def collect_summary_seizures(summary_pth):
    chb_seizures = {}

    with open(summary_pth, "r") as f:
        seizure_file = None

        for line in f:
            # Extract seizure file name
            match = re.search(r'File Name:\s*([\w\d_]+)\.edf', line)
            if match:
                seizure_file = match.group(1)

            # Extract seizure event times
            pattern = r"Seizure \d+ (Start|End) Time:"
            if seizure_file and re.search(pattern, line):  # Ensure seizure_file is set
                if seizure_file not in chb_seizures:
                    chb_seizures[seizure_file] = {}

                # Determine if it’s Start or End time point
                boundary = 'start' if 'start' in line.lower() else 'end'

                # Extract seizure number
                match = re.search(r'Seizure\s+(\d+)', line)
                seizure_number = match.group(1) if match else "Unknown"

                boundary_key = f"{boundary}_{seizure_number}"

                # Extract time point
                time_match = re.search(r'(\d+) seconds', line)
                time_point = int(time_match.group(1)) if time_match else None

                chb_seizures[seizure_file][boundary_key] = time_point

    return chb_seizures


def open_edf_file(f_pth):
    with pyedflib.EdfReader(f_pth) as f:
        n_channels = f.signals_in_file
        signals = np.array([f.readSignal(i) for i in range(n_channels)])
    return signals


def cut_eeg_signals(signal, summary, file_name):
    """ Cuts EEG signals to 64,000 samples while ensuring seizures are included. """

    total_samples = signal.shape[1]  # Total EEG time points

    if total_samples < 64000:  # Ignore short signals
        print(f"Skipping {file_name}: Only {total_samples} samples (too short)")
        return None

    if file_name in summary:
        start_p = summary[file_name]['start_1']
        end_p = summary[file_name]['end_1']

        start_signal = start_p
        end_signal = start_p + 64000

        # If the seizure is too close to the end, shift the window back
        if start_signal < end_p < end_signal:
            shift_back = end_signal - total_samples
            if shift_back > 0:  # Prevent index error
                start_signal = max(0, start_signal - shift_back)
                end_signal = min(total_samples, start_signal + 64000)

        # Ensure the cut segment is exactly 64,000 samples
        if end_signal - start_signal == 64000:
            return signal[:, start_signal:end_signal]
        else:
            print(
                f"Skipping {file_name}: Could not extract exact 64,000 samples (Adjusted to {end_signal - start_signal})")
            return None
    else:
        return signal[:, :64000]  # Take first 64,000 samples for normal signals


def save_signal_to_folder(folder, file_name, signal):
    """ Saves the EEG signal as a separate .npy file in the given folder. """
    os.makedirs(folder, exist_ok=True)  # Ensure folder exists
    np.save(os.path.join(folder, f"{file_name}.npy"), signal)


def process_file(args):
    """ Helper function to process a single EEG file in parallel. """
    file_pth, summary, normal_folder, abnormal_folder = args
    file_name = os.path.basename(file_pth)
    signal_name = file_name.split('.')[0]  # Remove .edf extension

    signals = open_edf_file(file_pth)
    cut_signal = cut_eeg_signals(signals, summary, signal_name)

    if cut_signal is None:
        return  # Skip invalid files

    folder = abnormal_folder if summary and signal_name in summary else normal_folder
    save_signal_to_folder(folder, signal_name, cut_signal)


def chb_to_normal_abnormal(from_dir, to_dir):
    normal_folder = os.path.join(to_dir, "normal")
    abnormal_folder = os.path.join(to_dir, "abnormal")

    os.makedirs(normal_folder, exist_ok=True)
    os.makedirs(abnormal_folder, exist_ok=True)

    file_list = []

    for chb_folder in os.listdir(from_dir):
        if not chb_folder.startswith("chb"):  # Skip non-CHB folders
            continue

        chb_folder_pth = os.path.join(from_dir, chb_folder)
        if not os.path.isdir(chb_folder_pth):
            continue  # Skip non-folder files

        summary = None  # Reset for each folder
        for file_name in os.listdir(chb_folder_pth):
            if file_name == f"{chb_folder}-summary.txt":
                summary = collect_summary_seizures(os.path.join(chb_folder_pth, file_name))

        for file_name in os.listdir(chb_folder_pth):
            file_pth = os.path.join(chb_folder_pth, file_name)
            if file_name.endswith(".edf"):  # Process EEG data
                file_list.append((file_pth, summary, normal_folder, abnormal_folder))

    # Use multiprocessing to process files in parallel
    with Pool() as pool:
        pool.map(process_file, file_list)

    print("EEG data saved:")
    print(f"   - Normal signals -> {normal_folder}")
    print(f"   - Abnormal signals -> {abnormal_folder}")


if __name__ == '__main__':
    path = '/Users/etayar/PycharmProjects/eec_chb'
    to_pth = '/Users/etayar/PycharmProjects'

    chb_to_normal_abnormal(from_dir=path, to_dir=to_pth)

    # Load individual EEG files from normal/ and abnormal/ folders
    normal_files = glob.glob(os.path.join(to_pth, "normal", "*.npy"))
    abnormal_files = glob.glob(os.path.join(to_pth, "abnormal", "*.npy"))

    if normal_files:
        sample_normal = np.load(normal_files[0])
        print(f"Loaded Normal Sample {normal_files[0]} with shape {sample_normal.shape}")

    if abnormal_files:
        sample_abnormal = np.load(abnormal_files[0])
        print(f"Loaded Abnormal Sample {abnormal_files[0]} with shape {sample_abnormal.shape}")

    exit()
