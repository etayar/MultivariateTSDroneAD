import pandas as pd
import pyedflib
import numpy as np
import os
import re


def collect_summary_seizures(summary_pth):
    chb_seizures = {}

    with open(summary_pth, "r") as f:
        seizure_file = None  # Ensure it's defined before use

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


def open_dot_seizures(f_pth):
    try:
        df = pd.read_csv(f_pth, sep="\t", encoding="latin-1")  # Try tab-separated first
        print(df.head())
        return df
    except Exception:
        try:
            df = pd.read_csv(f_pth, encoding="latin-1")  # Try CSV
            print(df.head())
            return df
        except Exception as e:
            print(f"Could not read as CSV/TSV: {e}")


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

    if summary is None:
        print("stop")

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


def append_to_disk(file_path, new_data):
    """ Append new data to an existing .npy file without keeping everything in RAM. """
    if os.path.exists(file_path):
        existing_data = np.load(file_path, allow_pickle=True).item()
        existing_data.update(new_data)  # Merge new data
        np.save(file_path, existing_data)  # Save back to disk
    else:
        np.save(file_path, new_data)


def chb_to_normal_abnormal(from_dir, to_dir):
    os.makedirs(to_dir, exist_ok=True)

    temp_normal_path = os.path.join(to_dir, "temp_chb_normal.npy")
    temp_abnormal_path = os.path.join(to_dir, "temp_chb_abnormal.npy")

    for chb_folder in os.listdir(from_dir):
        if not chb_folder.startswith("chb"):  # Skip non-CHB folders
            continue

        chb_folder_pth = os.path.join(from_dir, chb_folder)
        if not os.path.isdir(chb_folder_pth):
            continue  # Skip non-folder files

        summary = None  # Reset for each folder
        for file_name in os.listdir(chb_folder_pth):
            file_pth = os.path.join(chb_folder_pth, file_name)
            if file_name == f"{chb_folder}-summary.txt":
                summary = collect_summary_seizures(file_pth)

        for file_name in os.listdir(chb_folder_pth):
            file_pth = os.path.join(chb_folder_pth, file_name)

            if file_name.endswith(".seizures"):  # Ignore .seizures files
                continue

            elif file_name.endswith(".edf"):  # Process EEG data
                signals = open_edf_file(file_pth)
                signal_name = file_name.split('.')[0]  # Remove .edf extension
                if summary is None:
                    print("stop")
                # Cut signal to 64,000 samples
                cut_signal = cut_eeg_signals(signals, summary, signal_name)

                if cut_signal is None:
                    continue  # Skip short/invalid signals

                temp_dict = {signal_name: cut_signal}

                if summary and signal_name in summary:  # Abnormal case
                    append_to_disk(temp_abnormal_path, temp_dict)
                else:  # Normal case
                    append_to_disk(temp_normal_path, temp_dict)

    # Load temp files & merge into final npy files
    chb_normal = np.load(temp_normal_path, allow_pickle=True).item() if os.path.exists(temp_normal_path) else {}
    chb_abnormal = np.load(temp_abnormal_path, allow_pickle=True).item() if os.path.exists(temp_abnormal_path) else {}

    # Save final files
    np.save(os.path.join(to_dir, "chb_normal.npy"), chb_normal)
    np.save(os.path.join(to_dir, "chb_abnormal.npy"), chb_abnormal)

    # Delete temporary files
    if os.path.exists(temp_normal_path):
        os.remove(temp_normal_path)
    if os.path.exists(temp_abnormal_path):
        os.remove(temp_abnormal_path)

    print("Final files saved:")
    print(f"   - {os.path.join(to_dir, 'chb_normal.npy')}")
    print(f"   - {os.path.join(to_dir, 'chb_abnormal.npy')}")


if __name__ == '__main__':
    path = '/Users/etayar/PycharmProjects/eec_chb'
    to_pth = '/Users/etayar/PycharmProjects'
    chb_to_normal_abnormal(from_dir=path, to_dir=path)

    # load for inspection:
    # Load normal EEG data
    chb_normal = np.load(os.path.join(path, "chb_normal.npy"), allow_pickle=True).item()

    # Load abnormal EEG data
    chb_abnormal = np.load(os.path.join(path, "chb_abnormal.npy"), allow_pickle=True).item()

    exit()