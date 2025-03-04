import pandas as pd
import pyedflib
import numpy as np
import os


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


def chb_to_normal_abnormal(dir_pth: str):
    CHB_normal = {}
    CHB_abnormal = {}

    for chb_folder_name in os.listdir(dir_pth):
        if chb_folder_name.startswith('.') or not chb_folder_name.startswith("chb"):
            continue  # Ignore hidden files and non-CHB folders

        chb_folder_pth = os.path.join(dir_pth, chb_folder_name)
        if not os.path.isdir(chb_folder_pth):
            continue  # Skip non-folder files like SUBJECT-INFO

        summary = None  # Reset summary for each patient folder

        for file_name in os.listdir(chb_folder_pth):
            file_pth = os.path.join(chb_folder_pth, file_name)

            if file_name.endswith(".seizures"):  # Ignore .seizures files
                continue

            elif file_name == f"{chb_folder_name}-summary.txt":  # Summary file
                summary = collect_summary_seizures(file_pth)

            elif file_name.endswith(".edf"):  # EEG data files
                signals = open_edf_file(file_pth)
                CHB_normal[file_name.split('.')[0]] = signals
                print(f"Processed {file_name}, Shape: {signals.shape}")  # (n_channels, time_points)

        if summary:  # Only cut EEG if summary was found
            CHB_normal, CHB_abnormal = cut_eeg_signals(
                chb_normal=CHB_normal,
                chb_abnormal=CHB_abnormal,
                summary=summary
            )

    return CHB_normal, CHB_abnormal


def cut_eeg_signals(chb_normal, chb_abnormal, summary):
    new_chb_normal = {}

    for chb_file_name, chb_file in chb_normal.items():
        total_samples = chb_file.shape[1]  # Total EEG time points

        # Immediately print and skip if the file is too short
        if total_samples < 64000:
            print(f"Skipping {chb_file_name}: Only {total_samples} samples (too short)")
            continue  # Skip this file

        if chb_file_name in summary:
            start_p = summary[chb_file_name]['start_point']
            end_p = summary[chb_file_name]['end_point']

            # Default window
            start_signal = start_p
            end_signal = start_p + 64000

            # If the seizure ends too close to the window end, shift window back
            if start_signal < end_p < end_signal:
                shift_back = end_signal - total_samples
                if shift_back > 0:  # Prevent index error
                    start_signal = max(0, start_signal - shift_back)
                    end_signal = min(total_samples, start_signal + 64000)

            # Ensure final segment is exactly 64,000 samples
            if end_signal - start_signal == 64000:
                chb_abnormal[chb_file_name] = chb_file[:, start_signal:end_signal]
            else:
                print(f"Skipping {chb_file_name}: Could not extract exact 64,000 samples (Adjusted to {end_signal - start_signal})")

        else:  # Normal signals
            new_chb_normal[chb_file_name] = chb_file[:, :64000]  # Take first 64,000 samples

    return new_chb_normal, chb_abnormal


def collect_summary_seizures(summary_pth):
    chb_seizures = {}
    previous_lines = []  # Store previous lines
    with open(summary_pth, "r") as f:
        for i, line in enumerate(f):
            previous_lines.append(line.strip())  # Keep track of previous lines
            if len(previous_lines) > 6:
                previous_lines.pop(0)  # Keep only last 3 lines in memory

            if 'Seizure End Time:' in line:
                num_seizures = int(previous_lines[3][-1])
                if num_seizures >= 1:
                    seizure_file = previous_lines[0].split(':')[-1].strip().split('.')[0]
                    chb_seizures[seizure_file] = {
                        'start_point': int(previous_lines[4].split(':')[-1].strip().split(' ')[0]),
                        'end_point': int(previous_lines[5].split(':')[-1].strip().split(' ')[0])
                    }
    return chb_seizures


def save_normal_abnormal(from_dir: str, to_dir: str):
    chb_normal, chb_abnormal = chb_to_normal_abnormal(dir_pth=from_dir)

    os.makedirs(to_dir, exist_ok=True)  # Ensure the directory exists

    # Save normal signals
    np.save(os.path.join(to_dir, "chb_normal.npy"), chb_normal)
    print(f"Saved chb_normal to {os.path.join(to_dir, 'chb_normal.npy')}")

    # Save abnormal signals
    np.save(os.path.join(to_dir, "chb_abnormal.npy"), chb_abnormal)
    print(f"Saved chb_abnormal to {os.path.join(to_dir, 'chb_abnormal.npy')}")

