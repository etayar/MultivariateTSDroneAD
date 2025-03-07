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
            match = re.search(r'File Name:\s*([\w\d_]+)\.edf', line)
            if match:
                seizure_file = match.group(1)
            match = re.search(r'Seizure\s+(\d+)\s+(Start|End) Time:\s+(\d+)\s+seconds', line)
            if match and seizure_file:
                seizure_number = match.group(1)
                boundary = match.group(2).lower()
                time_point = int(match.group(3))
                if seizure_file not in chb_seizures:
                    chb_seizures[seizure_file] = {}
                chb_seizures[seizure_file][f"{boundary}_{seizure_number}"] = time_point
    return chb_seizures

def open_edf_file(f_pth):
    with pyedflib.EdfReader(f_pth) as f:
        n_channels = f.signals_in_file
        signals = np.array([f.readSignal(i) for i in range(n_channels)])
    return signals

def cut_eeg_signals(signal, summary, file_name, sample_length=16000, sampling_rate=256):
    total_samples = signal.shape[1]
    extracted_segments = []
    if total_samples < sample_length:
        print(f"Skipping {file_name}: Only {total_samples} samples (too short)")
        return None
    if file_name in summary:
        seizure_segments = []
        seizure_indices = sorted([key for key in summary[file_name] if key.startswith("start")])
        for i, seizure_key in enumerate(seizure_indices):
            seizure_start_sec = summary[file_name][seizure_key]
            seizure_start_sample = seizure_start_sec * sampling_rate
            segment_start = max(0, min(seizure_start_sample, total_samples - sample_length))
            seizure_segments.append(signal[:, segment_start:segment_start + sample_length])
        return seizure_segments
    else:
        for i in range(2):
            start_idx = np.random.randint(0, total_samples - sample_length + 1)
            extracted_segments.append(signal[:, start_idx:start_idx + sample_length])
        return extracted_segments

def save_signal_to_folder(folder, file_name, segments):
    os.makedirs(folder, exist_ok=True)
    for i, segment in enumerate(segments):
        np.save(os.path.join(folder, f"{file_name}_{i}.npy"), segment)

def process_file(args):
    file_pth, summary_for_file, normal_folder, abnormal_folder = args
    file_name = os.path.basename(file_pth).split('.')[0]
    signals = open_edf_file(file_pth)
    if signals.shape[0] != 23:
        print(f"Skipping {file_name}: {signals.shape[0]} channels (not 23)")
        return
    cut_segments = cut_eeg_signals(signals, summary_for_file, file_name)
    if cut_segments is None:
        return
    folder = abnormal_folder if file_name in summary_for_file else normal_folder
    save_signal_to_folder(folder, file_name, cut_segments)

def chb_to_normal_abnormal(from_dir, to_dir):
    normal_folder = os.path.join(to_dir, "normal")
    abnormal_folder = os.path.join(to_dir, "abnormal")
    os.makedirs(normal_folder, exist_ok=True)
    os.makedirs(abnormal_folder, exist_ok=True)
    file_list = []
    for chb_folder in os.listdir(from_dir):
        if not chb_folder.startswith("chb"):
            continue
        chb_folder_pth = os.path.join(from_dir, chb_folder)
        if not os.path.isdir(chb_folder_pth):
            continue
        summary_path = os.path.join(chb_folder_pth, f"{chb_folder}-summary.txt")
        summary = collect_summary_seizures(summary_path) if os.path.exists(summary_path) else {}
        for file_name in os.listdir(chb_folder_pth):
            file_pth = os.path.join(chb_folder_pth, file_name)
            if file_name.endswith(".edf"):
                signal_name = file_name.split(".")[0]
                summary_for_file = summary.get(signal_name, {})
                file_list.append((file_pth, summary_for_file, normal_folder, abnormal_folder))
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
