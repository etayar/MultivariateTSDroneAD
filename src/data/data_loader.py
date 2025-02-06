import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class UAVTimeSeriesDataset(Dataset):
    """
    Custom PyTorch Dataset for multivariate time-series UAV data.
    Each sample is a full patch file, with a corresponding label.
    """

    def __init__(self, data, labels):
        """
        Args:
            data (list): List of numpy arrays representing full patches.
            labels (list): List of labels (0 or 1) corresponding to each patch.
        """
        self.data = torch.tensor(np.array(data, dtype=np.float32))  # Ensures valid tensor
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_uav_data(normal_path: str, failure_path: str):
    """
    Loads UAV time-series data from two directories and assigns labels.

    Args:
        normal_path (str): Directory containing normal flight patches (label 0).
        failure_path (str): Directory containing anomalous flight patches (label 1).

    Returns:
        data (list): List of numpy arrays containing full patches.
        labels (list): List of corresponding labels (0 or 1).
    """
    data = []
    labels = []

    # Load normal patches (label = 0)
    for file in os.listdir(normal_path):
        if file.endswith(".csv"):
            full_path = os.path.join(normal_path, file)
            df = pd.read_csv(full_path)

            # Drop timestamp column if exists
            if "timestamp" in df.columns:
                df = df.drop(columns=["timestamp"])

            # Keep only numeric columns
            df = df.select_dtypes(include=[np.number])

            data.append(df.values)  # Store full patch
            labels.append(0)  # Normal label

    # Load anomalous patches (label = 1)
    for file in os.listdir(failure_path):
        if file.endswith(".csv"):
            full_path = os.path.join(failure_path, file)
            df = pd.read_csv(full_path)

            # Drop timestamp column if exists
            if "timestamp" in df.columns:
                df = df.drop(columns=["timestamp"])

            # Keep only numeric columns
            df = df.select_dtypes(include=[np.number])

            data.append(df.values)  # Store full patch
            labels.append(1)  # Anomalous label

    # Check if all shapes are the same
    shapes = set(arr.shape for arr in data)
    if len(shapes) > 1:
        raise ValueError(f"Data patches have inconsistent shapes: {shapes}")

    return data, labels


def load_and_split_time_series_data(normal_path: str, failure_path: str, batch_size=32, random_state=42):
    """
    Loads UAV time-series data from normal and failure directories, splits into train/val/test sets, and returns DataLoaders.
    """
    # Load UAV time-series patches and labels
    data, labels = load_uav_data(normal_path, failure_path)

    # Convert data into a proper NumPy array of float32
    data = np.array(data, dtype=np.float32)  # Fix dtype issue

    labels = np.array(labels)

    # Split into training and temporary sets (validation + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        data, labels, test_size=0.3, random_state=random_state
    )

    # Split temporary set into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state
    )

    # Create PyTorch datasets
    train_dataset = UAVTimeSeriesDataset(X_train, y_train)
    val_dataset = UAVTimeSeriesDataset(X_val, y_val)
    test_dataset = UAVTimeSeriesDataset(X_test, y_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    synthetic_data_querying = True

    if synthetic_data_querying:
        # synthetic_data # synthetic_data # synthetic_data # synthetic_data # synthetic_data # synthetic_data # synthetic_data
        normal_data_path = "/Users/etayar/PycharmProjects/MultivariateTSDroneAD/uav_data/synthetic_data/anomalous_data"
        failure_data_path = "/Users/etayar/PycharmProjects/MultivariateTSDroneAD/uav_data/synthetic_data/normal_data"
        # synthetic_data # synthetic_data # synthetic_data # synthetic_data # synthetic_data # synthetic_data # synthetic_data
    else:
        # REAL DATA DIRECTORIES
        normal_data_path = "/Users/etayar/PycharmProjects/MultivariateTSDroneAD/uav_data/boaz_csv_flight_data/normal_data"
        failure_data_path = "/Users/etayar/PycharmProjects/MultivariateTSDroneAD/uav_data/boaz_csv_flight_data/anomalous_data"
        # REAL DATA DIRECTORIES

    train_loader, val_loader, test_loader = load_and_split_time_series_data(
        normal_data_path, failure_data_path, batch_size=32
    )

    exit()