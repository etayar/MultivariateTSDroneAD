import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter


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


def load_multiclass_data(pth: str):

    return 'X', 'y'


def load_multilabel_data(pth: str):

    return 'X', 'y'


def load_uea_multivariate_ts(dataset_name):

    # Load CSV file
    if "COLAB_GPU" in os.environ:

        folder = "/content/drive/My Drive/My_PHD/My_First_Paper/uea_datasets"
    else:
        folder = "/Users/etayar/PycharmProjects/MultivariateTSDroneAD/uea_datasets"

    csv_path = os.path.join(folder, dataset_name + ".csv")

    # Read metadata from the first row
    with open(csv_path, "r") as f:
        metadata = f.readline().strip()  # Read first row

    if metadata.startswith("@"):  # Ensure it's a metadata row
        num_channels, time_steps = map(int, metadata[1:].split(","))
    else:
        raise ValueError("Metadata row missing or incorrectly formatted!")

    # Load the actual data (excluding the first row)
    df = pd.read_csv(csv_path, skiprows=1)  # Skip metadata row

    # Extract features and labels
    X_flat = df.iloc[:, :-1].values  # Drop label column
    y = df.iloc[:, -1].values  # Extract labels

    # Reshape back to original 3D format
    num_samples = X_flat.shape[0]
    X = X_flat.reshape(num_samples, num_channels, time_steps)

    print(f"Loaded dataset '{dataset_name}' with shape: {X.shape}")
    return X, y


def load_data(
    multilabel_path: str = None,
    multiclass_path: str = None,
    normal_path: str = None,
    failure_path: str = None,
    experimental_dataset_name: str = None
):

    if normal_path:
        data, label = load_uav_data(normal_path, failure_path)
    elif multiclass_path:
        data, label = load_multiclass_data(multiclass_path)
    elif multilabel_path:
        data, label = load_multilabel_data(multilabel_path)
    else:
        data, label = load_uea_multivariate_ts(experimental_dataset_name)
    return data, label


def load_and_split_time_series_data(split_rates=(0.2, 0.5), batch_size=32, random_state=42, **kwargs):
    """
    Loads UAV time-series data from normal and failure directories, splits into train/val/test sets,
    returns DataLoaders along with class label counts.
    """

    # Load UAV time-series patches and labels
    data, labels = load_data(**kwargs)
    # data, labels = load_uav_data(normal_path, failure_path)

    # Convert data into a proper NumPy array of float32
    data = np.array(data, dtype=np.float32)  # Ensure dtype consistency
    labels = np.array(labels)

    # Count occurrences of each class **before splitting** for proper weighting
    label_counts = Counter(labels)  # Returns a dictionary {class_0: count, class_1: count, ...}

    # Split into training and temporary sets (validation + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        data, labels, test_size=split_rates[0], stratify=labels, random_state=random_state
    )

    # Split temporary set into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=split_rates[1], stratify=y_temp, random_state=random_state
    )

    # Check for missing classes in validation and test
    missing_classes_val = set(np.unique(y_train)) - set(np.unique(y_val))
    missing_classes_test = set(np.unique(y_train)) - set(np.unique(y_test))

    # If missing classes exist, raise an error and abort execution
    if missing_classes_val or missing_classes_test:
        error_message = "ERROR: Some classes are missing in validation/test sets!\n"
        if missing_classes_val:
            error_message += f"Missing in validation set: {missing_classes_val}\n"
        if missing_classes_test:
            error_message += f"Missing in test set: {missing_classes_test}\n"
        raise ValueError(error_message)

    # Create PyTorch datasets
    train_dataset = UAVTimeSeriesDataset(X_train, y_train)
    val_dataset = UAVTimeSeriesDataset(X_val, y_val)
    test_dataset = UAVTimeSeriesDataset(X_test, y_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, label_counts


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

    kwargs = {
        'normal_path': None, # normal_data_path,
        'failure_path': None, # failure_data_path,
        'multilabel_path': '',
        'multiclass_path': ''
    }

    train_loader, val_loader, test_loader, label_counts = load_and_split_time_series_data(**kwargs)

    exit()