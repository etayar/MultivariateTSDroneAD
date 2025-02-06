import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class UAVTimeSeriesDataset(Dataset):
    """
    Custom PyTorch Dataset for multivariate time-series UAV data.
    Each sample is a patch of shape (S, T), with a corresponding label.
    """

    def __init__(self, patches, labels):
        """
        Args:
            patches (numpy.ndarray): A list or array of shape (m, S, T), where m is the number of patches.
            labels (numpy.ndarray): A list or array of shape (m,), representing the labels (0 or 1).
        """
        self.patches = torch.tensor(patches, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx], self.labels[idx]


def load_uav_time_series(data_path, patch_size_T, label_mapping):
    """
    Loads UAV multivariate time-series data and organizes it into patches.

    Args:
        data_path (str): Path to the directory containing UAV CSV files.
        patch_size_T (int): The fixed time length T for each patch.
        label_mapping (dict): A dictionary mapping each file name to its corresponding label (0 or 1).

    Returns:
        patches (numpy.ndarray): Array of shape (m, S, T) containing the multivariate time-series patches.
        labels (numpy.ndarray): Array of shape (m,) with corresponding labels (0 or 1).
    """
    patches = []
    labels = []

    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            full_path = os.path.join(data_path, file)
            df = pd.read_csv(full_path)

            # Assuming first column is timestamps, drop it if necessary
            if "timestamp" in df.columns:
                df = df.drop(columns=["timestamp"])

            data_array = df.values.T  # Transpose to get (S, T) shape

            # Ensure patch size fits within the available time steps
            num_time_steps = data_array.shape[1]
            num_patches = num_time_steps // patch_size_T  # Number of patches

            for i in range(num_patches):
                patch = data_array[:, i * patch_size_T:(i + 1) * patch_size_T]  # Extract patch
                patches.append(patch)

                # Assign the label based on the filename
                if file in label_mapping:
                    labels.append(label_mapping[file])
                else:
                    raise ValueError(f"Label for {file} is missing in label_mapping.")

    patches = np.array(patches)  # Shape: (m, S, T)
    labels = np.array(labels)  # Shape: (m,)

    return patches, labels

def load_and_split_time_series_data(data_path, patch_size_T, label_mapping, batch_size=32, random_state=42):
    """
    Loads UAV time-series data, splits it into train/val/test sets, and returns DataLoaders.

    Args:
        data_path (str): Path to the directory containing UAV CSV files.
        patch_size_T (int): The fixed time length T for each patch.
        label_mapping (dict): A dictionary mapping each file name to its corresponding label (0 or 1).
        batch_size (int): Batch size for DataLoaders.
        random_state (int): Random seed for reproducibility.

    Returns:
        train_loader, val_loader, test_loader: PyTorch DataLoaders for training, validation, and testing.
    """
    # Load UAV time-series patches and labels
    patches, labels = load_uav_time_series(data_path, patch_size_T, label_mapping)

    # Split into training and temporary sets (validation + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        patches, labels, test_size=0.3, random_state=random_state
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
    data_path = "/path/to/your/csv/files"
    patch_size_T = 50  # Define the length of each time-series patch
    label_mapping = {
        "flight_1.csv": 0,  # Normal
        "flight_2.csv": 1,  # Structural Failure
        "flight_3.csv": 0,
        "flight_4.csv": 1
    }

    train_loader, val_loader, test_loader = load_and_split_time_series_data(
        data_path, patch_size_T, label_mapping, batch_size=32
    )

    # Example: Iterating through the DataLoader
    for batch in train_loader:
        X_batch, y_batch = batch
        print("Batch X shape:", X_batch.shape)  # Expected: (batch_size, S, T)
        print("Batch y shape:", y_batch.shape)  # Expected: (batch_size,)
        break

    exit()