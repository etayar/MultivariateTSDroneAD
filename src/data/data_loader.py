import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch


def load_uav_data(data_path, label_column="label"):
    """
    Loads UAV data from CSV files in the specified directory.

    Args:
        data_path (str): Path to the directory containing UAV CSV files.
        label_column (str): Name of the column containing labels (default: "label").

    Returns:
        X (numpy.ndarray): Features from the dataset.
        y (numpy.ndarray): Labels from the dataset.
    """
    data_frames = []
    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            full_path = os.path.join(data_path, file)
            data_frames.append(pd.read_csv(full_path))

    # Combine all CSVs into one DataFrame
    data = pd.concat(data_frames, ignore_index=True)

    # Extract features (X) and labels (y)
    y = data[label_column].values  # Assuming "label" column exists
    X = data.drop(columns=[label_column]).values

    return X, y


def load_and_split_data(data_path, label_column="label", batch_size=32, random_state=42):
    """
    Loads UAV data, splits it into train/validation/test sets, and returns DataLoaders.

    Args:
        data_path (str): Path to the directory containing UAV CSV files.
        label_column (str): Name of the column containing labels.
        batch_size (int): Batch size for DataLoaders.
        random_state (int): Random seed for reproducibility.

    Returns:
        train_loader, val_loader, test_loader: PyTorch DataLoaders for training, validation, and testing.
    """
    # Load data from UAV files
    X, y = load_uav_data(data_path, label_column=label_column)

    # Split into training and temporary sets (validation + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )

    # Split temporary set into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state
    )

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
