import os
import pandas as pd
import numpy as np

# Load CSV file
if "COLAB_GPU" in os.environ:

    folder = "/content/drive/My Drive/My_PHD/My_First_Paper/uea_datasets"
else:
    folder = "/Users/etayar/PycharmProjects/MultivariateTSDroneAD/uea_datasets"


def load_uea_multivariate_ts(csv_path):

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
    return X, y


def joint_dimensions_calc():
    uea_datasets = []
    S_max, T_max = 0, 0
    for file in os.listdir(folder):
        uea_datasets.append(file)
        if file.endswith(".csv"):
            full_path = os.path.join(folder, file)
            X, y = load_uea_multivariate_ts(full_path)

            S = X.shape[1]
            T = X.shape[2]

            S_max = S if S > S_max else S_max
            T_max = T if T > T_max else T_max

    joint_dimensions = (S_max, int(np.log(S_max) * T_max + 1))
    print(f"UEA DATA SETS: {uea_datasets}")
    print(f"S Max: {S_max}, T Max: {T_max}")
    print(f"The joint dimensions S X T: {joint_dimensions}")
    return joint_dimensions


if __name__ == '__main__':

    jd = joint_dimensions_calc()

    exit()