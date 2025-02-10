r"""
Citation:
@misc{skab,
  author = {Katser, Iurii D. and Kozitsin, Vyacheslav O.},
  title = {Skoltech Anomaly Benchmark (SKAB)},
  year = {2020},
  publisher = {Kaggle},
  howpublished = {\url{https://www.kaggle.com/dsv/1693952}},
  DOI = {10.34740/KAGGLE/DSV/1693952}
}

The SKAB v0.9 corpus contains 35 individual data files in .csv format (datasets). The data folder
 contains datasets from the benchmark. The structure of the data folder is presented in the structure
 file. Each dataset represents a single experiment and contains a single anomaly. The datasets represent
 a multivariate time series collected from the sensors installed on the testbed. Columns in each data
 file are following:

datetime - Represents dates and times of the moment when the value is written to the database (YYYY-MM-DD hh:mm:ss)
Accelerometer1RMS - Shows a vibration acceleration (Amount of g units)
Accelerometer2RMS - Shows a vibration acceleration (Amount of g units)
Current - Shows the amperage on the electric motor (Ampere)
Pressure - Represents the pressure in the loop after the water pump (Bar)
Temperature - Shows the temperature of the engine body (The degree Celsius)
Thermocouple - Represents the temperature of the fluid in the circulation loop (The degree Celsius)
Voltage - Shows the voltage on the electric motor (Volt)
RateRMS - Represents the circulation flow rate of the fluid inside the loop (Liter per minute)
anomaly - Shows if the point is anomalous (0 or 1)
changepoint - Shows if the point is a changepoint for collective anomalies (0 or 1)


└── data                        # Data files and processing Jupyter Notebook
      ├── Load data.ipynb         # Jupyter Notebook to load all data
      ├── anomaly-free
      │   └── anomaly-free.csv     # Data obtained from the experiments with normal mode
      ├── valve2                  # Data obtained from the experiments with closing the valve at the outlet of the flow from the pump.
      │   ├── 1.csv
      │   ├── 2.csv
      │   ├── 3.csv
      │   └── 4.csv
      ├── valve1                  # Data obtained from the experiments with closing the valve at the flow inlet to the pump.
      │   ├── 1.csv
      │   ├── 2.csv
      │   ├── 3.csv
      │   ├── 4.csv
      │   ├── 5.csv
      │   ├── 6.csv
      │   ├── 7.csv
      │   ├── 8.csv
      │   ├── 9.csv
      │   ├── 10.csv
      │   ├── 11.csv
      │   ├── 12.csv
      │   ├── 12.csv
      │   ├── 13.csv
      │   ├── 14.csv
      │   ├── 15.csv
      │   └── 16.csv
      └── other                   # Data obtained from the other experiments
          ├── 1.csv               # Simulation of fluid leaks and fluid additions
          ├── 2.csv               # Simulation of fluid leaks and fluid additions
          ├── 3.csv               # Simulation of fluid leaks and fluid additions
          ├── 4.csv               # Simulation of fluid leaks and fluid additions
          ├── 5.csv               # Sharply behavior of rotor imbalance
          ├── 6.csv               # Linear behavior of rotor imbalance
          ├── 7.csv               # Step behavior of rotor imbalance
          ├── 8.csv               # Dirac delta function behavior of rotor imbalance
          ├── 9.csv               # Exponential behavior of rotor imbalance
          ├── 10.csv              # The slow increase in the amount of water in the circuit
          ├── 11.csv              # The sudden increase in the amount of water in the circuit
          ├── 12.csv              # Draining water from the tank until cavitation
          ├── 13.csv              # Two-phase flow supply to the pump inlet (cavitation)
          └── 14.csv              # Water supply of increased temperature

"""

import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from fontTools.unicodedata import block


def read_csv(path: str):

    with open(path, "r") as file:
        reader = csv.reader(file)  # Create a CSV reader object
        data = list(reader)

    return data


def construct_tagged_dataset(pths):

    # Define paths to save processed samples
    normal_save_path = "/Users/etayar/PycharmProjects/MultivariateTSDroneAD/skab_data/normal_data"
    anomalous_save_path = "/Users/etayar/PycharmProjects/MultivariateTSDroneAD/skab_data/anomalous_data"

    # Ensure directories exist
    os.makedirs(normal_save_path, exist_ok=True)
    os.makedirs(anomalous_save_path, exist_ok=True)

    # Determine the minimum dataset size
    min_size = min([len(pd.read_csv(p, delimiter=";")) for p in pths])
    sample_size = int(min_size * 0.5)  # Sample size is 50% of the smallest dataset

    print(f"Minimum dataset size: {min_size}, Sample size: {sample_size}")

    # Initialize separate lists for normal and anomalous samples
    normal_dataset = []
    anomalous_dataset = []

    for pth in pths:
        df = pd.read_csv(pth, delimiter=";")

        # Convert datetime column
        df["datetime"] = pd.to_datetime(df["datetime"])

        residual_df = pd.DataFrame()  # Store residual rows

        for i in range(0, len(df), sample_size):  # Step by sample_size (no overlap)
            sample_df = pd.concat([residual_df, df.iloc[i: i + sample_size]]).copy()  # Add residual rows to next sample

            if len(sample_df) < sample_size:
                residual_df = sample_df  # Save remaining rows for next iteration
                continue  # Skip incomplete samples for now

            residual_df = pd.DataFrame()  # Reset residuals after using them

            # Determine label (1 if any anomaly is present in the sample)
            label = 1 if sample_df["anomaly"].sum() > 0 else 0

            # Drop 'anomaly' and 'changepoint' columns before saving
            sample_df = sample_df.drop(columns=["anomaly", "changepoint"])

            # Save sample in the correct dataset
            if label == 0:
                normal_dataset.append(sample_df)
            else:
                anomalous_dataset.append(sample_df)

    # Save normal and anomalous samples as CSV files in separate directories
    for idx, sample_df in enumerate(normal_dataset):
        sample_df.to_csv(os.path.join(normal_save_path, f"sample_{idx}.csv"), index=False)

    for idx, sample_df in enumerate(anomalous_dataset):
        sample_df.to_csv(os.path.join(anomalous_save_path, f"sample_{idx}.csv"), index=False)

    print(f"Saved {len(normal_dataset)} normal samples and {len(anomalous_dataset)} anomalous samples.")

    return {'normal_dataset': normal_dataset, 'anomalous_dataset': anomalous_dataset}


if __name__ == '__main__':
    pths = [
        "/Users/etayar/PycharmProjects/MultivariateTSDroneAD/skab_data/other/1.csv",
        "/Users/etayar/PycharmProjects/MultivariateTSDroneAD/skab_data/other/2.csv",
        "/Users/etayar/PycharmProjects/MultivariateTSDroneAD/skab_data/other/3.csv",
        "/Users/etayar/PycharmProjects/MultivariateTSDroneAD/skab_data/other/4.csv",
        "/Users/etayar/PycharmProjects/MultivariateTSDroneAD/skab_data/other/5.csv",
        "/Users/etayar/PycharmProjects/MultivariateTSDroneAD/skab_data/other/6.csv",
        "/Users/etayar/PycharmProjects/MultivariateTSDroneAD/skab_data/other/7.csv",
        "/Users/etayar/PycharmProjects/MultivariateTSDroneAD/skab_data/other/8.csv",
        "/Users/etayar/PycharmProjects/MultivariateTSDroneAD/skab_data/other/9.csv",
        "/Users/etayar/PycharmProjects/MultivariateTSDroneAD/skab_data/other/10.csv",
        "/Users/etayar/PycharmProjects/MultivariateTSDroneAD/skab_data/other/11.csv",
        "/Users/etayar/PycharmProjects/MultivariateTSDroneAD/skab_data/other/12.csv",
        "/Users/etayar/PycharmProjects/MultivariateTSDroneAD/skab_data/other/13.csv",
        "/Users/etayar/PycharmProjects/MultivariateTSDroneAD/skab_data/other/14.csv"
    ]

    # ds = construct_tagged_dataset(pths)

    # Define paths to save processed samples
    normal_save_path = "/Users/etayar/PycharmProjects/MultivariateTSDroneAD/skab_data/normal_data"
    anomalous_save_path = "/Users/etayar/PycharmProjects/MultivariateTSDroneAD/skab_data/anomalous_data"

    # Ensure directories exist
    os.makedirs(normal_save_path, exist_ok=True)
    os.makedirs(anomalous_save_path, exist_ok=True)

    # Determine the minimum dataset size
    min_size = min([len(pd.read_csv(p, delimiter=";")) for p in pths])
    sample_size = int(min_size * 0.5)  # Sample size is 50% of the smallest dataset

    print(f"Minimum dataset size: {min_size}, Sample size: {sample_size}")

    # Initialize separate lists for normal and anomalous samples
    normal_dataset = []
    anomalous_dataset = []

    for pth in pths:
        df = pd.read_csv(pth, delimiter=";")

        # Convert datetime column
        df["datetime"] = pd.to_datetime(df["datetime"])

        residual_df = pd.DataFrame()  # Store residual rows

        for i in range(0, len(df), sample_size):  # Step by sample_size (no overlap)
            sample_df = pd.concat([residual_df, df.iloc[i: i + sample_size]]).copy()  # Add residual rows to next sample

            if len(sample_df) < sample_size:
                residual_df = sample_df  # Save remaining rows for next iteration
                continue  # Skip incomplete samples for now

            residual_df = pd.DataFrame()  # Reset residuals after using them

            # Determine label (1 if any anomaly is present in the sample)
            label = 1 if sample_df["anomaly"].sum() > 0 else 0

            # Drop 'anomaly' and 'changepoint' columns before saving
            # sample_df = sample_df.drop(columns=["anomaly", "changepoint"])

            # Save sample in the correct dataset
            if label == 0:
                normal_dataset.append(sample_df)
            else:
                anomalous_dataset.append(sample_df)

    ds = {'normal_dataset': normal_dataset, 'anomalous_dataset': anomalous_dataset}

    i = 0
    # Iterate over the dataset
    for k, v in ds.items():
        label = 1 if k == 'anomalous_dataset' else 0

        for df in v:
            plt.figure(figsize=(12, 5))

            # Plot two example columns: 'Accelerometer1RMS' and 'Voltage'
            plt.plot(df["datetime"], df["Accelerometer1RMS"], label="Accelerometer1RMS", alpha=0.7)
            plt.plot(df["datetime"], df["Voltage"], label="Voltage", alpha=0.7, linestyle="dashed")

            # Highlight anomalous points
            anomalous_points = df[df["anomaly"] == 1]  # Filter rows where anomaly == 1
            plt.scatter(anomalous_points["datetime"], anomalous_points["Accelerometer1RMS"],
                        color="red", label="Anomaly (Accel)", marker="o", zorder=3)
            plt.scatter(anomalous_points["datetime"], anomalous_points["Voltage"],
                        color="blue", label="Anomaly (Voltage)", marker="x", zorder=3)

            # Formatting
            plt.legend()
            plt.xlabel("Time")
            plt.ylabel("Sensor Readings")
            plt.title(f"Sample {i + 1} - Label: {'Anomalous' if label == 1 else 'Normal'}")
            plt.xticks(rotation=45)

            plt.show(block=True)
            plt.close()
            i += 1

    for pth in pths:
        # if int(pth.split('/')[-1].split('.')[0]) != 7:
        #     continue
        df = pd.read_csv(pth, delimiter=";")

        csv_content = read_csv(pth)

        # Convert datetime column
        df["datetime"] = pd.to_datetime(df["datetime"])

        # Plot data
        plt.figure(figsize=(12, 5))
        plt.plot(df["datetime"], df["Voltage"], label="Voltage", alpha=0.7)
        plt.scatter(df["datetime"][df["anomaly"] == 1], df["Voltage"][df["anomaly"] == 1], color="red", label="Anomalies",
                    marker="o")
        plt.scatter(df["datetime"][df["changepoint"] == 1], df["Voltage"][df["changepoint"] == 1], color="blue",
                    label="Changepoints", marker="x")
        plt.legend()
        plt.title(f"Anomalies & Changepoints in Voltage Data\n {pth.split('/')[-1]}")
        plt.show(block=True)
        plt.close()

    exit()