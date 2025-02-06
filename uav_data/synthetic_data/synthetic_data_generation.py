import numpy as np
import pandas as pd


def generate_synthetic_data(original_df, n_files=5, n_rows=None, output_dir="/Users/etayar/PycharmProjects/MultivariateTSDroneAD/uav_data/synthetic_data"):
    """
    Generates multiple synthetic CSV files based on the statistical distribution of the original dataset.

    Args:
        original_df (pd.DataFrame): The reference dataset to base the synthetic data on.
        n_files (int): Number of synthetic data files to generate.
        n_rows (int, optional): Number of rows per synthetic file (default: same as original dataset).
        output_dir (str): Directory to save the generated synthetic files.

    Returns:
        List of file paths for the generated synthetic datasets.
    """
    import os

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Determine number of rows per file
    if n_rows is None:
        n_rows = len(original_df)

    # Separate columns by type
    num_cols = original_df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = original_df.select_dtypes(include=['object']).columns

    # Compute mean & std for numerical columns
    num_stats = original_df[num_cols].agg(['mean', 'std'])

    # Collect generated file paths
    generated_files = []

    for i in range(n_files):
        # Generate synthetic numerical data at once (fixes warning)
        num_data = np.random.normal(
            loc=num_stats.loc['mean'],
            scale=num_stats.loc['std'],
            size=(n_rows, len(num_cols))
        )

        num_df = pd.DataFrame(num_data, columns=num_cols).astype(original_df[num_cols].dtypes)

        # Generate synthetic categorical data at once
        cat_data = {col: np.random.choice(original_df[col].dropna().unique(), size=n_rows) for col in cat_cols}
        cat_df = pd.DataFrame(cat_data)

        # Combine numerical and categorical data
        synthetic_data = pd.concat([num_df, cat_df], axis=1)

        # Save the synthetic dataset
        file_name = f"synthetic_flight_{i+1}.csv"
        file_path = os.path.join(output_dir, file_name)
        synthetic_data.to_csv(file_path, index=False)
        generated_files.append(file_path)

    return generated_files


if __name__ == '__main__':

    # Data sample to for analysis to generate synthetic data with the same distribution
    df = pd.read_csv("/Users/etayar/PycharmProjects/MultivariateTSDroneAD/uav_data/boaz_csv_flight_data/boaz_flight_1.01.csv")

    # synthetic_data paths
    anomalous_output_path = "/Users/etayar/PycharmProjects/MultivariateTSDroneAD/uav_data/synthetic_data/anomalous_data"
    normal_output_path = "/Users/etayar/PycharmProjects/MultivariateTSDroneAD/uav_data/synthetic_data/normal_data"

    for k, output_path in {'Anomalous_output_path': anomalous_output_path, 'Normal_output_path': normal_output_path}.items():
        synthetic_files = generate_synthetic_data(df, n_files=50, output_dir=output_path)

        print(f"{k.split('_')[0]} data: {synthetic_files}")

    exit()
