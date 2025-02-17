import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def metrics_visualisation(root_pth: str, specific_date: str = None):

    # Get all folders in the directory
    directory_path = Path(root_pth)
    folders = [f.resolve() for f in directory_path.iterdir() if f.is_dir() and f.name == specific_date] \
        if specific_date else [f.name for f in directory_path.iterdir() if f.is_dir()]

    files = [ff.name + '/' + f.name for ff in folders for f in ff.iterdir() if f.is_file() and f.name.split('.')[-1] == 'json']

    for file in files:

        # if file.split('/')[-1].split('.')[0] == 'test':
        #     continue

        df = pd.read_json(root_pth + '/' + file)
        print(df.columns)

        fields_to_inspect = ['train_loss', 'train_accuracy', 'val_loss', 'auc_roc']

        # Set figure size
        plt.figure(figsize=(12, 6))

        # Plot each field against "epoch"
        for column in fields_to_inspect:
            plt.plot(df["epoch"], df[column], label=column)

        # Add labels and title
        plt.xlabel("Epoch")
        plt.ylabel("Values")
        plt.title("Training and Validation Metrics Over Epochs")
        plt.legend()
        plt.grid()

        # Show the plot
        plt.show(block=True)


if __name__ == '__main__':
    root_path = "/Users/etayar/PycharmProjects/MultivariateTSDroneAD/src/data/models_metrics"

    metrics_visualisation(root_path, specific_date='2025-02-17')

    exit()