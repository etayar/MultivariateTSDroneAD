import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import json
from datetime import datetime
import torch
from networkx import config


def load_config_and_training_results(date: str, data_set: str):
    if "COLAB_GPU" in os.environ:
        base_path = f'/content/drive/My Drive/My_PHD/My_First_Paper/models_metrics/{date}/{data_set}'
    else:
        base_path = f'/Users/etayar/PycharmProjects/MultivariateTSDroneAD/src/data/models_metrics/{date}/{data_set}'

    checkpoint_path = os.path.join(base_path, 'best_model.pth')
    training_path = os.path.join(base_path, 'training.json')
    test_path = os.path.join(base_path, 'test.json')

    # Load the saved model checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")  # Load on CPU to avoid device issues
    config = checkpoint["config"]

    with open(training_path, "r") as f:
        training_results = json.load(f)

    with open(test_path, "r") as f:
        test_results = json.load(f)

    return {'config': config, 'training_results': training_results, 'test_results': test_results}


def metrics_visualisation(root_pth: str, specific_date: str = None):

    # Get all folders in the directory
    directory_path = Path(root_pth)
    folders = [f.resolve() for f in directory_path.iterdir() if f.is_dir() and f.name == specific_date] \
        if specific_date else [f.name for f in directory_path.iterdir() if f.is_dir()]

    files = [ff.name + '/' + f.name for ff in folders for f in ff.iterdir() if f.is_file() and f.name.split('.')[-1] == 'json']

    for file in files:

        if file.split('/')[-1].split('.')[0] == 'test':
            continue

        df = pd.read_json(root_pth + '/' + file)
        print(df.columns)

        loss = ['train_loss', 'val_loss']
        acc_scores = ['train_accuracy (auc_roc)', 'auc_roc' ]

        # Set figure size
        plt.figure(figsize=(12, 6))

        # Plot each loss field against "epoch"
        for column in loss:
            plt.plot(df["epoch"], df[column], label=column)

        # Add labels and title
        plt.xlabel("Epoch")
        plt.ylabel("Values")
        plt.title("Training and Validation loss Epochs")
        plt.legend()
        plt.grid()

        plt.figure(figsize=(12, 6))

        # Plot accuracy fields against "epoch"
        for column in acc_scores:
            plt.plot(df["epoch"], df[column], label=column)

        # Add labels and title
        plt.xlabel("Epoch")
        plt.ylabel("Values")
        plt.title("Training and Validation accuracy metrics")
        plt.legend()
        plt.grid()

        # Show the plot
        plt.show(block=True)


if __name__ == '__main__':
    if "COLAB_GPU" in os.environ:
        root_path = "/content/drive/My Drive/My_PHD/My_First_Paper/models_metrics"
    else:
        root_path = "/Users/etayar/PycharmProjects/MultivariateTSDroneAD/src/data/models_metrics"

    current_date = datetime.now().strftime("%Y-%m-%d")
    metrics_visualisation(root_path, specific_date=current_date)

    exit()