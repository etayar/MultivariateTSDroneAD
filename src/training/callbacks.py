"""
Contains callback classes for training.
For example:
    * EarlyStopping: Stops training if the validation loss stops improving for a given number of epochs.
    * ModelCheckpoint: Saves the model weights at certain intervals or based on validation performance.
"""

import torch
import json

class EarlyStopping:
    """
    Stop training if validation loss does not improve after a specified patience.
    """
    def __init__(self, patience=5, verbose=True, save_path="best_model.pth"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.save_path = save_path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            # Save the best model
            torch.save(model.state_dict(), self.save_path)
            if self.verbose:
                print(f"Validation loss improved to {val_loss}. Saving model to {self.save_path}.")
        else:
            self.counter += 1
            if self.verbose:
                print(f"Validation loss did not improve. Early stopping counter: {self.counter}/{self.patience}.")
            if self.counter >= self.patience:
                self.early_stop = True


class MetricLogger:
    """
    Log metrics (loss, precision, recall, F1-score) to a JSON file for later analysis.
    """
    def __init__(self, save_path="training_metrics.json"):
        self.save_path = save_path
        self.metrics = {"epoch": [], "val_loss": [], "precision": [], "recall": [], "f1_score": []}

    def log(self, epoch, val_loss, precision, recall, f1_score):
        self.metrics["epoch"].append(epoch)
        self.metrics["val_loss"].append(val_loss)
        self.metrics["precision"].append(precision)
        self.metrics["recall"].append(recall)
        self.metrics["f1_score"].append(f1_score)

    def save(self):
        with open(self.save_path, "w") as f:
            json.dump(self.metrics, f, indent=4)
        print(f"Metrics saved to {self.save_path}.")
