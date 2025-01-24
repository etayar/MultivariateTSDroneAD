"""
This will contain the main training logic.
We define a Trainer class that handles the full training loop, including:
    * Loading data
    * Model training and validation
    * Saving the model
    * Logging metrics
"""

import json
import torch
from metrics import compute_metrics
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from callbacks import EarlyStopping, MetricLogger

class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler=None, patience=5, save_path="best_model.pth"):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.early_stopping = EarlyStopping(patience=patience, save_path=save_path)
        self.logger = MetricLogger()

    def train_one_epoch(self, dataloader, device):
        self.model.train()
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

    def train(self, train_loader, val_loader, device, epochs):
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            self.train_one_epoch(train_loader, device)

            # Evaluate on validation set
            val_loss, precision, recall, f1_score = self.evaluate(val_loader, device)

            # Log metrics
            self.logger.log(epoch + 1, val_loss, precision, recall, f1_score)

            # Early stopping
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping triggered. Stopping training.")
                break

            # Step the scheduler, if provided
            if self.scheduler:
                self.scheduler.step()

        # Save metrics
        self.logger.save()

    def evaluate(self, dataloader, device):
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self.model(inputs)
                total_loss += self.criterion(outputs, labels).item()

                predictions = (outputs > 0.5).float()
                all_labels.append(labels.cpu())
                all_predictions.append(predictions.cpu())

        # Metrics
        all_labels = torch.cat(all_labels).numpy()
        all_predictions = torch.cat(all_predictions).numpy()

        avg_loss = total_loss / len(dataloader)
        precision = precision_score(all_labels, all_predictions, zero_division=0)
        recall = recall_score(all_labels, all_predictions, zero_division=0)
        f1 = f1_score(all_labels, all_predictions, zero_division=0)

        print(f"Validation Loss: {avg_loss}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
        return avg_loss, precision, recall, f1
