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


class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler=None, save_path="best_model.pth"):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.save_path = save_path
        self.best_val_loss = float("inf")  # Track the best validation loss
        self.best_model_state = None      # Track the model's state with the best validation loss
        self.metrics_history = []         # Store metrics for all epochs

    def train_one_epoch(self, dataloader, device):
        """
        Train the model for one epoch.
        """
        self.model.train()
        running_loss = 0

        for batch in tqdm(dataloader, desc="Training Batches"):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Training Loss: {avg_loss}")

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

                # Convert probabilities to binary predictions
                predictions = (outputs > 0.5).float()

                # Collect predictions and labels for metrics computation
                all_labels.append(labels.cpu())
                all_predictions.append(predictions.cpu())

        # Concatenate all batches for metrics calculation
        all_labels = torch.cat(all_labels).numpy()
        all_predictions = torch.cat(all_predictions).numpy()

        # Compute metrics
        avg_loss = total_loss / len(dataloader)
        precision = precision_score(all_labels, all_predictions, zero_division=0)
        recall = recall_score(all_labels, all_predictions, zero_division=0)
        f1 = f1_score(all_labels, all_predictions, zero_division=0)

        print(f"Validation Loss: {avg_loss}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
        return avg_loss, precision, recall, f1


    def train(self, train_loader, val_loader, device, epochs):
        """
        Main training loop for multiple epochs.
        """
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Train one epoch
            self.train_one_epoch(train_loader, device)

            # Evaluate on the validation set
            val_loss, val_metrics = self.evaluate(val_loader, device)
            print(f"Validation Loss: {val_loss}, Metrics: {val_metrics}")

            # Save metrics for this epoch
            self.metrics_history.append({
                "epoch": epoch + 1,
                "val_loss": val_loss,
                **val_metrics
            })

            # Check for improvement in validation loss
            if val_loss < self.best_val_loss:
                print(f"Validation loss improved from {self.best_val_loss} to {val_loss}")
                self.best_val_loss = val_loss
                self.best_model_state = {
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                }

            # Step the scheduler, if provided
            if self.scheduler:
                self.scheduler.step()

        # Save the best model after all epochs are done
        if self.best_model_state is not None:
            print(f"Saving the best model with validation loss {self.best_val_loss}...")
            torch.save(self.best_model_state, self.save_path)

        # Save metrics history to a file
        metrics_file = "training_metrics.json"
        print(f"Saving metrics history to {metrics_file}...")
        with open(metrics_file, "w") as f:
            json.dump(self.metrics_history, f, indent=4)
