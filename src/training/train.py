"""
This will contain the main training logic.
We define a Trainer class that handles the full training loop, including:
    * Loading data
    * Model training and validation
    * Saving the model
    * Logging metrics
"""
import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score


class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler=None, save_path="best_model.pth"):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.save_path = save_path
        self.best_val_loss = float("inf")  # To track the best validation loss
        self.metrics_history = []  # Store metrics for all epochs

    def train_one_epoch(self, dataloader, device):
        """
        Train the model for one epoch.
        """
        self.model.train()
        running_loss = 0

        for batch in tqdm(dataloader, desc="Training Batches"):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            if isinstance(self.criterion, torch.nn.BCELoss):
                labels = labels.unsqueeze(1)  # Only apply for BCE loss in binary classification.

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Training Loss: {avg_loss}")
        return avg_loss

    def evaluate(self, dataloader, device):
        """
        Evaluate the model on the validation set and compute metrics.
        # TODO: Implement AUC-ROC calculations for imbalanced data set.
        """
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating Batches"):
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                if isinstance(self.criterion, torch.nn.BCELoss):
                    labels = labels.unsqueeze(1)  # Only apply for BCE loss in binary classification.

                outputs = self.model(inputs)
                total_loss += self.criterion(outputs, labels).item()

                predictions = (outputs > 0.5).float()
                all_labels.append(labels.cpu())
                all_predictions.append(predictions.cpu())

        # Metrics computation
        all_labels = torch.cat(all_labels).numpy()
        all_predictions = torch.cat(all_predictions).numpy()

        avg_loss = total_loss / len(dataloader)
        precision = precision_score(all_labels, all_predictions, zero_division=0)
        recall = recall_score(all_labels, all_predictions, zero_division=0)
        f1 = f1_score(all_labels, all_predictions, zero_division=0)

        print(f"Validation Loss: {avg_loss}, Precision: {precision:.4f}, "
              f"Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        return avg_loss, {"precision": precision, "recall": recall, "f1": f1}

    def train(self, train_loader, val_loader, device, epochs, config, start_epoch=0):
        """
        Main training loop for multiple epochs.
        """
        for epoch in range(start_epoch, epochs):  # Start from the given epoch
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Train one epoch
            train_loss = self.train_one_epoch(train_loader, device)

            # Evaluate on validation set
            val_loss, val_metrics = self.evaluate(val_loader, device)

            # Save metrics for this epoch
            self.metrics_history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **val_metrics
            })

            # Save the latest checkpoint
            self.save_checkpoint(epoch + 1, config, val_loss, checkpoint_path="checkpoint_epoch.pth")

            # Save the best model if validation loss improves
            if val_loss < self.best_val_loss:
                print(f"Validation loss improved from {self.best_val_loss} to {val_loss}. Saving best model...")
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch + 1, config, val_loss, checkpoint_path="best_model.pth")

            # Step the scheduler, if provided
            if self.scheduler:
                self.scheduler.step()

    def save_model_with_config(self, config):
        """
        Save the model's state and configuration.
        """
        save_data = {
            "model_state_dict": self.model.state_dict(),
            "config": config,
        }
        torch.save(save_data, self.save_path)
        print(f"Model and configuration saved to {self.save_path}")

    def save_checkpoint(self, epoch, config, val_loss, checkpoint_path="checkpoint.pth"):
        """
        Save a checkpoint with model state, optimizer state, scheduler state, and other info.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "config": config,
            "val_loss": val_loss,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

