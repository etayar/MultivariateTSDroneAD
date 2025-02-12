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
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


class EarlyStopping:
    def __init__(self, patience=5, es_threshold=1e-3):
        """
        Early stops training if validation loss doesn't improve after a given patience.
        Args:
            patience (int): How many epochs to wait before stopping if no improvement.
            es_threshold (float): Minimum change in validation loss to qualify as an improvement.
        """
        self.patience = patience
        self.es_threshold = es_threshold
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss):
        """
        Call this function after each epoch to check if training should stop.
        Returns True if training should stop, False otherwise.
        """

        if val_loss < self.best_loss - self.es_threshold:
            self.best_loss = val_loss
            self.counter = 0  # Reset counter if loss improves
        elif val_loss < self.best_loss:
            self.best_loss = val_loss  # Update best loss but don't reset patience if improvement is small
        else:
            self.counter += 1  # Increase counter if no improvement

        return self.counter >= self.patience  # Stop if patience threshold is reached


class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
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

            # Ensure labels have the correct shape for BCE-based losses
            is_binary = isinstance(self.criterion, torch.nn.BCELoss)
            is_multi_label = isinstance(self.criterion, torch.nn.BCEWithLogitsLoss)

            if is_binary or is_multi_label:
                labels = labels.float()  # Ensure labels are float for BCE losses
                if is_binary:
                    labels = labels.unsqueeze(1)  # Reshape for binary classification

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)  # Compute loss
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Training Loss: {avg_loss}")
        return avg_loss

    def evaluate(self, dataloader, device, prediction_threshold=0.5):
        """
        Evaluate the model on the validation set and compute metrics, including AUC-ROC.
        Handles binary, multi-class, and multi-label classification.
        """
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_predictions = []
        all_probs = []  # Store probabilities for AUC-ROC

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating Batches"):
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                is_binary = isinstance(self.criterion, torch.nn.BCELoss)
                is_multi_label = isinstance(self.criterion, torch.nn.BCEWithLogitsLoss)

                if is_binary or is_multi_label:
                    labels = labels.float()  # Ensure float type for BCE-based losses
                    if is_binary:
                        labels = labels.unsqueeze(1)  # Reshape for BCE Loss

                outputs = self.model(inputs)
                total_loss += self.criterion(outputs, labels).item()

                # Convert outputs to probabilities
                if is_binary or is_multi_label:
                    probs = torch.sigmoid(outputs)  # Sigmoid for binary & multi-label
                    predictions = (probs > prediction_threshold).float()  # Dynamic threshold for predictions
                else:
                    probs = torch.softmax(outputs, dim=1)  # Softmax for multi-class
                    predictions = torch.argmax(probs, dim=1)  # Multi-class classification

                all_labels.append(labels.cpu())
                all_predictions.append(predictions.cpu())
                all_probs.append(probs.cpu())

        # Convert lists to NumPy arrays
        all_labels = torch.cat(all_labels).numpy()
        all_predictions = torch.cat(all_predictions).numpy()
        all_probs = torch.cat(all_probs).numpy()

        avg_loss = total_loss / len(dataloader)

        # Compute evaluation metrics
        if is_multi_label:
            precision = precision_score(all_labels, all_predictions, average="samples", zero_division=0)
            recall = recall_score(all_labels, all_predictions, average="samples", zero_division=0)
            f1 = f1_score(all_labels, all_predictions, average="samples", zero_division=0)
            auc_roc = roc_auc_score(all_labels, all_probs, average="macro")
        elif is_binary:
            precision = precision_score(all_labels, all_predictions, average="binary", zero_division=0)
            recall = recall_score(all_labels, all_predictions, average="binary", zero_division=0)
            f1 = f1_score(all_labels, all_predictions, average="binary", zero_division=0)
            auc_roc = roc_auc_score(all_labels, all_probs)
        else:
            precision = precision_score(all_labels, all_predictions, average="weighted", zero_division=0)
            recall = recall_score(all_labels, all_predictions, average="weighted", zero_division=0)
            f1 = f1_score(all_labels, all_predictions, average="weighted", zero_division=0)
            auc_roc = roc_auc_score(all_labels, all_probs, multi_class="ovr")

        print(f"Validation Loss: {avg_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
              f"F1 Score: {f1:.4f}, AUC-ROC: {auc_roc:.4f}")

        return avg_loss, {"precision": precision, "recall": recall, "f1": f1, "auc_roc": auc_roc}

    def train(self, train_loader, val_loader, device, epochs, config, start_epoch=0, patience=5):
        """
        Main training loop with early stopping.
        """
        early_stopping = EarlyStopping(patience=patience)

        for epoch in range(start_epoch, epochs):
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

            # Save latest checkpoint
            self.save_model_with_config(
                epoch + 1, config, val_loss, path=config['checkpoint_epoch_path']
            )

            # Save the best model if validation loss improves
            if val_loss < self.best_val_loss:
                print(f"Validation loss improved from {self.best_val_loss} to {val_loss}. Saving best model...")
                self.best_val_loss = val_loss
                self.save_model_with_config(
                    epoch + 1, config, val_loss, path=config['best_model_path']
                )

            # Step the scheduler (if provided)
            if self.scheduler:
                self.scheduler.step(val_loss)  # Adjust LR based on validation loss

            # **Check for Early Stopping**
            if early_stopping.step(val_loss):
                print(f"Early stopping triggered at epoch {epoch + 1}. Training stopped.")
                # break  # Stop training loop  <--- Turned off, only save the model.
                self.save_model_with_config(
                    epoch + 1,
                    config,
                    val_loss,
                    path=config['best_model_path'],
                    early_stopping_triggered=True
                )

    def save_model_with_config(
            self, epoch, config, val_loss, path, save_usage='best_model', early_stopping_triggered=False
    ):
        """
        Save a checkpoint with model state, optimizer state, scheduler state, and other info.
        """
        model_config = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "config": config,
            "val_loss": val_loss,
            "early_stopping_triggered": early_stopping_triggered
        }
        torch.save(model_config, path)

        save_type = 'Best model' if save_usage == 'best_model' else 'Checkpoint'
        print(f"{save_type} saved to {path}")


