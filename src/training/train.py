"""
This will contain the main training logic.
We define a Trainer class that handles the full training loop, including:
    * Loading data
    * Model training and validation
    * Saving the model
    * Logging metrics
"""
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score


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
    def __init__(
            self,
            model,
            optimizer,
            criterion,
            scheduler=None,
            is_binary=True,
            is_multi_label=False,
            is_multi_class=False,
            num_classes=2,
            prediction_threshold=0.5,
            ema_alpha=0.1,
            window_s=5
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.is_multi_label = is_multi_label
        self.is_multi_class = is_multi_class
        self.is_binary = is_binary
        self.num_classes = num_classes
        self.prediction_threshold = prediction_threshold
        self.best_val_loss = float("inf")  # To track the best validation loss
        self.metrics_history = []  # Store metrics for all epochs
        self.val_loss = []
        self.EMA = []
        self.ema_alpha = ema_alpha
        self.window_s = window_s

        if is_binary and is_multi_label:
            raise ValueError("A model cannot be both binary and multi-label. Check configuration.")

    def train_one_epoch(self, dataloader, device):
        """
        Train the model for one epoch and return loss along with predictions for metrics.
        """
        self.model.train()
        running_loss = 0
        all_labels = []
        all_predictions = []
        all_probs = []

        for batch in tqdm(dataloader, desc="Training Batches"):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            if self.is_binary:
                # Binary classification: BCEWithLogitsLoss expects (batch_size, 1)
                labels = labels.view(-1, 1) if labels.dim() == 1 else labels
                labels = labels.float()  # Convert to float for BCE loss
            elif self.is_multi_label:
                # Multilabel classification: BCEWithLogitsLoss expects (batch_size, num_classes)
                labels = labels.view(-1, self.num_classes) if labels.dim() == 1 else labels
                labels = labels.float()  # Ensure floating-point labels for BCE loss
            elif self.is_multi_class:
                # Multiclass classification: CrossEntropyLoss expects (batch_size,)
                labels = labels.long()  # Ensure integer class labels for CrossEntropyLoss

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)  # Compute loss
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            # Convert outputs to probabilities and predictions
            if self.is_binary or self.is_multi_label:
                probs = torch.sigmoid(outputs)
                predictions = (probs > self.prediction_threshold).float()
            else:
                probs = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probs, dim=1)

            all_labels.append(labels.cpu())
            all_predictions.append(predictions.cpu())
            all_probs.append(probs.detach().cpu())  # Detach required to remove computation graph

        avg_loss = running_loss / len(dataloader)

        return avg_loss, all_labels, all_predictions, all_probs

    def evaluate(self, dataloader, device):
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
            for batch in dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                if self.is_binary:
                    # Binary classification: BCEWithLogitsLoss expects (batch_size, 1)
                    labels = labels.view(-1, 1) if labels.dim() == 1 else labels
                    labels = labels.float()  # Convert to float for BCE loss
                elif self.is_multi_label:
                    # Multilabel classification: BCEWithLogitsLoss expects (batch_size, num_classes)
                    labels = labels.view(-1, self.num_classes) if labels.dim() == 1 else labels
                    labels = labels.float()  # Ensure floating-point labels for BCE loss
                elif self.is_multi_class:
                    # Multiclass classification: CrossEntropyLoss expects (batch_size,)
                    labels = labels.long()  # Ensure integer class labels for CrossEntropyLoss

                outputs = self.model(inputs)
                total_loss += self.criterion(outputs, labels).item()

                # Convert outputs to probabilities
                if self.is_binary or self.is_multi_label:
                    probs = torch.sigmoid(outputs)  # Sigmoid for binary & multi-label
                    predictions = (probs > self.prediction_threshold).float()
                else:
                    probs = torch.softmax(outputs, dim=1)  # Softmax for multi-class
                    predictions = torch.argmax(probs, dim=1)

                all_labels.append(labels.cpu())
                all_predictions.append(predictions.cpu())
                all_probs.append(probs.cpu())

        # Convert lists to NumPy arrays
        all_labels = torch.cat(all_labels).numpy()
        all_predictions = torch.cat(all_predictions).numpy()
        all_probs = torch.cat(all_probs).numpy()

        avg_loss = total_loss / len(dataloader)

        # Choose correct averaging strategy
        if self.is_multi_label:
            avg_strategy = "samples"
        elif self.is_binary:
            avg_strategy = "binary"
        else:
            avg_strategy = "weighted"

        # Compute evaluation metrics
        precision = precision_score(all_labels, all_predictions, average=avg_strategy, zero_division=0)
        recall = recall_score(all_labels, all_predictions, average=avg_strategy, zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average=avg_strategy, zero_division=0)

        # Compute AUC-ROC
        if self.is_binary:
            auc_roc = roc_auc_score(all_labels, all_probs)
        elif self.is_multi_label:
            auc_roc = roc_auc_score(all_labels, all_probs, average="macro")
        else:
            auc_roc = roc_auc_score(all_labels, all_probs, multi_class="ovr")

        print(f"Validation Loss: {avg_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
              f"F1 Score: {f1:.4f}, AUC-ROC: {auc_roc:.4f}")

        return avg_loss, {"precision": precision, "recall": recall, "f1": f1, "auc_roc": auc_roc}

    def train(self, train_loader, val_loader, device, epochs, config, start_epoch=0, patience=5):
        """
        Main training loop with early stopping and training accuracy, precision, and recall monitoring.
        """
        early_stopping = EarlyStopping(patience=patience)

        for epoch in range(start_epoch, epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Train one epoch
            train_loss, all_labels, all_predictions, all_probs = self.train_one_epoch(train_loader, device)

            # Compute Training Metrics
            all_labels = torch.cat(all_labels).numpy()
            all_predictions = torch.cat(all_predictions).numpy()
            all_probs = torch.cat(all_probs).numpy()

            # accuracy = accuracy_score(all_labels, all_predictions)
            precision = precision_score(all_labels, all_predictions, average="binary" if self.is_binary else "weighted",
                                        zero_division=0)
            recall = recall_score(all_labels, all_predictions, average="binary" if self.is_binary else "weighted",
                                  zero_division=0)

            # Compute AUC-ROC
            if self.is_binary:
                auc_roc = roc_auc_score(all_labels, all_probs)
            elif self.is_multi_label:
                auc_roc = roc_auc_score(all_labels, all_probs, average="macro")
            else:
                auc_roc = roc_auc_score(all_labels, all_probs, multi_class="ovr")

            # Train monitoring
            print(
                f"Training Loss: {train_loss:.4f}, "
                f"Accuracy (auc_roc): {auc_roc:.4f}, "
                f"Precision: {precision:.4f}, "
                f"Recall: {recall:.4f}"
            )

            # Evaluate on validation set
            val_loss, val_metrics = self.evaluate(val_loader, device)
            self.val_loss.append(val_loss)
            if epoch < self.window_s - 1:
                rolling_avg = np.mean(self.val_loss)
            else:
                rolling_avg = np.mean(self.val_loss[-self.window_s:])

            if epoch == 0:
                ema_t = val_loss
            else:
                bias_correction = 1 - (1 - self.ema_alpha) ** (epoch + 1)
                ema_t = (self.ema_alpha * val_loss + (1 - self.ema_alpha) * self.EMA[epoch - 1]) / bias_correction
            self.EMA.append(ema_t)

            # Save metrics for this epoch
            self.metrics_history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy (auc_roc)": auc_roc,
                "train_precision": precision,
                "train_recall": recall,
                "val_loss": val_loss,
                "rolling_avg": rolling_avg,
                "EMA": ema_t,
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
                print(f"Early stopping triggered at epoch {epoch + 1}.")
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


