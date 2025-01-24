"""
This will contain the main training logic.
We define a Trainer class that handles the full training loop, including:
    * Loading data
    * Model training and validation
    * Saving the model
    * Logging metrics
"""
import torch


class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler=None, save_path="best_model.pth"):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.save_path = save_path
        self.best_val_loss = float("inf")  # Track the best validation loss
        self.best_model_state = None      # Track the model's state with the best validation loss

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

    def evaluate(self, dataloader, device):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                total_loss += self.criterion(outputs, labels).item()
        return total_loss / len(dataloader)

    def train(self, train_loader, val_loader, device, epochs):
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            self.train_one_epoch(train_loader, device)

            # Evaluate on the validation set
            val_loss = self.evaluate(val_loader, device)
            print(f"Validation Loss: {val_loss}")

            # Check for improvement in validation loss
            if val_loss < self.best_val_loss:
                print(f"Validation loss improved from {self.best_val_loss} to {val_loss}")
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict()  # Save the best model state

            # Step the scheduler, if provided
            if self.scheduler:
                self.scheduler.step()

        # Save the best model after all epochs are done
        if self.best_model_state is not None:
            print(f"Saving the best model with validation loss {self.best_val_loss}...")
            torch.save(self.best_model_state, self.save_path)
