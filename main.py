from src.training.train import Trainer
from src.multivariate_univariate_fusion_anomaly_detection import build_model
from torch.utils.data import DataLoader
import torch


def save_metrics(self, metrics, filename="training_metrics.json"):
    with open(filename, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {filename}")


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloaders (replace with real datasets)
    train_loader = DataLoader(...)  # Replace with real training DataLoader
    val_loader = DataLoader(...)  # Replace with real validation DataLoader

    # Dynamically infer the input shape from a batch
    sample_batch = next(iter(train_loader))  # Get one batch
    inputs, _ = sample_batch  # Unpack inputs and labels
    input_shape = inputs.shape[1:]  # Exclude batch size (e.g., [S, T])

    # Define model, optimizer, and criterion
    model = build_model(input_shape, fuser_name="ConvFuser1", transformer_variant="vanilla")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()  # Binary cross-entropy loss

    # Learning rate scheduler (optional)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Initialize the Trainer
    save_path = os.path.join("models", "best_model.pth")  # Save best model in a directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists
    trainer = Trainer(model, optimizer, criterion, scheduler, save_path)

    # Train for multiple epochs and save metrics
    num_epochs = 10
    metrics = {"epoch": [], "val_loss": [], "precision": [], "recall": [], "f1_score": []}

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        trainer.train_one_epoch(train_loader, device)

        # Evaluate on validation set
        val_loss, precision, recall, f1 = trainer.evaluate(val_loader, device)

        # Store metrics
        metrics["epoch"].append(epoch + 1)
        metrics["val_loss"].append(val_loss)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["f1_score"].append(f1)

    # Save metrics after training
    metrics_path = os.path.join("models", "training_metrics.json")
    save_metrics(metrics, filename=metrics_path)

    print("Training complete!")
    print(f"Best model saved at {save_path}.")
    print(f"Metrics saved at {metrics_path}.")


if __name__ == "__main__":
    main()
