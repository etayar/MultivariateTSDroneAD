import json
from torch.utils.data import DataLoader
from src.data.data_loader import load_and_split_data
from src.training.train import Trainer
from src.multivariate_univariate_fusion_anomaly_detection import build_model
import torch
from torch.utils.data import DataLoader


def save_metrics(metrics_history, metrics_file = "training_metrics/training.json"):

    print(f"Saving metrics history to {metrics_file}...")
    with open(metrics_file, "w") as f:
        json.dump(metrics_history, f, indent=4)


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Path to the UAV dataset
    data_path = "src/uav_data"  # Path to your dataset
    label_column = "label"  # Column name containing labels in the CSV files

    # Load data and create DataLoaders
    train_loader, val_loader, test_loader = load_and_split_data(
        data_path=data_path,
        label_column=label_column,
        batch_size=32,
        random_state=42
    )

    # Dynamically infer input shape from a batch
    sample_batch = next(iter(train_loader))
    inputs, _ = sample_batch  # Unpack inputs and labels
    input_shape = inputs.shape[1:]  # Exclude batch size

    # Build model
    model = build_model(input_shape, fuser_name="ConvFuser1", transformer_variant="vanilla")
    model.to(device)

    # Define optimizer, loss, and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()  # Binary Cross-Entropy Loss
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Initialize Trainer
    trainer = Trainer(model, optimizer, criterion, scheduler=scheduler)

    # Train the model
    num_epochs = 20
    trainer.train(train_loader, val_loader, device, epochs=num_epochs)
    metrics_history = trainer.metrics_history
    save_metrics(metrics_history)

    # Evaluate on the test set
    test_loss, test_metrics = trainer.evaluate(test_loader, device)
    print(f"Test Loss: {test_loss}, Test Metrics: {test_metrics}")
    save_metrics(test_metrics, "training_metrics/test.json")



if __name__ == "__main__":
    main()
