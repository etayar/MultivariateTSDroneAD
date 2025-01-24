import os
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

    # Path to the UAV dataset
    data_path = "src/uav_data"  # Update this to match your dataset location
    label_column = "label"  # Update this to the name of your label column

    # Load and split data into DataLoaders
    train_loader, val_loader, test_loader = load_and_split_data(
        data_path=data_path,
        label_column=label_column,
        batch_size=32
    )

    # Dynamically infer the input shape
    sample_batch = next(iter(train_loader))
    inputs, _ = sample_batch
    input_shape = inputs.shape[1:]

    # Define model, optimizer, and criterion
    model = build_model(input_shape, fuser_name="ConvFuser1", transformer_variant="vanilla")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()

    # Define save paths
    save_path = os.path.join("models", "best_model.pth")
    metrics_path = os.path.join("models", "training_metrics.json")

    # Create Trainer with callbacks
    trainer = Trainer(model, optimizer, criterion, save_path=save_path, patience=5)

    # Train
    trainer.train(train_loader, val_loader, device, epochs=10)

    # Load the best model
    model.load_state_dict(torch.load(save_path))

    # Evaluate on the test set
    test_loss, precision, recall, f1_score = trainer.evaluate(test_loader, device)
    print(f"Test Set Results - Loss: {test_loss}, Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")


if __name__ == "__main__":
    main()
