from src.training.train import Trainer
from src.multivariate_univariate_fusion_anomaly_detection import build_model
from torch.utils.data import DataLoader
import torch


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model, optimizer, and criterion
    S, T = 64, 640
    input_shape = (S, T)
    model = build_model(input_shape, fuser_name="ConvFuser1", transformer_variant="vanilla")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()  # Binary cross-entropy loss

    # Dataloaders (replace with real datasets)
    train_loader = DataLoader(...)
    val_loader = DataLoader(...)

    # Initialize the Trainer
    trainer = Trainer(model, optimizer, criterion)

    # Train for multiple epochs
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        trainer.train_one_epoch(train_loader, device)
        val_loss = trainer.evaluate(val_loader, device)
        print(f"Validation Loss: {val_loss}")


if __name__ == "__main__":
    main()
