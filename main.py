import json
from src.data.data_loader import load_and_split_time_series_data
from src.training.train import Trainer
from src.multivariate_fusion_anomaly_detection import build_model
import torch


def save_metrics(metrics_history, metrics_file_path):

    print(f"Saving metrics history to {metrics_file_path}...")
    with open(metrics_file_path, "w") as f:
        json.dump(metrics_history, f, indent=4)


def main(model_config=None, checkpoint_path=None):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader, test_loader = load_and_split_time_series_data(
        normal_path=model_config['normal_path'],
        failure_path=model_config['fault_path'],
        batch_size=32,
        random_state=42,
    )

    # Initialize model
    if checkpoint_path: # If a checkpoint is provided, due to previous training interruption, load it
        print(f"Loading model from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load the configuration from the checkpoint
        model_config = checkpoint["config"]
        model = build_model(model_config=model_config)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Define optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Get the starting epoch from the checkpoint
        start_epoch = checkpoint["epoch"]
        print(f"Resuming training from epoch {start_epoch + 1}")
    else:
        # If no checkpoint is provided, start from scratch
        sample_batch = next(iter(train_loader)) # Dynamically infer input shape from a batch
        inputs, _ = sample_batch
        input_shape = inputs.shape[1:]  # Exclude batch size
        model_config['input_shape'] = input_shape
        model = build_model(model_config=model_config)

        model.to(device)

        # Define optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.1, verbose=True
        )

        start_epoch = 0  # Start from the first epoch

    # Define criterion dynamically
    if model_config["class_neurons_num"] == 1:  # Binary classification
        criterion = torch.nn.BCELoss()
    else:  # Multi-class classification
        criterion = torch.nn.CrossEntropyLoss()

    # Initialize Trainer
    trainer = Trainer(model, optimizer, criterion, scheduler=scheduler)

    # Train the model (start from the saved epoch if resuming)
    trainer.train(
        train_loader,
        val_loader,
        device,
        epochs=model_config['num_epochs'],
        config=model_config,
        start_epoch=start_epoch,
        patience=7
    )

    # Save training metrics
    metrics_history = trainer.metrics_history
    save_metrics(metrics_history, metrics_file_path=config['training_res'])

    # Evaluate on the test set
    test_loss, test_metrics = trainer.evaluate(test_loader, device)
    print(f"Test Loss: {test_loss}, Test Metrics: {test_metrics}")
    save_metrics(test_metrics, metrics_file_path=config['test_res'])


if __name__ == "__main__":
    # Add - tau to the architecture. tau if the T contraction-extraction (tau * T) for optimal length of the
    # time-series representative introduced to the transformer by the CNN. - DONE: time_scaler

    synthetic_data_querying = True

    if synthetic_data_querying:
        # synthetic_data paths
        normal_path = '/Users/etayar/PycharmProjects/MultivariateTSDroneAD/uav_data/synthetic_data/normal_data'
        fault_path = '/Users/etayar/PycharmProjects/MultivariateTSDroneAD/uav_data/synthetic_data/anomalous_data'
        # synthetic_data paths
    else:
        normal_path = ''
        fault_path = ''

    configs = [
        {
            'normal_path': normal_path,
            'fault_path': fault_path,
            'checkpoint_epoch_path': "src/data/models_metrics/checkpoint_epoch.pth",
            'best_model_path': "src/data/models_metrics/best_model.pth",
            'training_res': "src/data/models_metrics/training_metrics/training.json",
            'test_res': "src/data/models_metrics/training_metrics/test.json",
            'fuser_name': 'ConvFuser1',
            'transformer_variant': 'vanilla',  # Choose transformer variant
            'use_learnable_pe': True,  # Use learnable positional encoding
            'aggregator': 'attention',  # Use attention-based (time) aggregation
            'num_epochs': 50,
            'class_neurons_num': 1,
            'd_model': 256,
            'nhead': 8,  # # transformer heads
            'num_layers': 6,  # transformer layers
            'dropout': 0.15,
            'time_scaler': 0.678  # The portion of T for conv output time-series latent representative
        },
    ]
    for config in configs:
        main(config)
