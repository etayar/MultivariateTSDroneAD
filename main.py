import os
import json
from datetime import datetime
from src.data.data_loader import load_and_split_time_series_data
from src.training.train import Trainer
from src.multivariate_fusion_anomaly_detection import build_model
import torch


def save_metrics(metrics_history, metrics_file_path):

    print(f"Saving metrics history to {metrics_file_path}...")
    with open(metrics_file_path, "w") as f:
        json.dump(metrics_history, f, indent=4)


def get_criterion(model_config, label_counts):
    """
    Dynamically computes class weights based on label distribution
    and applies the correct loss function for binary, multi-class, and multi-label classification.

    Args:
        model_config (dict): Contains classification type information.
        label_counts (dict): Dictionary with class frequencies {class_id: count}.

    Returns:
        torch loss function with computed class weights.
    """

    total_samples = sum(label_counts.values())  # Total dataset size
    class_weights = {cls: total_samples / count for cls, count in label_counts.items()}  # Compute inverse frequency

    # Convert to PyTorch tensor and sort by class index
    class_weights_tensor = torch.tensor([class_weights[cls] for cls in sorted(label_counts.keys())])

    # Apply correct loss function
    if model_config["class_neurons_num"] == 1 and not model_config.get("multi_label", False):
        # Binary classification (Single Sigmoid neuron)
        num_negative = label_counts[0]  # Count of negative class
        num_positive = label_counts[1]  # Count of positive class
        pos_weight = torch.tensor([num_negative / num_positive])  # Adjust weight for imbalance
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # Binary class weighting

    elif model_config.get("multi_label", False):
        # Multi-label classification (Multiple independent Sigmoid neurons)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor)  # Per-class weighting

    else:
        # Multi-class classification (Softmax output)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)  # Apply class weights

    return criterion


def main(model_config=None, checkpoint_path=None):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader, test_loader, label_counts = load_and_split_time_series_data(
        normal_path=model_config['normal_path'],
        failure_path=model_config['fault_path'],
        batch_size=model_config['batch_size'],
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
    criterion = get_criterion(model_config, label_counts)

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
    save_metrics(metrics_history, metrics_file_path=model_config['training_res'])

    # Evaluate on the test set
    test_loss, test_metrics = trainer.evaluate(
        test_loader,
        device,
        is_multi_label = model_config['multi_label'],
        prediction_threshold=model_config['prediction_threshold']
    )
    print(f"Test Loss: {test_loss}, Test Metrics: {test_metrics}")
    save_metrics(test_metrics, metrics_file_path=model_config['test_res'])


if __name__ == "__main__":

    synthetic_data_querying = False

    if synthetic_data_querying:
        # synthetic_data paths
        normal_path = '/Users/etayar/PycharmProjects/MultivariateTSDroneAD/uav_data/synthetic_data/normal_data'
        fault_path = '/Users/etayar/PycharmProjects/MultivariateTSDroneAD/uav_data/synthetic_data/anomalous_data'
        # synthetic_data paths
    else:
        normal_path = '/Users/etayar/PycharmProjects/MultivariateTSDroneAD/ServerMachineDataset/normal_data'
        fault_path = '/Users/etayar/PycharmProjects/MultivariateTSDroneAD/ServerMachineDataset/anomalous_data'

    # Get the current date in "YYYY-MM-DD" format
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Create the directory for today's date if it doesn't exist
    date_dir = os.path.join("src/data/models_metrics", current_date)
    os.makedirs(date_dir, exist_ok=True)

    checkpoint_path = os.path.join(date_dir, "checkpoint_epoch.pth")
    best_model_path = os.path.join(date_dir, "best_model.pth")
    training_res = os.path.join(date_dir, "training.json")
    test_res = os.path.join(date_dir, "test.json")

    configs = [
        {
            'normal_path': normal_path,
            'fault_path': fault_path,
            'checkpoint_epoch_path': checkpoint_path,
            'best_model_path': best_model_path,
            'training_res': training_res,
            'test_res': test_res,
            'class_neurons_num': 1,  # Depends on the classification task (1 for binary...)
            'fuser_name': 'ConvFuser1',
            'transformer_variant': 'performer',  # Choose transformer variant
            'use_learnable_pe': True,  # Use learnable positional encoding
            'aggregator': 'attention',  # Use attention-based (time) aggregation
            'num_epochs': 50,
            'd_model': 128,
            'nhead': 8,  # # transformer heads
            'num_layers': 4,  # transformer layers
            'batch_size': 8,
            'dropout': 0.15,
            'time_scaler': 1,  # The portion of T for conv output time-series latent representative
            'multi_label': False,
            'prediction_threshold': 0.5
        },
    ]
    for config in configs:
        main(config)
