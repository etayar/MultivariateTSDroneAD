import os
import json
from datetime import datetime
from src.data.data_loader import load_and_split_time_series_data
from src.training.train import Trainer
from src.multivariate_fusion_anomaly_detection import build_model
import torch
from torch import nn
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingWarmRestarts


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
    if model_config["binary"]:
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


def main(model_config, by_checkpoint=False, by_best_model=True):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    kwargs = {
        'normal_path': model_config.get('normal_path'),
        'abnormal_path': model_config.get('abnormal_path'),
        'csv_data': model_config.get('csv_data'),
        'npy_data': model_config.get('npy_data')
    }

    print("Loading data and splitting....")
    train_loader, val_loader, test_loader, label_counts = load_and_split_time_series_data(
        split_rates=model_config['split_rates'],
        batch_size=model_config['batch_size'],
        random_state=42,
        **kwargs
    )

    sample_batch = next(iter(train_loader))  # Dynamically infer input shape from a batch
    inputs, _ = sample_batch
    input_shape = inputs.shape[1:]  # Exclude batch size
    model_config['input_shape'] = input_shape

    num_classes = len(label_counts)

    model_config['binary'] = True if num_classes == 2 else False
    model_config['multi_label'] = False if model_config['binary'] or  model_config['multi_class'] else True

    # Define criterion dynamically
    criterion = get_criterion(model_config, label_counts)
    criterion.to(device)

    model_config['criterion'] = criterion
    model_config['class_neurons_num'] = 1 if num_classes == 2 else num_classes

    lr = model_config['learning_rate']
    weight_decay = model_config['weight_decay']
    num_epochs = model_config['num_epochs']

    print('Initialize model...')
    if by_checkpoint:
        # If training was interrupted, resume from checkpoint. It can't deal with multiple datasets training.
        checkpoint_path = model_config['checkpoint_epoch_path']
        print(f"Loading model from checkpoint: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Ensure the checkpoint contains the necessary keys
        required_keys = ["config", "model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "epoch"]
        for key in required_keys:
            assert key in checkpoint, f"Checkpoint is missing key: {key}"

        # Load the model configuration from the checkpoint
        model_config = checkpoint["config"]

        # Build the model and load weights
        model = build_model(model_config=model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        # Reinitialize optimizer and load its state
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        warmup_epochs = 3
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),  # Gradual LR increase
                CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
            ],
            milestones=[warmup_epochs]
        )
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Resume training from the last saved epoch
        start_epoch = checkpoint["epoch"] + 1  # Start from the next epoch
        print(f"Resuming training from epoch {start_epoch}")
    elif by_best_model:
        best_model_path = model_config['previous_dataset_path']
        print(f"Loading model from best_model.pth: {best_model_path}")
        best_model = torch.load(best_model_path, map_location=device)

        # Load the model
        model = build_model(model_config=model_config)
        model.to(device)

        # Update layers that depend on T_i and class count
        new_T_i = input_shape[1]  # Get time steps from dataset config
        new_class_neurons_num = model_config["class_neurons_num"]

        # Compute new projection layer output size
        new_projection_out_channels = int(model.time_scaler * new_T_i)

        existing_projection_layer = model.conv_fuser.projection[0]  # First layer in Sequential
        in_channels = existing_projection_layer.in_channels  # Extract in_channels

        # Replace Projection Layer
        model.conv_fuser.projection = nn.Sequential(
            nn.Conv2d(in_channels, model_config['d_model'], kernel_size=1),
            nn.AdaptiveAvgPool2d((new_projection_out_channels, 1))
        )

        # Replace Fully Connected Layer (fc2) for classification
        in_features = model.fc2.in_features
        model.fc2 = nn.Linear(in_features, new_class_neurons_num)

        model.to(device)

        # Load only matching parameters from checkpoint
        model_state_dict = model.state_dict()
        checkpoint_state_dict = best_model["model_state_dict"]

        filtered_checkpoint_state = {
            k: v for k, v in checkpoint_state_dict.items() if
            k in model_state_dict and model_state_dict[k].shape == v.shape
        }
        model_state_dict.update(filtered_checkpoint_state)
        model.load_state_dict(model_state_dict, strict=False)  # strict=False allows skipping mismatches

        # Reset BatchNorm stats if needed
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d):
                layer.reset_running_stats()

        # Reinitialize the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        warmup_epochs = 3
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),  # Gradual LR increase
                CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
            ],
            milestones=[warmup_epochs]
        )

        # Reset start epoch
        start_epoch = 0
        print(f"Resuming training from epoch {start_epoch + 1}")
    else:
        model = build_model(model_config=model_config)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        warmup_epochs = 3  # First 3 epochs with warmup
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),  # Gradual LR increase
                CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
            ],
            milestones=[warmup_epochs]
        )

        start_epoch = 0  # Start from the first epoch

    print('Initialize Trainer...')
    trainer = Trainer(
        model,
        optimizer,
        criterion,
        num_classes=len(label_counts),
        scheduler=scheduler,
        is_binary=model_config['binary'],
        is_multi_label=model_config['multi_label'],
        is_multi_class=model_config['multi_class'],
        prediction_threshold=model_config['prediction_threshold']
    )

    trainer.train(
        train_loader,
        val_loader,
        device,
        epochs=model_config['num_epochs'],
        config=model_config,
        start_epoch=start_epoch,
        patience=5
    )

    # Save training metrics
    metrics_history = trainer.metrics_history
    save_metrics(metrics_history, metrics_file_path=model_config['training_res'])

    # Evaluate on the test set
    test_loss, test_metrics = trainer.evaluate(
        test_loader,
        device
    )
    print(f"Test Loss: {test_loss}, Test Metrics: {test_metrics}")
    save_metrics(test_metrics, metrics_file_path=model_config['test_res'])


def get_normal_abnormal_paths(directory: str):
    paths = {}

    for category in ["normal", "abnormal"]:
        category_path = os.path.join(directory, category)

        if not os.path.exists(category_path):
            raise FileNotFoundError(f"Missing required folder: {category_path}")

        if not os.path.isdir(category_path):
            raise NotADirectoryError(f"Expected a folder, but found a file: {category_path}")

        paths[category] = category_path

    return paths


if __name__ == "__main__":
    ################### DATASETS ###################
    UEA_DATASETS = {
        #TODO: Those datasets need to be adapted to the new convention - normal_path and abnormal_path
        'Heartbeat': 'binary',
        'Handwriting': 'multiclass',
        'PhonemeSpectra': 'multiclass',
        'SelfRegulationSCP1': 'multiclass',
        'EthanolConcentration': 'multiclass',
        'FaceDetection': 'multiclass'
    }

    if "COLAB_GPU" in os.environ:
        EEG_DATASETS = {
            'CHBMIT': 'binary'
        }
    else:

        EEG_DATASETS = {
            'CHBMIT2_1': 'binary'
        }
    csv_data = False
    npy_data = True
    ################### DATASETS ###################
    experiment_num = 1  # In case we want to train different configuration at the same day.
    retrieve_last_training_session = False
    multiple_data_sets_training_mode = True
    training_sets = EEG_DATASETS

    # Get the current date in "YYYY-MM-DD" format
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Detect if running in Google Colab
    if "COLAB_GPU" in os.environ:
        print("Running in Google Colab - Updating paths!")
        base_path = "/content/drive/My Drive/My_PHD/My_First_Paper"

        # Create the directory for today's date if it doesn't exist
        date_dir = os.path.join(
            "/content/drive/My Drive/My_PHD/My_First_Paper/models_metrics", current_date)
    else:
        print("Running locally - Using Mac paths.")
        base_path = "/Users/etayar/PycharmProjects/MultivariateTSDroneAD"

        # Create the directory for today's date if it doesn't exist
        date_dir = os.path.join("src/data/models_metrics", current_date)

    previous_dataset_path = ''
    for ds_num, k_v in enumerate(training_sets.items()):
        data_set, task = k_v

        directory = os.path.join(base_path, data_set)
        paths = get_normal_abnormal_paths(directory)

        normal_path = paths["normal"]
        abnormal_path = paths["abnormal"]

        multi_class = True if task == 'multiclass' else False

        ################## DESTINATION PATHS ###################
        def get_experiment_path(date_dir, data_set):
            """Automatically determine the next available experiment number."""

            # Get today's date folder
            os.makedirs(date_dir, exist_ok=True)  # Ensure the date folder exists

            # Find the next available experiment number
            experiment_num = 1
            while os.path.exists(os.path.join(date_dir, f"{data_set}_{experiment_num}")):
                experiment_num += 1  # Increment until a free folder is found

            # Create the final path for this experiment
            full_path = os.path.join(date_dir, f"{data_set}_{experiment_num}")
            os.makedirs(full_path, exist_ok=True)

            return full_path

        full_path = get_experiment_path(date_dir, data_set)
        # full_path = os.path.join(date_dir, data_set + f'_{experiment_num}')
        # os.makedirs(full_path, exist_ok=True)

        checkpoint_path = os.path.join(full_path, "checkpoint_epoch.pth")
        best_model_path = os.path.join(full_path, "best_model.pth")
        training_res = os.path.join(full_path, "training.json")
        test_res = os.path.join(full_path, "test.json")
        ########################################################

        config = {
            # Data paths
            'csv_data': csv_data,
            'npy_data': npy_data,
            'normal_path': normal_path,
            'abnormal_path': abnormal_path,
            'checkpoint_epoch_path': checkpoint_path,
            'best_model_path': best_model_path,
            'previous_dataset_path': previous_dataset_path,

            # Results storage
            'training_res': training_res,
            'test_res': test_res,

            # Model structure
            'multi_class': multi_class,
            'fuser_name': 'ConvFuser2',
            'blocks': (3, 4, 5, 3),
            'transformer_variant': 'performer',
            'use_learnable_pe': True,
            'aggregator': 'conv',

            # Training parameters
            'num_epochs': 50,
            'batch_size': 8,
            'learning_rate': 5e-5,
            'weight_decay': 5e-4,

            # Model parameters
            'd_model': 256,
            'nhead': 8,
            'num_layers': 4,
            'dropout': 0.28,
            'time_scaler': 1,

            # Evaluation parameters
            'prediction_threshold': 0.5,
            'split_rates': (0.3, 0.5)  # Train/Validation/Test split
        }

        print(
            f"d_model: {config['d_model']}\n"
            f"nhead: {config['nhead']}\n"
            f"num_layers: {config['num_layers']}\n"
            f"batch_size: {config['batch_size']}\n"
            f"dropout: {config['dropout']}\n"
            f"learning_rate: {config['learning_rate']}\n"
            f"time_scaler: {config['time_scaler']}\n"
            f"multi_class: {config['multi_class']}\n"
            f"blocks: {config['blocks']}"
        )

        by_best_model = True if multiple_data_sets_training_mode and ds_num > 0 else None
        by_checkpoint = True if retrieve_last_training_session else None

        main(config, by_checkpoint=by_checkpoint, by_best_model=by_best_model)
        previous_dataset_path = best_model_path
