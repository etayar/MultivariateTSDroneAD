import os
import json
from datetime import datetime
from src.data.data_loader import load_and_split_time_series_data
from src.training.train import Trainer
from src.multivariate_fusion_anomaly_detection import build_model
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR


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
        'failure_path': model_config.get('fault_path'),
        'multilabel_path': model_config.get('multilabel_path'),
        'multiclass_path': model_config.get('multiclass_path'),
        'experimental_dataset_name': model_config.get('experimental_dataset_name')
    }
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
    num_epochs = model_config['num_epochs']

    # Initialize model
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
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Reinitialize scheduler and load its state
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Resume training from the last saved epoch
        start_epoch = checkpoint["epoch"] + 1  # Start from the next epoch
        print(f"Resuming training from epoch {start_epoch}")
    elif by_best_model:
        best_model_path = model_config['best_model_path']
        print(f"Loading model from best_model.pth: {best_model_path}")
        best_model = torch.load(best_model_path, map_location=device)

        # Load the model
        model = build_model(model_config=model_config)

        # Update layers that depend on T_i and class count
        new_T_i = input_shape[1]  # Get time steps from dataset config
        new_class_neurons_num = model_config["class_neurons_num"]

        # Compute new projection layer output size
        new_projection_out_channels = int(model.time_scaler * new_T_i)

        existing_projection_layer = model.conv_fuser.projection[0]  # First layer in Sequential
        in_channels = existing_projection_layer.in_channels  # Extract in_channels

        # Replace Projection Layer
        model.conv_fuser.projection = nn.Sequential(
            nn.Conv2d(in_channels, new_projection_out_channels, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Replace Fully Connected Layer (fc2) for classification
        in_features = model.fc2.in_features
        model.fc2 = nn.Linear(in_features, new_class_neurons_num)

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
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        # Reinitialize the scheduler (with cosine annealing)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

        # Reset start epoch
        start_epoch = 0
        print(f"Resuming training from epoch {start_epoch + 1}")
    else:
        model = build_model(model_config=model_config)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

        start_epoch = 0  # Start from the first epoch

    # Initialize Trainer
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

    # Train the model (start from the saved epoch if resuming)
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


if __name__ == "__main__":
    ################### UEA DATASETS ###################
    UEA_DATASETS = {
        'Heartbeat': 'binary',
        'Handwriting': 'multiclass',
        'PhonemeSpectra': 'multiclass',
        'SelfRegulationSCP1': 'multiclass',
        'EthanolConcentration': 'multiclass',
        'FaceDetection': 'multiclass'
    }
    experimental_dataset_name = 'Handwriting'
    # experimental_dataset_name = 'Heartbeat'
    ################### UEA DATASETS ###################

    # Get the current date in "YYYY-MM-DD" format
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Detect if running in Google Colab
    if "COLAB_GPU" in os.environ:
        print("Running in Google Colab - Updating paths!")
        base_path = "/content/drive/My Drive/My_PHD/My_First_Paper/MultivariateTSDroneAD"
        multilabel_base_path = ""
        multiclass_base_path = ""

        # Create the directory for today's date if it doesn't exist
        date_dir = os.path.join(
            "/content/drive/My Drive/My_PHD/My_First_Paper/models_metrics", current_date)
    else:
        print("Running locally - Using Mac paths.")
        base_path = "/Users/etayar/PycharmProjects/MultivariateTSDroneAD"
        multilabel_base_path = ""
        multiclass_base_path = ""

        # Create the directory for today's date if it doesn't exist
        date_dir = os.path.join("src/data/models_metrics", current_date)

    folder = ""
    normal_path = None  # os.path.join(base_path, folder, "normal_data")
    fault_path = None  # os.path.join(base_path, folder, "anomalous_data")

    multilabel_path = None if not multilabel_base_path else os.path.join(multilabel_base_path, "multilabel_path")
    multiclass_path = None if not multiclass_base_path else os.path.join(multiclass_base_path, "multiclass_path")

    retrieve_last_training_session = False
    multiple_data_sets_training_mode = True
    training_sets = UEA_DATASETS if multiple_data_sets_training_mode else [experimental_dataset_name]

    for ds_num, k_v in enumerate(training_sets.items()):
        data_set, task = k_v

        full_path = os.path.join(date_dir, data_set)
        os.makedirs(full_path, exist_ok=True)

        checkpoint_path = os.path.join(full_path, "checkpoint_epoch.pth")
        best_model_path = os.path.join(full_path, "best_model.pth")
        training_res = os.path.join(full_path, "training.json")
        test_res = os.path.join(full_path, "test.json")

        multi_class = True if task == 'multiclass' else False

        config = {
            'normal_path': normal_path,
            'fault_path': fault_path,
            'multilabel_path': multilabel_path,
            'multiclass_path': multiclass_path,
            'checkpoint_epoch_path': checkpoint_path,
            'best_model_path': best_model_path,
            'training_res': training_res,
            'test_res': test_res,
            'multi_class': multi_class, # binary class' is determined by the number of data classes. Multilabel class' is concluded.
            'fuser_name': 'ConvFuser2',
            'blocks': tuple([3 for _ in range(12)]),  # The ResNet skip connection blocks
            'transformer_variant': 'performer',  # Choose transformer variant
            'use_learnable_pe': True,  # Use learnable positional encoding
            'aggregator': 'conv',  # Use aggregation
            'num_epochs': 50,
            'd_model': 512,
            'nhead': 8,  # # transformer heads
            'num_layers': 8,  # transformer layers
            'batch_size': 16,
            'dropout': 0.05,
            'learning_rate': 1e-4,
            'time_scaler': None,  # The portion of T for conv output time-series latent representative
            'prediction_threshold': 0.5,
            'split_rates': (0.2, 0.3),
            'experimental_dataset_name': data_set
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

        # load_model = input("Load existing model (strictly yes or no answer)?").lower().strip()
        # while load_model not in ['yes', 'no']:
        #     load_model = input("Load existing model (strictly yes or no answer)?").lower().strip()
        # if load_model == 'no':
        #     print("Start training new model.")
        #     by_checkpoint = None

        main(config, by_checkpoint=by_checkpoint, by_best_model=by_best_model)
