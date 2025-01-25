import torch
from src.multivariate_univariate_fusion_anomaly_detection import build_model


def load_model(checkpoint_path, device):
    """
    Load a model and its configuration from a checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint file.
        device (str): Device to load the model on ("cpu" or "cuda").

    Returns:
        model (nn.Module): The loaded PyTorch model.
        config (dict): The model's configuration.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    # Rebuild model using saved configuration
    model = build_model(model_config=config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    return model, config


def run_inference(model, data_loader, device, is_binary=True):
    """
    Perform inference on data using the given model.

    Args:
        model (nn.Module): The loaded PyTorch model.
        data_loader (DataLoader): Data loader for input data.
        device (str): Device to run inference on ("cpu" or "cuda").
        is_binary (bool): Whether the task is binary classification.

    Returns:
        list: Predictions as a list of NumPy arrays.
    """
    predictions = []
    with torch.no_grad():
        for inputs in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            # Apply appropriate activation
            if is_binary:
                preds = (outputs > 0.5).float()  # Binary classification
            else:
                preds = torch.argmax(outputs, dim=1)  # Multi-class classification

            predictions.append(preds.cpu().numpy())

    return predictions
