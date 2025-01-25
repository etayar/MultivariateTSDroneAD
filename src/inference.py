import torch


def predict(model_path, model_class, data_loader, device="cpu"):
    """
    model_class (nn.Module): The class definition of the model.
    """
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for inputs in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = (outputs > 0.5).float()  # Binary classification
            predictions.append(preds.cpu().numpy())

    return predictions
