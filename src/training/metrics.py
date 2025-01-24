"""
This file can hold custom evaluation metrics you may use for monitoring training progress.
Examples:
    * Accuracy
    * F1 score
    * ROC AUC score
    * Or any domain-specific metric
"""

def compute_metrics(outputs, labels):
    """
    Compute precision, recall, and F1-score for binary classification.
    Args:
        outputs (torch.Tensor): Model predictions (probabilities).
        labels (torch.Tensor): Ground truth binary labels.
    Returns:
        dict: Precision, recall, F1-score, and accuracy.
    """
    # Apply threshold to convert probabilities into binary predictions
    predictions = (outputs > 0.5).float()

    # True positives, false positives, false negatives
    tp = (predictions * labels).sum().item()
    fp = (predictions * (1 - labels)).sum().item()
    fn = ((1 - predictions) * labels).sum().item()

    # Precision, recall, F1-score
    precision = tp / (tp + fp + 1e-8)  # Avoid division by zero
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    accuracy = (predictions == labels).float().mean().item()

    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

