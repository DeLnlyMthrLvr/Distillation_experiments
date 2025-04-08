import torch
import numpy as np
from processing.preprocess import to_tensor


def evaluate_model(model, data, labels, device=None):
    """
    Evaluate model accuracy.

    Args:
        model: PyTorch model (ART-wrapped or regular)
        data: Input samples (NumPy array or Tensor).
        labels: True labels as class indices (NumPy array or Tensor).
        device: "cpu" or "cuda" (use GPU if available).

    Returns:
        accuracy (float)
    """
    # Check if CUDA is available, otherwise fallback to CPU
    if device is None:
        device = "cpu"
    # Move model to device
    model.to(device)
    model.eval()  # Set model to evaluation mode

    # Convert NumPy arrays to PyTorch tensors if needed
    if isinstance(data, np.ndarray):
        data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
    else:
        data_tensor = data.to(device)

    if isinstance(labels, np.ndarray):
        labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
    else:
        labels_tensor = labels.to(device)

    # Forward pass: Compute predictions
    with torch.no_grad():
        outputs = model(data_tensor)  # Get raw logits
        predictions = torch.argmax(outputs, dim=1)  # Convert logits to class labels

    # Accuracy Calculation (No one-hot encoding assumption)
    accuracy = (predictions == labels_tensor).float().mean().item() * 100

    return accuracy


def calculate_asr(preds_clean, preds_adv):
    """
    Calculate Adversarial Success Rate (ASR).

    Args:
        preds_clean: Model predictions on clean data.
        preds_adv: Model predictions on adversarial data.
    Returns:
        asr (float): Adversarial Success Rate in percentage.
    """

    successful_attacks = (preds_clean != preds_adv).sum().item()
    asr = (successful_attacks / len(preds_clean)) * 100
    return asr


def calculate_mean_gradient_amplitude(model, data, labels, criterion, device=None):
    """Calculate Mean Gradient Amplitude divided into 10 bins.
    Args:
        model: PyTorch model.
        data: Input samples (NumPy array or Tensor).
        labels: True labels as class indices (NumPy array or Tensor).
        criterion: Loss function.
    Returns:
        mean_gradient_amplitude (list): Mean gradient amplitude divided into 10 bins.
    """

    # Check if CUDA is available, otherwise fallback to CPU
    if device is None:
        device = "cpu"
    # Move model to device
    model.to(device)
    model.eval()  # Set model to evaluation mode

    # Convert data to PyTorch tensors if needed
    data = to_tensor(data, device)
    labels = to_tensor(labels, device).long()

    data.requires_grad = True
    outputs = model(data)
    loss = criterion(outputs, labels)
    loss.backward()
    gradients = data.grad.abs().cpu().numpy()

    mean_gradient_amplitude = gradients.mean(axis=(0, 1, 2, 3))
    return mean_gradient_amplitude


def calculate_binned_gradient_amplitude(
    model, data, labels, criterion, num_bins=10, device=None
):
    """
    Calculate Mean Gradient Amplitude divided into bins.
    Args:
        model: PyTorch model.
        data: Input samples (Tensor).
        labels: True labels (Tensor).
        criterion: Loss function.
        num_bins: Number of bins to divide the gradient amplitudes.
        device: Device (cpu or cuda).
    Returns:
        bin_counts (list): List of counts of samples in each gradient amplitude bin.
    """
    if device is None:
        device = "cpu"
    model.to(device)
    model.eval()

    # Convert data to PyTorch tensors if needed
    data = to_tensor(data, device)
    labels = to_tensor(labels, device).long()

    # Enable gradient calculation
    data.requires_grad = True

    # Forward pass
    outputs = model(data)
    loss = criterion(outputs, labels)
    loss.backward()

    # Calculate mean gradient amplitude for each sample
    gradients = data.grad.abs().view(len(data), -1)
    mean_gradients = gradients.mean(dim=1).cpu().numpy()

    # Create bins and calculate histogram
    bin_edges = np.logspace(
        -40, 0, num_bins + 1
    )  # Logarithmic scale for better representation
    bin_counts, _ = np.histogram(mean_gradients, bins=bin_edges)

    return bin_counts, bin_edges


def calculate_robustness(data, adv_data):
    """
    Calculate Robustness as the average minimum effective perturbation.

    Args:
        data: Original input samples (NumPy array or Tensor).
        adv_data: Adversarial input samples (NumPy array or Tensor).
    Returns:
        robustness (float): Average minimum effective perturbation.
    """
    perturbations = (adv_data - data).view(len(data), -1).norm(p=2, dim=1)
    robustness = perturbations.mean().item()
    return robustness


def calculate_adversarial_confidence(outputs_adv):
    """
    Calculate the average classification confidence on adversarial examples.

    Assumes that outputs_adv are log probabilities (log softmax).

    Args:
        outputs_adv: Model outputs on adversarial data (log softmax).
    Returns:
        mean_confidence (float): Average confidence on adversarial examples.
    """
    # Exponentiate to get probabilities from log softmax
    probabilities = torch.exp(outputs_adv)  # Convert log softmax back to probabilities
    confidence_adv = probabilities.max(dim=1).values
    mean_confidence = confidence_adv.mean().item() * 100
    return mean_confidence


def evaluate_adversarial_metrics(model, data, labels, adv_data, device=None):
    """
    Evaluate model on adversarial metrics.

    Args:
        model: PyTorch model (ART-wrapped or regular).
        data: Clean input samples (NumPy array or Tensor).
        labels: True labels as class indices (NumPy array or Tensor).
        adv_data: Adversarial input samples (NumPy array or Tensor).
        device: "cpu" or "cuda" (use GPU if available).

    Returns:
        metrics (dict): Dictionary containing ASR, adversarial robustness, and classification confidence.
    """

    # Check if CUDA is available, otherwise fallback to CPU
    if device is None:
        device = "cpu"
    # Move model to device
    model.to(device)
    model.eval()  # Set model to evaluation mode

    # Convert data to PyTorch tensors if needed
    data = to_tensor(data, device)
    labels = to_tensor(labels, device).long()
    adv_data = to_tensor(adv_data, device)

    # Forward pass on clean data
    with torch.no_grad():
        outputs_clean = model(data)
        preds_clean = torch.argmax(outputs_clean, dim=1)

    # Forward pass on adversarial data
    with torch.no_grad():
        outputs_adv = model(adv_data)
        preds_adv = torch.argmax(outputs_adv, dim=1)

    # Calculate metrics
    metrics = {
        "Adversarial Success Rate": calculate_asr(preds_clean, preds_adv),
        "Adversarial Robustness % (Avg Min Perturbation)": calculate_robustness(
            data, adv_data
        ),
        "Adversarial Confidence": calculate_adversarial_confidence(outputs_adv),
    }

    return metrics
