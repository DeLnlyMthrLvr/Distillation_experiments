import torch
import numpy as np
import matplotlib.pyplot as plt
import random

def show_difference(original, adversarial, title="Method"):
    """
    Displays the original and adversarial images side by side, along with their difference.
    :param original: Original image (NumPy array or PyTorch Tensor).
    :param adversarial: Adversarial image (NumPy array or PyTorch Tensor).
    :param title: Title for the adversarial image.
    """
    diff = np.abs(original - adversarial)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original Image")

    plt.subplot(1, 3, 2)
    plt.imshow(adversarial, cmap='gray')
    plt.title(f"Adversarial {title}")

    plt.subplot(1, 3, 3)
    plt.imshow(diff, cmap='hot')
    plt.title("Difference (Perturbation)")

    plt.show()

def visualize_adversarial(data, adversarial_data, labels, num_samples=5):
    """
    Displays a comparison of original and adversarial images for a subset of samples.

    :param data: Original MNIST images (NumPy array or PyTorch Tensor) with shape (N, 1, 28, 28)
    :param adversarial_data: Adversarial MNIST images (same shape as `data`).
    :param labels: True labels for the images (can be one-hot or class indices).
    :param num_samples: Number of samples to display.
    """
    if isinstance(data, torch.Tensor):  # Convert tensors to numpy
        data = data.cpu().numpy()
    if isinstance(adversarial_data, torch.Tensor):
        adversarial_data = adversarial_data.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    num_samples = min(num_samples, len(data))  # Ensure we don't exceed dataset size
    indices = np.random.choice(len(data), num_samples, replace=False)  # Select random samples

    fig, axes = plt.subplots(num_samples, 2, figsize=(6, num_samples * 2))
    fig.suptitle("Original vs Adversarial Images", fontsize=14)

    for i, idx in enumerate(indices):
        # Original Image
        axes[i, 0].imshow(data[idx].squeeze(), cmap="gray")
        axes[i, 0].set_title(f"Original (Label: {labels[idx]})")
        axes[i, 0].axis("off")

        # Adversarial Image
        axes[i, 1].imshow(adversarial_data[idx].squeeze(), cmap="gray")
        axes[i, 1].set_title("Adversarial")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_from_dataloader(model, dataloader, temp, device='cpu'):
    """
    Selects a random MNIST image from the dataset, gets its label, and computes its soft labels.

    :param model: Trained teacher model.
    :param dataloader: PyTorch DataLoader containing the dataset.
    :param temp: Temperature parameter for soft labels.
    :param device: Device to run the model on.
    """
    model.eval()

    # Get a random batch
    images, labels = next(iter(dataloader))
    random_idx = random.randint(0, len(images) - 1)

    # Select a random image and its label
    image, label = images[random_idx], labels[random_idx].item()

    # Move image to the correct device
    image = image.unsqueeze(0).to(device)  # Add batch dimension

    # Get soft labels
    with torch.no_grad():
        logits = model(image)
        soft_labels = torch.softmax(logits / temp, dim=1).cpu().numpy().flatten()

    # Display image and labels
    plt.imshow(image.squeeze().cpu(), cmap="gray")  # Use grayscale colormap for MNIST
    plt.title(f"True Label: {label}\nSoft Labels: {soft_labels.round(2)}")
    plt.axis("off")
    plt.show()
