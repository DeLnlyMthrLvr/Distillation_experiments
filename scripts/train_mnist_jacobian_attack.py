"""
Train a teacher model on MNIST and generate adversarial examples with Jacobian-Saliency method.

This script trains a teacher model on the MNIST dataset, generates adversarial examples,
and evaluates the models' performance on both clean and adversarial data.

To run the script you can use the following command, adjusting argumments as needed:
ipython scripts/train_mnist_jacobian_attack.py -- --lr 0.001 --batch_size 256 --max_epochs 5 --temperature 20 --num_samples 5 --device 'mps'


"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import ssl
import random
from pathlib import Path
from art.estimators.classification import PyTorchClassifier
from matplotlib import pyplot as plt
import time
import logging
import argparse


from models.mnist_models import MnistNet
from evaluation.metrics import evaluate_adversarial_metrics
from evaluation.metrics import evaluate_model
from processing.visualize import show_difference
from processing.visualize import visualize_adversarial
from processing.distillation import train_student, train_teacher, load_model
from evaluation.metrics import (
    calculate_mean_gradient_amplitude,
    calculate_binned_gradient_amplitude,
)
from utils.experiment_saver import save_experiment_results
from utils.jsma_generate import generate_adversarial_samples


from art.attacks.evasion.saliency_map import SaliencyMapMethod


logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

LOGGER = logging.getLogger(__name__)


def main(
    lr: float,
    batch_size: int,
    n_channels: int,
    w: int,
    h: int,
    max_epochs: int,
    temperature: float,
    num_samples: int,
    device: str,
    save_path: str,
):
    """
    Main function to train a teacher and a distilled studemt model on MNIST, generate adversarial examples, and evaluate the models.

    Args:
        lr (float): Learning rate.
        batch_size (int): Batch size.
        n_channels (int): Number of channels in the input images.
        w (int): Width of the input images.
        h (int): Height of the input images.
        max_epochs (int): Number of epochs to train the model.
        temperature (float): Temperature parameter for softmax scaling / distillation.
        num_samples (int): Number of samples to use for generating adversarial examples.
        device (str): Device to run the training and evaluation on.
        save_path (str): Path to save the experiment results.
    """
    ssl._create_default_https_context = ssl._create_stdlib_context

    start_time = time.perf_counter()

    # Ensure reproducibility
    # Set a global random seed
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    LOGGER.info(f"Playing on {device}")

    dt_p = Path("data/mnist")

    # Specify classes as string and number of labels
    classes = [str(i) for i in range(10)]
    n_labels = len(classes)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # Load MNIST dataset
    trainset = datasets.MNIST(
        root=f"{dt_p.absolute()}/train", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    testset = datasets.MNIST(
        root=f"{dt_p.absolute()}/test", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    # Define a simple CNN model for MNIST classification
    teacher_model = MnistNet(input_size=w, temperature=temperature).to(device)
    student_model = MnistNet(input_size=w, temperature=temperature).to(device)

    # Specify the loss function and optimizer for teacher model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(teacher_model.parameters(), lr=lr)

    # Get seperate vars for targets and data
    mnist_targets = testset.targets.int().numpy()
    mnist_data = testset.data.unsqueeze(1).float().numpy()

    # Select a small test subset to attack
    # adversarial attacks can be slow, so we only use a small subset of the test set
    # First shuffle
    indices = torch.randperm(len(testset.data))
    mnist_data_shuffled = mnist_data[indices]
    mnist_targets_shuffled = mnist_targets[indices]
    # Then select subsets
    mnist_data_subset = mnist_data_shuffled[:num_samples] / 255
    mnist_targets_subset = mnist_targets_shuffled[:num_samples]

    ## Teacher Model

    # Load the model
    teacher_model = load_model(
        teacher_model,
        device=device,
        load_path=save_path,
        model_name="mnist_teacher_model",
    )

    # If the model is not loaded (returns None), train and save it
    if teacher_model is None:
        LOGGER.info("\nTraining Teacher Model")

        # Initialize the teacher model again
        teacher_model = MnistNet(input_size=w, temperature=temperature).to(device)

        # Train the teacher model
        train_teacher(
            teacher_model,
            trainloader,
            criterion=criterion,
            optimizer=torch.optim.Adam(teacher_model.parameters(), lr=0.001),
            device=device,
            epochs=max_epochs,
            save_path=save_path,
            model_name="mnist_teacher_model",
        )

    # Wrap in ART PyTorchClassifier
    art_model_t = PyTorchClassifier(
        model=teacher_model,
        clip_values=(0, 1),  # Min and Max pixel values (normalize if needed)
        loss=criterion,
        optimizer=optimizer,
        input_shape=(n_channels, w, h),
        nb_classes=n_labels,
    )

    # Evaluate model on entire testset
    teacher_accuracy = evaluate_model(
        art_model_t.model, mnist_data, mnist_targets, device=device
    )
    LOGGER.info(f"Test Accuracy: {teacher_accuracy:.2f}%")

    # Display Gradient Amplitude
    LOGGER.info(
        f"Mean Gradient Amplitude: {calculate_mean_gradient_amplitude(art_model_t.model, mnist_data, mnist_targets, criterion, device=device)}"
    )
    LOGGER.info(
        f"Binned Mean Gradient Amplitude (as implemented in Paparnot et. al 2016): {calculate_binned_gradient_amplitude(art_model_t.model, mnist_data, mnist_targets, criterion, device=device)}"
    )

    # Ensure teacher model is not on mps to create the attacks
    if device == "mps":
        teacher_model.to("cpu")

    # Adversarial attacks
    # Generate Adversarial Examples from the Teacher Model
    LOGGER.info("\nGenerating Adversarial Examples from Teacher Model:")

    LOGGER.info("Generating Jacobian-Saliency Adversarial Examples")

    expanded_data, expanded_labels, x_adv, y_adv = generate_adversarial_samples(mnist_data_subset, mnist_targets_subset, art_model_t, theta=0.8, gamma=0.7, batch_size=32)

    visualize_adversarial(expanded_data, x_adv, expanded_labels, rgb=False)
    x_adv = x_adv.reshape(-1, n_channels, w, h)
    show_difference(
        expanded_data[0][0], x_adv[0][0], title="Jacobian-Saliency Map Method"
    )
    # Flatten
    x_adv = x_adv.reshape(-1, n_channels, w, h)


    # Transfer model back to device
    teacher_model.to(device)

    # Evaluate the teacher model on adversarial examples
    LOGGER.info("Evaluating Teacher Model on Teacher-based Adversarial Examples:")

    LOGGER.info(
        evaluate_adversarial_metrics(
            art_model_t.model,
            expanded_data,
            expanded_labels,
            x_adv,
            device=device,
        )
    )

    ## Student Model

    LOGGER.info("\nTraining Student Model using Defensive Distillation")
    # Train the student model using knowledge distillation
    train_student(
        teacher_model,
        student_model,
        trainloader,
        criterion=criterion,
        epochs=max_epochs,
        lr=lr,
        temperature=temperature,
        device=device,
    )

    # Wrap in ART PyTorchClassifier
    art_model_s = PyTorchClassifier(
        model=student_model,
        clip_values=(0, 1),  # Min and Max pixel values (normalize if needed)
        loss=criterion,
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=10,
    )

    # Evaluate model on entire testset
    student_accuracy = evaluate_model(
        art_model_s.model, mnist_data, mnist_targets, device=device
    )
    LOGGER.info(f"Test Accuracy: {student_accuracy:.2f}%")

    # Display Gradient Amplitude
    LOGGER.info(
        f"Mean Gradient Amplitude: {calculate_mean_gradient_amplitude(art_model_s.model, mnist_data, mnist_targets, criterion, device=device)}"
    )
    LOGGER.info(
        f"Binned Mean Gradient Amplitude (as implemented in Paparnot et. al 2016): {calculate_binned_gradient_amplitude(art_model_s.model, mnist_data, mnist_targets, criterion, device=device)}"
    )

    # Ensure student model is not on mps to create the attacks
    if device == "mps":
        student_model.to("cpu")

    # Adversarial attacks
    # Generate Adversarial Examples from the Student Model
    LOGGER.info("\nGenerating Adversarial Examples from Student Model:")

    LOGGER.info("Generating FSGM Adversarial Examples")

    expanded_data, expanded_labels, x_adv_s, y_adv_s = generate_adversarial_samples(mnist_data_subset, mnist_targets_subset, art_model_t, theta=0.8, gamma=0.7, batch_size=32)


    visualize_adversarial(expanded_data, x_adv_s, expanded_labels, rgb=False)
    show_difference(
        expanded_data[0][0], x_adv_s[0][0], title="Jacobian-Saliency Map Method"
    )

    # Transfer model back to device
    student_model.to(device)

    # Evaluate the student model on adversarial examples
    LOGGER.info("Evaluating Student Model on Student-based Adversarial Examples:")
    # FSGM
    LOGGER.info(
        evaluate_adversarial_metrics(
            art_model_s.model,
            expanded_data,
            expanded_labels,
            x_adv_s,
            device=device,
        )
    )

    ## Cross-Result Evaluation
    LOGGER.info("\nCross-Result Evaluation:")
    LOGGER.info("Evaluating Student Model on Teacher-based Adversarial Examples:")
    # FSGM
    LOGGER.info(
        evaluate_adversarial_metrics(
            art_model_s.model,
            expanded_data,
            expanded_labels,
            x_adv,
            device=device,
        )
    )
 
    LOGGER.info("Evaluating Teacher Model on Student-based Adversarial Examples:")
    # FSGM
    LOGGER.info(
        evaluate_adversarial_metrics(
            art_model_t.model,
            expanded_data,
            expanded_labels,
            x_adv_s,
            device=device,
        )
    )


    # Save experiment results
    LOGGER.info("Saving experiment results")
    save_experiment_results(
        save_path,
        [
            "Epochs",
            "Learning Rate",
            "Batch Size",
            "Temperature",
            "Num Samples",
            "Accuracy (T)",
            "Mean Gradient Amplitude (T)",
            "Metrics (JSMA)",
            "Accuracy (S)",
            "Mean Gradient Amplitude (S)",
            "Metrics (JSMA)",
        ],
        [
            max_epochs,
            lr,
            batch_size,
            temperature,
            num_samples,
            teacher_accuracy,
            calculate_mean_gradient_amplitude(
                art_model_t.model, mnist_data, mnist_targets, criterion, device=device
            ),
            evaluate_adversarial_metrics(
                art_model_t.model,
                expanded_data,
                expanded_labels,
                x_adv,
                device=device,
            ),
            student_accuracy,
            calculate_mean_gradient_amplitude(
                art_model_s.model, mnist_data, mnist_targets, criterion, device=device
            ),
            evaluate_adversarial_metrics(
                art_model_s.model,
                expanded_data,
                expanded_labels,
                x_adv_s,
                device=device,
            ),
        ],
    )

    # Display total running time for script
    LOGGER.info(f"Running Time (seconds): {round(time.perf_counter() - start_time, 2)}")
    LOGGER.info(
        f"Running Time (minutes): {round((time.perf_counter() - start_time) / 60, 2)}"
    )
    LOGGER.info("Script complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--n_channels", type=int, default=1, help="Number of channels.")
    parser.add_argument("--w", type=int, default=28, help="Width of the input images.")
    parser.add_argument("--h", type=int, default=28, help="Height of the input images.")
    parser.add_argument("--max_epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument(
        "--save_path",
        type=str,
        default="experiments_jsma",
        help="Path to save the experiment results.",
    )
    parser.add_argument(
        "--temperature", type=float, default=20, help="Temperature for soft labels."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples for adversarial attacks.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the training and evaluation on.",
    )
    args = parser.parse_args()
    # Call the main function with parsed arguments
    main(
        lr=args.lr,
        batch_size=args.batch_size,
        n_channels=args.n_channels,
        w=args.w,
        h=args.h,
        max_epochs=args.max_epochs,
        temperature=args.temperature,
        num_samples=args.num_samples,
        device=args.device,
        save_path=args.save_path,
    )
