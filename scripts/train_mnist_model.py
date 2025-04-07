"""
Train a teacher model on MNIST, generate adversarial examples, and evaluate the models.

This script trains a teacher model on the MNIST dataset, generates adversarial examples using various attacks, 
and evaluates the models' performance on both clean and adversarial data.

To run the script you can use the following command, adjusting argumments as needed:
ipython scripts/train_mnist_model.py -- --lr 0.001 --batch_size 128 --max_epochs 20 --temperature 20 --num_samples 100 --device 'mps'

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
from processing.distillation import get_soft_labels
from processing.distillation import train_student
from evaluation.metrics import calculate_mean_gradient_amplitude


from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.attacks.evasion.deepfool import DeepFool
from art.attacks.evasion.pixel_threshold import PixelAttack

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

    dt_p = Path('data/mnist')

    # Specify classes as string and number of labels
    classes = [str(i) for i in range(10)]
    n_labels = len(classes)

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    # Load MNIST dataset
    trainset = datasets.MNIST(root=f'{dt_p.absolute()}/train', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = datasets.MNIST(root=f'{dt_p.absolute()}/test', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

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
    mnist_data_subset = mnist_data_shuffled[:num_samples]/255
    mnist_targets_subset = mnist_targets_shuffled[:num_samples]


    ## Teacher Model

    LOGGER.info("Training Teacher Model")

    # Train the teacher model
    teacher_losses = []

    for e in tqdm(range(max_epochs)):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            logits = teacher_model(images)
            # Compute loss
            loss = criterion(logits, labels)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        teacher_losses.append(loss.item())
        
        if e % 10 == 0 or e == max_epochs-1:
            LOGGER.info(f"Epoch {e}: {loss.item()}")
    
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
    original_accuracy = evaluate_model(
        art_model_t.model, mnist_data, mnist_targets, device=device
    )
    LOGGER.info(f"Test Accuracy: {original_accuracy:.2f}%")

    LOGGER.info(f"Mean Gradient Amplitude: {calculate_mean_gradient_amplitude(art_model_t.model, mnist_data, 
                                                                              mnist_targets, criterion, device=device)}")

    # Ensure teacher model is on CPU to create the attacks
    teacher_model.to('cpu')

    # Adversarial attacks
    # Generate Adversarial Examples from the Teacher Model
    LOGGER.info("Generating Adversarial Examples from Teacher Model:")

    LOGGER.info("Generating FSGM Adversarial Examples")
    attack = FastGradientMethod(estimator=art_model_t, eps=0.4, eps_step=0.1, batch_size=32, minimal=True, targeted=False, summary_writer=True)
    x_adv_fgm = attack.generate(x=mnist_data_subset, y=mnist_targets_subset)
    visualize_adversarial(mnist_data_subset, x_adv_fgm, mnist_targets_subset)
    show_difference(mnist_data_subset[0][0], x_adv_fgm[0][0], title="Fast-Gradient Method")

    LOGGER.info("Generating DeepFool Adversarial Examples")
    attack = DeepFool(classifier=art_model_t, epsilon=0.001, max_iter=50, batch_size=32)
    x_adv_deepfool = attack.generate(x=mnist_data_subset, y=mnist_targets_subset)
    visualize_adversarial(mnist_data_subset, x_adv_deepfool, mnist_targets_subset)
    show_difference(mnist_data_subset[0][0], x_adv_deepfool[0][0], title="Deepfool Method")

    LOGGER.info("Generating One Pixel Adversarial Examples")
    attack = PixelAttack(classifier=art_model_t, th=5, es=1, max_iter=50)
    x_adv_pixel = attack.generate(x=mnist_data_subset, y=mnist_targets_subset)
    visualize_adversarial(mnist_data_subset, x_adv_pixel, mnist_targets_subset)
    show_difference(mnist_data_subset[0][0], x_adv_pixel[0][0], title="Pixel Method")

    # Transfer model back to device
    teacher_model.to(device)

    # Evaluate the teacher model on adversarial examples
    LOGGER.info("Evaluating Teacher Model on Teacher-based Adversarial Examples:")
    # FSGM
    LOGGER.info(evaluate_adversarial_metrics(art_model_t.model, mnist_data_subset, mnist_targets_subset, x_adv_fgm, device=device))
    # DeepFool
    LOGGER.info(evaluate_adversarial_metrics(art_model_t.model, mnist_data_subset, mnist_targets_subset, x_adv_deepfool, device=device))
    # One Pixel Attack
    LOGGER.info(evaluate_adversarial_metrics(art_model_t.model, mnist_data_subset, mnist_targets_subset, x_adv_pixel, device=device))



    ## Student Model

    LOGGER.info("Training Student Model using Defensive Distillation")
    # Train the student model using knowledge distillation
    train_student(teacher_model, student_model, trainloader, temp=temperature, alpha=0.7, epochs=max_epochs, lr=lr, device=device)

    # Wrap in ART PyTorchClassifier
    art_model_s = PyTorchClassifier(
        model=student_model,
        clip_values=(0, 1),  # Min and Max pixel values (normalize if needed)
        loss=criterion,
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=10
    )

    # Evaluate model on entire testset
    original_accuracy = evaluate_model(
        art_model_s.model, mnist_data, mnist_targets, device=device
    )
    LOGGER.info(f"Test Accuracy: {original_accuracy:.2f}%")

    LOGGER.info(f"Mean Gradient Amplitude: {calculate_mean_gradient_amplitude(art_model_s.model, mnist_data, 
                                                                              mnist_targets, criterion, device=device)}")


    # Ensure teacher model is on CPU to create the attacks
    student_model.to('cpu')

    # Adversarial attacks
    # Generate Adversarial Examples from the Student Model
    LOGGER.info("Generating Adversarial Examples from Student Model:")

    LOGGER.info("Generating FSGM Adversarial Examples")
    attack = FastGradientMethod(estimator=art_model_s, eps=0.4, eps_step=0.1, batch_size=32, minimal=True, targeted=False, summary_writer=True)
    x_adv_fgm_s = attack.generate(x=mnist_data_subset, y=mnist_targets_subset)
    visualize_adversarial(mnist_data_subset, x_adv_fgm_s, mnist_targets_subset)
    show_difference(mnist_data_subset[0][0], x_adv_fgm_s[0][0], title="Fast-Gradient Method")

    LOGGER.info("Generating DeepFool Adversarial Examples")
    attack = DeepFool(classifier=art_model_s, epsilon=0.001, max_iter=50, batch_size=32)
    x_adv_deepfool_s = attack.generate(x=mnist_data_subset, y=mnist_targets_subset)
    visualize_adversarial(mnist_data_subset, x_adv_deepfool_s, mnist_targets_subset)
    show_difference(mnist_data_subset[0][0], x_adv_deepfool_s[0][0], title="Deepfool Method")

    LOGGER.info("Generating One Pixel Adversarial Examples")
    attack = PixelAttack(classifier=art_model_s, th=5, es=1, max_iter=50)
    x_adv_pixel_s = attack.generate(x=mnist_data_subset, y=mnist_targets_subset)
    visualize_adversarial(mnist_data_subset, x_adv_pixel_s, mnist_targets_subset)
    show_difference(mnist_data_subset[0][0], x_adv_pixel_s[0][0], title="Pixel Method")

    # Transfer model back to device
    student_model.to(device)

    # Evaluate the teacher model on adversarial examples
    LOGGER.info("Evaluating Teacher Model on Teacher-based Adversarial Examples:")
    # FSGM
    LOGGER.info(evaluate_adversarial_metrics(art_model_s.model, mnist_data_subset, mnist_targets_subset, x_adv_fgm_s, device=device))
    # DeepFool
    LOGGER.info(evaluate_adversarial_metrics(art_model_s.model, mnist_data_subset, mnist_targets_subset, x_adv_deepfool_s, device=device))
    # One Pixel Attack
    LOGGER.info(evaluate_adversarial_metrics(art_model_s.model, mnist_data_subset, mnist_targets_subset, x_adv_pixel_s, device=device))

    # Display total running time for script
    LOGGER.info(f"Running Time (seconds): {round(time.perf_counter() - start_time, 2)}")
    LOGGER.info(f"Running Time (minutes): {round((time.perf_counter() - start_time) / 60, 2)}")
    LOGGER.info("Script complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--n_channels", type=int, default=1, help="Number of channels.")
    parser.add_argument("--w", type=int, default=28, help="Width of the input images.")
    parser.add_argument("--h", type=int, default=28, help="Height of the input images.")
    parser.add_argument("--max_epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument("--temperature", type=float, default=20, help="Temperature for soft labels.")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples for adversarial attacks.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the training and evaluation on.")
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
    )