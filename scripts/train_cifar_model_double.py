"""
Train a teacher, a student, and a student-student model on Cifar-10, generate adversarial examples,
and evaluate the models.

This script trains a teacher model on the CIFAR-10 dataset, generates adversarial examples using various attacks,
evaluates the teacher and student models, then trains a student-student model using defensive distillation from the student.
It subsequently assesses all the models' performances on both clean and adversarial data.

To run the script you can use the following command, adjusting arguments as needed:
ipython scripts/train_cifar_model.py -- --lr 0.001 --batch_size 256 --max_epochs 20 --temperature 20 --num_samples 100 --device 'mps'
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

from models.cifar_models import Cifar10Net
from evaluation.metrics import evaluate_adversarial_metrics
from evaluation.metrics import evaluate_model
from processing.visualize import show_difference
from processing.visualize import visualize_adversarial
from processing.distillation import train_student, train_teacher
from evaluation.metrics import calculate_mean_gradient_amplitude
from utils.experiment_saver import save_experiment_results

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
    save_path: str,
):
    """
    Main function to train teacher, student, and a distilled student-student model on CIFAR-10,
    generate adversarial examples, and evaluate each model.

    Args:
        lr (float): Learning rate.
        batch_size (int): Batch size.
        n_channels (int): Number of channels in the input images.
        w (int): Width of the input images.
        h (int): Height of the input images.
        max_epochs (int): Number of epochs to train each model.
        temperature (float): Temperature parameter for softmax scaling / distillation.
        num_samples (int): Number of samples to use for generating adversarial examples.
        device (str): Device to run the training and evaluation on.
        save_path (str): Path to save the experiment results.
    """
    ssl._create_default_https_context = ssl._create_stdlib_context
    start_time = time.perf_counter()

    # Ensure reproducibility
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    LOGGER.info(f"Playing on {device}")

    dt_p = Path("data/cifar10-32")

    # Specify classes as string and number of labels
    classes = [str(i) for i in range(10)]
    n_labels = len(classes)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # Load CIFAR dataset
    trainset = datasets.CIFAR10(
        root=f"{dt_p.absolute()}/train", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    testset = datasets.CIFAR10(
        root=f"{dt_p.absolute()}/test", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    # Define teacher and student models (simple CNN for CIFAR-10 classification)
    teacher_model = Cifar10Net(input_size=w, temperature=temperature).to(device)
    student_model = Cifar10Net(input_size=w, temperature=temperature).to(device)

    # Set loss and optimizer for teacher training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(teacher_model.parameters(), lr=lr)

    # Prepare test data and targets
    cifar_targets = torch.tensor(testset.targets).int().numpy()
    cifar_data = torch.tensor(testset.data).permute(0, 3, 1, 2).float().numpy()

    # Create a small test subset for adversarial attacks
    indices = torch.randperm(len(testset.data))
    cifar_data_shuffled = cifar_data[indices]
    cifar_targets_shuffled = cifar_targets[indices]
    cifar_data_subset = cifar_data_shuffled[:num_samples] / 255
    cifar_targets_subset = cifar_targets_shuffled[:num_samples]

    ## Teacher Model

    LOGGER.info("\nTraining Teacher Model")
    teacher_losses = train_teacher(
        teacher_model,
        trainloader,
        epochs=max_epochs,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    art_model_t = PyTorchClassifier(
        model=teacher_model,
        clip_values=(0, 1),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(n_channels, w, h),
        nb_classes=n_labels,
    )

    teacher_accuracy = evaluate_model(
        art_model_t.model, cifar_data, cifar_targets, device=device
    )
    LOGGER.info(f"Test Accuracy (Teacher): {teacher_accuracy:.2f}%")
    LOGGER.info(
        f"Mean Gradient Amplitude (Teacher): {calculate_mean_gradient_amplitude(art_model_t.model, cifar_data, cifar_targets, criterion, device=device)}"
    )

    # Move teacher model to CPU if using mps for attack creation
    if device == "mps":
        teacher_model.to("cpu")

    # Adversarial attacks for Teacher Model
    LOGGER.info("\nGenerating Adversarial Examples from Teacher Model:")

    LOGGER.info("Generating FSGM Adversarial Examples")
    attack = FastGradientMethod(
        estimator=art_model_t,
        eps=0.4,
        eps_step=0.1,
        batch_size=32,
        minimal=True,
        targeted=False,
        summary_writer=True,
    )
    x_adv_fgm = attack.generate(x=cifar_data_subset, y=cifar_targets_subset)
    visualize_adversarial(cifar_data_subset, x_adv_fgm, cifar_targets_subset)
    show_difference(
        cifar_data_subset[0][0], x_adv_fgm[0][0], title="Fast-Gradient Method"
    )

    LOGGER.info("Generating DeepFool Adversarial Examples")
    attack = DeepFool(classifier=art_model_t, epsilon=0.001, max_iter=50, batch_size=32)
    x_adv_deepfool = attack.generate(x=cifar_data_subset, y=cifar_targets_subset)
    visualize_adversarial(cifar_data_subset, x_adv_deepfool, cifar_targets_subset)
    show_difference(
        cifar_data_subset[0][0], x_adv_deepfool[0][0], title="Deepfool Method"
    )

    LOGGER.info("Generating One Pixel Adversarial Examples")
    attack = PixelAttack(classifier=art_model_t, th=5, es=1, max_iter=50)
    x_adv_pixel = attack.generate(x=cifar_data_subset, y=cifar_targets_subset)
    visualize_adversarial(cifar_data_subset, x_adv_pixel, cifar_targets_subset)
    show_difference(cifar_data_subset[0][0], x_adv_pixel[0][0], title="Pixel Method")

    teacher_model.to(device)

    LOGGER.info("Evaluating Teacher Model on Teacher-based Adversarial Examples:")
    LOGGER.info(
        evaluate_adversarial_metrics(
            art_model_t.model, cifar_data_subset, cifar_targets_subset, x_adv_fgm, device=device
        )
    )
    LOGGER.info(
        evaluate_adversarial_metrics(
            art_model_t.model, cifar_data_subset, cifar_targets_subset, x_adv_deepfool, device=device
        )
    )
    LOGGER.info(
        evaluate_adversarial_metrics(
            art_model_t.model, cifar_data_subset, cifar_targets_subset, x_adv_pixel, device=device
        )
    )

    ## Student Model

    LOGGER.info("\nTraining Student Model using Defensive Distillation")
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

    # Note: Correcting input_shape to match CIFAR-10 dimensions.
    optimizer_student = optim.AdamW(student_model.parameters(), lr=lr)
    art_model_s = PyTorchClassifier(
        model=student_model,
        clip_values=(0, 1),
        loss=criterion,
        optimizer=optimizer_student,
        input_shape=(n_channels, w, h),
        nb_classes=n_labels,
    )

    student_accuracy = evaluate_model(
        art_model_s.model, cifar_data, cifar_targets, device=device
    )
    LOGGER.info(f"Test Accuracy (Student): {student_accuracy:.2f}%")
    LOGGER.info(
        f"Mean Gradient Amplitude (Student): {calculate_mean_gradient_amplitude(art_model_s.model, cifar_data, cifar_targets, criterion, device=device)}"
    )

    if device == "mps":
        student_model.to("cpu")

    LOGGER.info("\nGenerating Adversarial Examples from Student Model:")

    LOGGER.info("Generating FSGM Adversarial Examples")
    attack = FastGradientMethod(
        estimator=art_model_s,
        eps=0.4,
        eps_step=0.1,
        batch_size=32,
        minimal=True,
        targeted=False,
        summary_writer=True,
    )
    x_adv_fgm_s = attack.generate(x=cifar_data_subset, y=cifar_targets_subset)
    visualize_adversarial(cifar_data_subset, x_adv_fgm_s, cifar_targets_subset)
    show_difference(
        cifar_data_subset[0][0], x_adv_fgm_s[0][0], title="Fast-Gradient Method"
    )

    LOGGER.info("Generating DeepFool Adversarial Examples")
    attack = DeepFool(classifier=art_model_s, epsilon=0.001, max_iter=50, batch_size=32)
    x_adv_deepfool_s = attack.generate(x=cifar_data_subset, y=cifar_targets_subset)
    visualize_adversarial(cifar_data_subset, x_adv_deepfool_s, cifar_targets_subset)
    show_difference(
        cifar_data_subset[0][0], x_adv_deepfool_s[0][0], title="Deepfool Method"
    )

    LOGGER.info("Generating One Pixel Adversarial Examples")
    attack = PixelAttack(classifier=art_model_s, th=5, es=1, max_iter=50)
    x_adv_pixel_s = attack.generate(x=cifar_data_subset, y=cifar_targets_subset)
    visualize_adversarial(cifar_data_subset, x_adv_pixel_s, cifar_targets_subset)
    show_difference(
        cifar_data_subset[0][0], x_adv_pixel_s[0][0], title="Pixel Method"
    )

    student_model.to(device)

    LOGGER.info("Evaluating Student Model on Student-based Adversarial Examples:")
    LOGGER.info(
        evaluate_adversarial_metrics(
            art_model_s.model, cifar_data_subset, cifar_targets_subset, x_adv_fgm_s, device=device
        )
    )
    LOGGER.info(
        evaluate_adversarial_metrics(
            art_model_s.model, cifar_data_subset, cifar_targets_subset, x_adv_deepfool_s, device=device
        )
    )
    LOGGER.info(
        evaluate_adversarial_metrics(
            art_model_s.model, cifar_data_subset, cifar_targets_subset, x_adv_pixel_s, device=device
        )
    )

    ## Student-Student Model

    LOGGER.info("\nTraining Student-Student Model using Defensive Distillation from the Student Model")
    student_student_model = Cifar10Net(input_size=w, temperature=temperature).to(device)
    train_student(
        student_model,  # use the trained student as teacher
        student_student_model,
        trainloader,
        criterion=criterion,
        epochs=max_epochs,
        lr=lr,
        temperature=temperature,
        device=device,
    )

    optimizer_ss = optim.AdamW(student_student_model.parameters(), lr=lr)
    art_model_ss = PyTorchClassifier(
        model=student_student_model,
        clip_values=(0, 1),
        loss=criterion,
        optimizer=optimizer_ss,
        input_shape=(n_channels, w, h),
        nb_classes=n_labels,
    )

    student_student_accuracy = evaluate_model(
        art_model_ss.model, cifar_data, cifar_targets, device=device
    )
    LOGGER.info(f"Test Accuracy (Student-Student): {student_student_accuracy:.2f}%")
    LOGGER.info(
        f"Mean Gradient Amplitude (Student-Student): {calculate_mean_gradient_amplitude(art_model_ss.model, cifar_data, cifar_targets, criterion, device=device)}"
    )

    if device == "mps":
        student_student_model.to("cpu")

    LOGGER.info("\nGenerating Adversarial Examples from Student-Student Model:")

    LOGGER.info("Generating FSGM Adversarial Examples")
    attack = FastGradientMethod(
        estimator=art_model_ss,
        eps=0.4,
        eps_step=0.1,
        batch_size=32,
        minimal=True,
        targeted=False,
        summary_writer=True,
    )
    x_adv_fgm_ss = attack.generate(x=cifar_data_subset, y=cifar_targets_subset)
    visualize_adversarial(cifar_data_subset, x_adv_fgm_ss, cifar_targets_subset)
    show_difference(
        cifar_data_subset[0][0], x_adv_fgm_ss[0][0], title="Fast-Gradient Method"
    )

    LOGGER.info("Generating DeepFool Adversarial Examples")
    attack = DeepFool(classifier=art_model_ss, epsilon=0.001, max_iter=50, batch_size=32)
    x_adv_deepfool_ss = attack.generate(x=cifar_data_subset, y=cifar_targets_subset)
    visualize_adversarial(cifar_data_subset, x_adv_deepfool_ss, cifar_targets_subset)
    show_difference(
        cifar_data_subset[0][0], x_adv_deepfool_ss[0][0], title="Deepfool Method"
    )

    LOGGER.info("Generating One Pixel Adversarial Examples")
    attack = PixelAttack(classifier=art_model_ss, th=5, es=1, max_iter=50)
    x_adv_pixel_ss = attack.generate(x=cifar_data_subset, y=cifar_targets_subset)
    visualize_adversarial(cifar_data_subset, x_adv_pixel_ss, cifar_targets_subset)
    show_difference(
        cifar_data_subset[0][0], x_adv_pixel_ss[0][0], title="Pixel Method"
    )

    student_student_model.to(device)

    LOGGER.info("Evaluating Student-Student Model on Student-Student-based Adversarial Examples:")
    LOGGER.info(
        evaluate_adversarial_metrics(
            art_model_ss.model, cifar_data_subset, cifar_targets_subset, x_adv_fgm_ss, device=device
        )
    )
    LOGGER.info(
        evaluate_adversarial_metrics(
            art_model_ss.model, cifar_data_subset, cifar_targets_subset, x_adv_deepfool_ss, device=device
        )
    )
    LOGGER.info(
        evaluate_adversarial_metrics(
            art_model_ss.model, cifar_data_subset, cifar_targets_subset, x_adv_pixel_ss, device=device
        )
    )

    ## Extended Cross-Result Evaluation

    LOGGER.info("\nCross-Result Evaluation:")

    # Student evaluated on Teacher-based adversarial examples (existing)
    LOGGER.info("Evaluating Student Model on Teacher-based Adversarial Examples:")
    LOGGER.info(
        evaluate_adversarial_metrics(
            art_model_s.model, cifar_data_subset, cifar_targets_subset, x_adv_fgm, device=device
        )
    )
    LOGGER.info(
        evaluate_adversarial_metrics(
            art_model_s.model, cifar_data_subset, cifar_targets_subset, x_adv_deepfool, device=device
        )
    )
    LOGGER.info(
        evaluate_adversarial_metrics(
            art_model_s.model, cifar_data_subset, cifar_targets_subset, x_adv_pixel, device=device
        )
    )

    # Teacher evaluated on Student-based adversarial examples (existing)
    LOGGER.info("Evaluating Teacher Model on Student-based Adversarial Examples:")
    LOGGER.info(
        evaluate_adversarial_metrics(
            art_model_t.model, cifar_data_subset, cifar_targets_subset, x_adv_fgm_s, device=device
        )
    )
    LOGGER.info(
        evaluate_adversarial_metrics(
            art_model_t.model, cifar_data_subset, cifar_targets_subset, x_adv_deepfool_s, device=device
        )
    )
    LOGGER.info(
        evaluate_adversarial_metrics(
            art_model_t.model, cifar_data_subset, cifar_targets_subset, x_adv_pixel_s, device=device
        )
    )

    # New cross evaluations involving Student-Student Model
    LOGGER.info("Evaluating Student-Student Model on Teacher-based Adversarial Examples:")
    LOGGER.info(
        evaluate_adversarial_metrics(
            art_model_ss.model, cifar_data_subset, cifar_targets_subset, x_adv_fgm, device=device
        )
    )
    LOGGER.info(
        evaluate_adversarial_metrics(
            art_model_ss.model, cifar_data_subset, cifar_targets_subset, x_adv_deepfool, device=device
        )
    )
    LOGGER.info(
        evaluate_adversarial_metrics(
            art_model_ss.model, cifar_data_subset, cifar_targets_subset, x_adv_pixel, device=device
        )
    )

    LOGGER.info("Evaluating Student-Student Model on Student-based Adversarial Examples:")
    LOGGER.info(
        evaluate_adversarial_metrics(
            art_model_ss.model, cifar_data_subset, cifar_targets_subset, x_adv_fgm_s, device=device
        )
    )
    LOGGER.info(
        evaluate_adversarial_metrics(
            art_model_ss.model, cifar_data_subset, cifar_targets_subset, x_adv_deepfool_s, device=device
        )
    )
    LOGGER.info(
        evaluate_adversarial_metrics(
            art_model_ss.model, cifar_data_subset, cifar_targets_subset, x_adv_pixel_s, device=device
        )
    )

    LOGGER.info("Evaluating Teacher Model on Student-Student-based Adversarial Examples:")
    LOGGER.info(
        evaluate_adversarial_metrics(
            art_model_t.model, cifar_data_subset, cifar_targets_subset, x_adv_fgm_ss, device=device
        )
    )
    LOGGER.info(
        evaluate_adversarial_metrics(
            art_model_t.model, cifar_data_subset, cifar_targets_subset, x_adv_deepfool_ss, device=device
        )
    )
    LOGGER.info(
        evaluate_adversarial_metrics(
            art_model_t.model, cifar_data_subset, cifar_targets_subset, x_adv_pixel_ss, device=device
        )
    )

    LOGGER.info("Evaluating Student Model on Student-Student-based Adversarial Examples:")
    LOGGER.info(
        evaluate_adversarial_metrics(
            art_model_s.model, cifar_data_subset, cifar_targets_subset, x_adv_fgm_ss, device=device
        )
    )
    LOGGER.info(
        evaluate_adversarial_metrics(
            art_model_s.model, cifar_data_subset, cifar_targets_subset, x_adv_deepfool_ss, device=device
        )
    )
    LOGGER.info(
        evaluate_adversarial_metrics(
            art_model_s.model, cifar_data_subset, cifar_targets_subset, x_adv_pixel_ss, device=device
        )
    )

    # Save experiment results including the new student-student model metrics
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
            "Metrics (FSGM, T)",
            "Metrics (DeepFool, T)",
            "Metrics (Pixel, T)",
            "Accuracy (S)",
            "Mean Gradient Amplitude (S)",
            "Metrics (FSGM, S)",
            "Metrics (DeepFool, S)",
            "Metrics (Pixel, S)",
            "Accuracy (SS)",
            "Mean Gradient Amplitude (SS)",
            "Metrics (FSGM, SS)",
            "Metrics (DeepFool, SS)",
            "Metrics (Pixel, SS)",
        ],
        [
            max_epochs,
            lr,
            batch_size,
            temperature,
            num_samples,
            teacher_accuracy,
            calculate_mean_gradient_amplitude(art_model_t.model, cifar_data, cifar_targets, criterion, device=device),
            evaluate_adversarial_metrics(art_model_t.model, cifar_data_subset, cifar_targets_subset, x_adv_fgm, device=device),
            evaluate_adversarial_metrics(art_model_t.model, cifar_data_subset, cifar_targets_subset, x_adv_deepfool, device=device),
            evaluate_adversarial_metrics(art_model_t.model, cifar_data_subset, cifar_targets_subset, x_adv_pixel, device=device),
            student_accuracy,
            calculate_mean_gradient_amplitude(art_model_s.model, cifar_data, cifar_targets, criterion, device=device),
            evaluate_adversarial_metrics(art_model_s.model, cifar_data_subset, cifar_targets_subset, x_adv_fgm_s, device=device),
            evaluate_adversarial_metrics(art_model_s.model, cifar_data_subset, cifar_targets_subset, x_adv_deepfool_s, device=device),
            evaluate_adversarial_metrics(art_model_s.model, cifar_data_subset, cifar_targets_subset, x_adv_pixel_s, device=device),
            student_student_accuracy,
            calculate_mean_gradient_amplitude(art_model_ss.model, cifar_data, cifar_targets, criterion, device=device),
            evaluate_adversarial_metrics(art_model_ss.model, cifar_data_subset, cifar_targets_subset, x_adv_fgm_ss, device=device),
            evaluate_adversarial_metrics(art_model_ss.model, cifar_data_subset, cifar_targets_subset, x_adv_deepfool_ss, device=device),
            evaluate_adversarial_metrics(art_model_ss.model, cifar_data_subset, cifar_targets_subset, x_adv_pixel_ss, device=device),
        ],
    )

    LOGGER.info(f"Running Time (seconds): {round(time.perf_counter() - start_time, 2)}")
    LOGGER.info(f"Running Time (minutes): {round((time.perf_counter() - start_time) / 60, 2)}")
    LOGGER.info("Script complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--n_channels", type=int, default=3, help="Number of channels.")
    parser.add_argument("--w", type=int, default=32, help="Width of the input images.")
    parser.add_argument("--h", type=int, default=32, help="Height of the input images.")
    parser.add_argument("--max_epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument(
        "--save_path",
        type=str,
        default="experiments/cifar_results.csv",
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
