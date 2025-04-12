"""
Train a teacher model on Cifar-10, generate adversarial examples, and evaluate the models.

This script trains a teacher model on the Cifar-10 dataset, generates adversarial examples using various attacks,
and evaluates the models' performance on both clean and adversarial data.

To run the script you can use the following command, adjusting argumments as needed:
ipython scripts/train_cifar_jacobian_attack.py -- --lr 0.001 --batch_size 128 --max_epochs 2 \
--temperature 2 --num_samples 100 --device 'mps' --save_fig True --headless True

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
from processing.distillation import train_student, train_teacher, load_model

from evaluation.metrics import calculate_mean_gradient_amplitude, calculate_binned_gradient_amplitude
from utils.experiment_saver import save_experiment_results


from utils.jsma_generate import generate_adversarial_samples


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
    headless: bool,
    save_fig: bool,
):
    """
    Main function to train a teacher and a distilled studemt model on cifar, generate adversarial examples, and evaluate the models.

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
        headless (bool): If True, run in headless mode (no GUI).
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

    dt_p = Path("data/cifar10-32")

    # Specify classes as string and number of labels
    classes = [str(i) for i in range(10)]
    n_labels = len(classes)

    transform = transforms.Compose(
        [transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
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

    
    teacher_name = f"cifar_teacher_model_temp{temperature}_ep{max_epochs}_lr{lr}_batch{batch_size}"

    # Load the model
    teacher_model = load_model(
        Cifar10Net(input_size=w, temperature=temperature, raw_logits=True).to(device),
        device=device,
        load_path=save_path,
        model_name=teacher_name,
    )

    if teacher_model is None:
        # Define a simple CNN model for CIFAR-10 classification
        teacher_model = Cifar10Net(input_size=w, temperature=temperature, raw_logits=True).to(device)
        training = True
    else:
        training = False
    student_model = Cifar10Net(input_size=w, temperature=temperature, raw_logits=False).to(device)

    # Specify the loss function and optimizer for teacher model
    criterion = nn.CrossEntropyLoss()
    criterion_dist = nn.KLDivLoss(reduction="batchmean", log_target=True)

    optimizer = optim.SGD(teacher_model.parameters(), lr=lr, momentum=0.9)

    # Convert list to tensor first, then change to int and numpy
    cifar_targets = torch.tensor(testset.targets).int().numpy()
    # Convert data to tensor, permute to correct shape, and then convert to numpy
    cifar_data = torch.tensor(testset.data).permute(0, 3, 1, 2).float().numpy()

    # Select a small test subset to attack
    # adversarial attacks can be slow, so we only use a small subset of the test set
    # First shuffle
    indices = torch.randperm(len(testset.data))
    cifar_data_shuffled = cifar_data[indices]
    cifar_targets_shuffled = cifar_targets[indices]
    # Then select subsets
    cifar_data_subset = cifar_data_shuffled[:num_samples] / 255
    cifar_targets_subset = cifar_targets_shuffled[:num_samples]

    ## Teacher Model




    
    # If the model is not loaded (returns None), train and save it
    if training:
        LOGGER.info("\nTraining Teacher Model")

        # Train the teacher model
        train_teacher(
        teacher_model,
        trainloader,
        epochs=max_epochs,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_path=save_path,
        model_name=teacher_name,
    )


    # Set to evaluation mode
    teacher_model.eval()
    # Set model temperature to 1 after training
    teacher_model.temperature = 1.0
    # Apply softmax probabilities during inference
    teacher_model.raw_logits = False

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
        art_model_t.model, cifar_data/255, cifar_targets, device=device
    )
    LOGGER.info(f"Test Accuracy: {teacher_accuracy:.2f}%")
    
    # Criterion needs logits
    teacher_model.raw_logits = True

    LOGGER.info(
        f"Mean Gradient Amplitude: {calculate_mean_gradient_amplitude(art_model_t.model, cifar_data, cifar_targets, criterion, device=device)}"
    )
    LOGGER.info(
        f"Binned Mean Gradient Amplitude (as implemented in Paparnot et. al 2016): {calculate_binned_gradient_amplitude(art_model_t.model, cifar_data, cifar_targets, criterion, device=device)}"
    )

    # Apply softmax probabilities during inference
    teacher_model.raw_logits = False

    # Ensure teacher model is not on mps to create the attacks
    if device == "mps":
        teacher_model.to("cpu")

    # Adversarial attacks
    # Generate Adversarial Examples from the Teacher Model
    LOGGER.info("\nGenerating Adversarial Examples from Teacher Model:")

    LOGGER.info("Generating Jacobian-Saliency Adversarial Examples")

    expanded_data, expanded_labels, x_adv, y_adv = generate_adversarial_samples(cifar_data_subset, cifar_targets_subset, art_model_t, theta=0.4, gamma=0.5, batch_size=32, device=device)

    if not headless:
        visualize_adversarial(expanded_data, x_adv, expanded_labels, save_fig=save_fig,
                          save_path='jsma_t.png')
        show_difference(
            expanded_data[0][0], x_adv[0][0], title="Jacobian-Saliency Map Method", save_fig=save_fig,
            save_path='jsma_diff_t.png'
        )
        
    
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
    if not headless:
        visualize_adversarial(cifar_data_subset, x_adv_fgm, cifar_targets_subset)
        show_difference(
            cifar_data_subset[0][0], x_adv_fgm[0][0], title="Fast-Gradient Method"
        )
        

    LOGGER.info("Generating DeepFool Adversarial Examples")
    attack = DeepFool(classifier=art_model_t, epsilon=0.001, max_iter=50, batch_size=32)
    x_adv_deepfool = attack.generate(x=cifar_data_subset, y=cifar_targets_subset)
    if not headless:
        visualize_adversarial(cifar_data_subset, x_adv_deepfool, cifar_targets_subset)
        show_difference(
            cifar_data_subset[0][0], x_adv_deepfool[0][0], title="Deepfool Method"
        )

    LOGGER.info("Generating One Pixel Adversarial Examples")
    attack = PixelAttack(classifier=art_model_t, th=5, es=1, max_iter=50)
    x_adv_pixel = attack.generate(x=cifar_data_subset, y=cifar_targets_subset)
    if not headless:
        visualize_adversarial(cifar_data_subset, x_adv_pixel, cifar_targets_subset)
        show_difference(cifar_data_subset[0][0], x_adv_pixel[0][0], title="Pixel Method")

    
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
    LOGGER.info(
        evaluate_adversarial_metrics(
            art_model_t.model,
            cifar_data_subset,
            cifar_targets_subset,
            x_adv_fgm,
            device=device,
        )
    )
    # DeepFool
    LOGGER.info(
        evaluate_adversarial_metrics(
            art_model_t.model,
            cifar_data_subset,
            cifar_targets_subset,
            x_adv_deepfool,
            device=device,
        )
    )
    # One Pixel Attack
    LOGGER.info(
        evaluate_adversarial_metrics(
            art_model_t.model,
            cifar_data_subset,
            cifar_targets_subset,
            x_adv_pixel,
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
        criterion=criterion_dist,
        epochs=max_epochs,
        lr=0.0001,
        temperature=temperature,
        device=device,
    )

    # Set to evaluation mode
    student_model.eval()   
    # Set model temperature to 1 after training
    student_model.temperature = 1.0    

    # Wrap in ART PyTorchClassifier
    art_model_s = PyTorchClassifier(
        model=student_model,
        clip_values=(0, 1),  # Min and Max pixel values (normalize if needed)
        loss=criterion_dist,
        optimizer=optimizer,
        input_shape=(n_channels, w, h),
        nb_classes=10,
    )

    # Evaluate model on entire testset
    student_accuracy = evaluate_model(
        art_model_s.model, cifar_data/255, cifar_targets, device=device
    )
    LOGGER.info(f"Test Accuracy: {student_accuracy:.2f}%")

    # Criterion needs logits
    student_model.raw_logits = True 

    LOGGER.info(
        f"Mean Gradient Amplitude: {calculate_mean_gradient_amplitude(art_model_s.model, cifar_data, cifar_targets, criterion, device=device)}"
    )
    LOGGER.info(
        f"Binned Mean Gradient Amplitude (as implemented in Paparnot et. al 2016): {calculate_binned_gradient_amplitude(art_model_s.model, cifar_data, cifar_targets, criterion, device=device)}"
    )

    # Adversarial attacks
    # Generate Adversarial Examples from the Student Model
    LOGGER.info("\nGenerating Adversarial Examples from Student Model:")

    LOGGER.info("Generating Jacobian-Saliency Adversarial Examples")

    expanded_data, expanded_labels, x_adv_s, y_adv = generate_adversarial_samples(cifar_data_subset, cifar_targets_subset, art_model_s, theta=0.4, gamma=0.5, batch_size=32, device=device)


    if not headless:
        visualize_adversarial(expanded_data, x_adv_s, expanded_labels, save_fig=save_fig, save_path="jsma_s.png")
        show_difference(
            expanded_data[0][0], x_adv_s[0][0], title="Jacobian-Saliency Method", save_fig=save_fig, 
            save_path="jsma_diff_s.png"
        )

    LOGGER.info("Generating FSGM Adversarial Examples")
    attack = FastGradientMethod(
        estimator=art_model_s,
        eps=0.4,
        eps_step=0.1,
        batch_size=32,
        minimal=True,
        targeted=False,
    )
    x_adv_fgm_s = attack.generate(x=cifar_data_subset, y=cifar_targets_subset)
    if not headless:
        visualize_adversarial(cifar_data_subset, x_adv_fgm_s, cifar_targets_subset)
        show_difference(
            cifar_data_subset[0][0], x_adv_fgm_s[0][0], title="Fast-Gradient Method"
        )
        

    LOGGER.info("Generating DeepFool Adversarial Examples")
    attack = DeepFool(classifier=art_model_s, epsilon=0.001, max_iter=50, batch_size=32)
    x_adv_deepfool_s = attack.generate(x=cifar_data_subset, y=cifar_targets_subset)
    if not headless:
        visualize_adversarial(cifar_data_subset, x_adv_deepfool_s, cifar_targets_subset)
        show_difference(
            cifar_data_subset[0][0], x_adv_deepfool_s[0][0], title="Deepfool Method"
        )

    LOGGER.info("Generating One Pixel Adversarial Examples")
    attack = PixelAttack(classifier=art_model_s, th=5, es=1, max_iter=50)
    x_adv_pixel_s = attack.generate(x=cifar_data_subset, y=cifar_targets_subset)
    if not headless:
        visualize_adversarial(cifar_data_subset, x_adv_pixel_s, cifar_targets_subset)
        show_difference(cifar_data_subset[0][0], x_adv_pixel_s[0][0], title="Pixel Method")

    # Transfer model back to device
    student_model.to(device)

    # Apply softmax probabilities during inference
    student_model.raw_logits = False

    # Evaluate the teacher model on adversarial examples
    LOGGER.info("Evaluating Student Model on Student-based Adversarial Examples:")
    
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
            "Metrics (FSGM)",
            "Metrics (DeepFool)",
            "Metrics (Pixel)",
            "Accuracy (S)",
            "Mean Gradient Amplitude (S)",
            "Metrics (JSMA)",
            "Metrics (FSGM)",
            "Metrics (DeepFool)",
            "Metrics (Pixel)",
        ],
        [
            max_epochs,
            lr,
            batch_size,
            temperature,
            num_samples,
            teacher_accuracy,
            calculate_mean_gradient_amplitude(
                art_model_t.model, cifar_data, cifar_targets, criterion, device=device
            ),
            evaluate_adversarial_metrics(
                art_model_t.model,
                expanded_data,
                expanded_labels,
                x_adv,
                device=device,
            ),
            evaluate_adversarial_metrics(
                art_model_t.model,
                cifar_data_subset,
                cifar_targets_subset,
                x_adv_fgm,
                device=device,
            ),
            evaluate_adversarial_metrics(
                art_model_t.model,
                cifar_data_subset,
                cifar_targets_subset,
                x_adv_deepfool,
                device=device,
            ),
            evaluate_adversarial_metrics(
                art_model_t.model,
                cifar_data_subset,
                cifar_targets_subset,
                x_adv_pixel,
                device=device,
            ),
            student_accuracy,
            calculate_mean_gradient_amplitude(
                art_model_s.model, cifar_data, cifar_targets, criterion, device=device
            ),
            evaluate_adversarial_metrics(
                art_model_s.model,
                expanded_data,
                expanded_labels,
                x_adv_s,
                device=device,
            ),
            evaluate_adversarial_metrics(
                art_model_s.model,
                cifar_data_subset,
                cifar_targets_subset,
                x_adv_fgm_s,
                device=device,
            ),
            evaluate_adversarial_metrics(
                art_model_s.model,
                cifar_data_subset,
                cifar_targets_subset,
                x_adv_deepfool_s,
                device=device,
            ),
            evaluate_adversarial_metrics(
                art_model_s.model,
                cifar_data_subset,
                cifar_targets_subset,
                x_adv_pixel_s,
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
    parser.add_argument("--n_channels", type=int, default=3, help="Number of channels.")
    parser.add_argument("--w", type=int, default=32, help="Width of the input images.")
    parser.add_argument("--h", type=int, default=32, help="Height of the input images.")
    parser.add_argument("--max_epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument(
        "--save_path",
        type=str,
        default="experiments/cifar_final",
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
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the training and evaluation on.",
    )
    parser.add_argument(
        "--headless",
        type=bool,
        default=False,
        help="Whether to run in headless mode (no GUI).",
    )
    parser.add_argument(
        "--save_fig",
        type=bool,
        default=False,
        help="Whether to save figures.",
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
        headless=args.headless,
        save_fig=args.save_fig
    )
