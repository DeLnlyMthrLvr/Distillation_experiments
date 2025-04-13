# 🛡️ REDACT: REsilience through DISTillation Against Complex Threats

## 🎯 Goal

This repository explores **defensive distillation** as a technique to improve the robustness of DNNs against adversarial attacks. We train image classifiers on **CIFAR-10** and **MNIST**, expose them to adversarial samples, and apply distillation as mention in [Papernot et al., 2016](https://arxiv.org/abs/1511.04508). 

## 🧪 Overview

We compare:
- A *teacher model** trained normally
- A **student model** trained with soft labels at various softmax temperatures

### 🔥 Attacks included:
- **FGSM** (baseline, gradient-based)
- **DeepFool**
- **JSMA** (with increased gamma; max ratio of modified pixels)
- **One-Pixel Attack** (non-gradient-based)

---

## 🌿 Branches

| Branch   | Dataset | Image Shape            | Notes                                      |
|----------|---------|------------------------|--------------------------------------------|
| `main`   | CIFAR-10| `3x32x32`              | RGB images                                 |
| `mnist`  | MNIST   | `1x28x28`              | Grayscale images                           |

---

## 📜 Scripts

For training and evaluation. Located in `scripts/`:

| Script                              | Description |
|-------------------------------------|-------------|
| `train_cifar_model.py`             | Trains a CIFAR-10 model if needed, and runs FGSM, DeepFool, and One-Pixel attacks |
| `train_mnist_model.py`             | Same as above, for MNIST |
| `train_cifar_all.py`   | Trains a CIFAR-10 teacher if needed, and runs JSMA FGSM, DeepFool, and One-Pixel attacks |

---

## ⚙️ Command-Line Arguments

e.g. ipython scripts/train_cifar_jacobian_attack.py -- --lr 0.001 --batch_size 256 --max_epochs 50 --device 'mps'

### CIFAR-10

| Argument       | Default                          | Description |
|----------------|----------------------------------|-------------|
| `--lr`         | `0.01`                          | Learning rate |
| `--batch_size` | `128`                            | Batch size |
| `--n_channels` | `3`                              | Number of channels |
| `--w`          | `32`                             | Image width |
| `--h`          | `32`                             | Image height |
| `--max_epochs` | `50`                             | Training epochs |
| `--save_path`  | `"experiments/cifar_jacobian_exper"` | Where to store experiment results |
| `--temperature`| `20`                             | Softmax temperature |
| `--num_samples`| `100`                            | Number of adversarial samples |
| `--device`     | `"cpu"`                   | Device to use ('cpu', 'cuda', 'mps', etc.) |
| `--headless`   | `False`                           |Headless (No gui)|
| `--save_fig`   | `False`                          | Save figures or not |

### MNIST

| Argument       | Default        | Description |
|----------------|----------------|-------------|
| `--lr`         | `0.001`        | Learning rate |
| `--batch_size` | `128`          | Batch size |
| `--n_channels` | `1`            | Number of channels |
| `--w`          | `28`           | Image width |
| `--h`          | `28`           | Image height |
| `--max_epochs` | `50`           | Training epochs |
| `--save_path`  | `"experiments/"` | Where to store experiment results |
| `--temperature`| `20`           | Softmax temperature |
| `--num_samples`| `100`          | Number of adversarial samples |
| `--device`     | `"cpu"`        | Device to use ('cpu', 'cuda', 'mps', etc.) |

---

## ⚙️ Environment Setup

Below are instructions in order to setup all pre-requisites before running the code.

### 🧰 1. Create the environment and install dependencies

Run `make setup` in the root directory. This will:
- Create a virtual environment in a folder named `env`
- Upgrade `pip`, `setuptools`, and `wheel`
- Install dependencies listed in `requirements.txt`
- Install `matplotlib` using a prebuilt binary to avoid OS-specific build errors
- Install `ipython` for running Python scripts via terminal

### 🧠 2. Activate the virtual environment

After setup, activate the environment:
- On macOS or Linux, run `source env/bin/activate`
- On Windows (Command Prompt), run `env\Scripts\activate.bat`
- On Windows (PowerShell), run `env\Scripts\Activate.ps1`

### 🧠 3. Install PyTorch (CPU-only, manual step)

To avoid compatibility issues with CUDA and GPU drivers, install the CPU-only version of PyTorch manually. Run `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` inside the activated environment.

### ✅ 4. Run a quick test

To verify everything works correctly, run `make test`. This executes a minimal training script (`scripts/train_mnist_jacobian_attack.py`) with the arguments `--max_epochs 2` and `--num_samples 2`. It performs a short training pass and generates adversarial examples to ensure setup is working properly.

### 🧹 5. Clean the environment

To delete the environment and start fresh, run `make clean`. This removes the `env` folder and all installed dependencies.


## 📖 Paper Reference

> **Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks**  
> Nicolas Papernot, Patrick McDaniel, Xi Wu, Somesh Jha, Ananthram Swami  
> [IEEE S&P 2016](https://arxiv.org/abs/1511.04508)


---

