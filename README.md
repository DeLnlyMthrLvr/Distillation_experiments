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
| `train_cifar_jacobian_attack.py`   | Trains a CIFAR-10 teacher if needed, and runs JSMA |
| `train_mnist_jacobian_attack.py`   | Same as above, for MNIST |

---

## ⚙️ Command-Line Arguments

e.g. ipython scripts/train_cifar_jacobian_attack.py -- --lr 0.001 --batch_size 256 --max_epochs 50 --device 'mps'

### CIFAR-10

| Argument       | Default                          | Description |
|----------------|----------------------------------|-------------|
| `--lr`         | `0.001`                          | Learning rate |
| `--batch_size` | `128`                            | Batch size |
| `--n_channels` | `3`                              | Number of channels |
| `--w`          | `32`                             | Image width |
| `--h`          | `32`                             | Image height |
| `--max_epochs` | `50`                             | Training epochs |
| `--save_path`  | `"experiments/cifar_jacobian_exper"` | Where to store experiment results |
| `--temperature`| `20`                             | Softmax temperature |
| `--num_samples`| `100`                            | Number of adversarial samples |
| `--device`     | `"cpu"`                          | Device to use |
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
| `--device`     | `"cpu"`        | Device to use |

---

## 📖 Paper Reference

> **Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks**  
> Nicolas Papernot, Patrick McDaniel, Xi Wu, Somesh Jha, Ananthram Swami  
> [IEEE S&P 2016](https://arxiv.org/abs/1511.04508)


---

