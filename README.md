# Distillation experiments
The goal of this repository is to train a classifier model and expose it to adversarial samples. A distilled version of the model will then be created augmenting the latter to defend against adversarial perturbations following the work of [(Papernot et al., 2016)](https://arxiv.org/abs/1511.04508).

Notes: Crafting Adverserial samples [(Goodfellow et al., 2014)](https://arxiv.org/pdf/1412.6572)

Main branch: [Cifar 10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) with 3x32x32 samples and 3x64x64 upscaled samples.
Mnist branch: [Mnist dataset](http://yann.lecun.com/exdb/mnist/) with 1x28x28 samples.
