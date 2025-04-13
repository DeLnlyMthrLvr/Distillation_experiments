import torch
import torch.nn as nn
import torch.nn.functional as F


class Cifar10Net(nn.Module):
    def __init__(self, input_size=32, temperature=1, raw_logits=True):
        """
        A simple CNN model for CIFAR-10 classification.
        Args:
            input_size (int): The size of the input images (default is 28 for MNIST).
            temperature (float): Temperature parameter for softmax scaling.
        """
        super().__init__()
        self.conv1a = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1b = nn.Conv2d(64, 64, 3, padding=1)

        self.conv2a = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2b = nn.Conv2d(128, 128, 3, padding=2)

        self.pool = nn.MaxPool2d(2, 2)

        # Compute the number of features dynamically
        self._to_linear = self._get_conv_output(input_size)

        self.fc1 = nn.Linear(self._to_linear, 256)
        self.fc2 = nn.Linear(256, 10)

        self.flatten = nn.Flatten()
        self.activation = nn.LeakyReLU()
        self.l_activ = nn.ReLU()

        self.dropout = nn.Dropout(p=0.5)

        self.temperature = temperature
        self.raw_logits = raw_logits

    def _get_conv_output(self, size):
        """Helper function to compute the output size after convolutions"""
        x = torch.zeros(1, 3, size, size)  # Create a dummy tensor
        x = self.pool(F.relu(self.conv1b(F.relu(self.conv1a(x)))))
        x = self.pool(F.relu(self.conv2b(F.relu(self.conv2a(x)))))
        return x.numel()

    def forward(self, x):
        x = self.activation(self.conv1a(x))
        x = self.activation(self.conv1b(x))
        x = self.pool(x)

        x = self.activation(self.conv2a(x))
        x = self.activation(self.conv2b(x))
        x = self.pool(x)

        x = self.flatten(x)  # Flatten before FC layers
        x = self.dropout(x)
        x = self.l_activ(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        if self.raw_logits is False:
            x = F.log_softmax(
                x / self.temperature, dim=1
            )  # Use log_softmax for numerical stability
        return x
