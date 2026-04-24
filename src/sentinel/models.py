import torch
import torch.nn as nn

class AdultIncomeMLP(nn.Module):
    """
    A 3-layer Multi-Layer Perceptron for binary classification.
    Input: 14 features (from UCI Adult dataset)
    Output: 2 classes (<=50K, >50K)
    """
    def __init__(self, input_dim=14):
        super(AdultIncomeMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2) # Binary classification output
        )

    def forward(self, x):
        return self.network(x)