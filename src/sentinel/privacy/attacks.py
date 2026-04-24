import torch
import torch.nn as nn
import random

class MembershipInferenceAttacker(nn.Module):
    """
    A simple MLP that tries to guess if a data point was in the training set
    based on the target model's output probabilities.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def calculate_mia_score(target_model, X_train_sample, X_test_sample):
    if target_model is None:
        return torch.tensor([0.5])

    target_model.eval()
    with torch.no_grad():
        train_probs = torch.softmax(target_model(X_train_sample), dim=1)
        test_probs = torch.softmax(target_model(X_test_sample), dim=1)
        
        # Calculate Shannon Entropy: -Sum(p * log(p))
        # Adding 1e-10 to avoid log(0) which results in NaN
        train_entropy = -torch.sum(train_probs * torch.log(train_probs + 1e-10), dim=1).mean().item()
        test_entropy = -torch.sum(test_probs * torch.log(test_probs + 1e-10), dim=1).mean().item()
        
    # If the model is 'less surprised' (lower entropy) by the training data, 
    # that is a privacy leak.
    entropy_gap = test_entropy - train_entropy
    
    # Scaling factor (e.g., 5.0) to make subtle shifts visible in MLflow
    # This maps a 0.02 entropy difference to a 0.60 leakage score
    leakage = 0.5 + (max(0, entropy_gap) * 5.0) 
    
    return torch.tensor([min(1.0, leakage)])



from opacus import PrivacyEngine

def train_private_model(model, train_loader, optimizer, target_epsilon, epochs=3):
    """
    Wraps a standard PyTorch training setup with Differential Privacy.
    """
    # If we are just testing the 'plumbing' with No-Models
    if model is None:
        print("Mocking Private Training (No model provided)")
        return None, None, None, None

    privacy_engine = PrivacyEngine()
    
    # DP-SGD transformation
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=target_epsilon,
        target_delta=1e-5,
        epochs=epochs,
        max_grad_norm=1.0,
    )
    
    print(f"Privacy Shield Active: Using Sigma {optimizer.noise_multiplier}")
    return model, optimizer, train_loader, privacy_engine

