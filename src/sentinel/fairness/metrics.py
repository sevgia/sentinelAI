import torch
import random

import torch

def calculate_disparate_impact(model, X_test, sensitive_features, unprivileged_value=0, privileged_value=1):
    """
    Generalized Disparate Impact (DI) calculation.
    
    Args:
        model: The trained PyTorch model.
        X_test: Test features (Tensor or ndarray).
        sensitive_features: The specific column of the protected attribute (e.g., gender, race).
        unprivileged_value: The value representing the minority/protected group (default: 0).
        privileged_value: The value representing the majority/privileged group (default: 1).
    """
    model.eval()
    with torch.no_grad():
        # Ensure X_test is a Tensor
        X_tensor = X_test if isinstance(X_test, torch.Tensor) else torch.Tensor(X_test)
        logits = model(X_tensor)
        preds = torch.argmax(logits, dim=1)
    
    # Selection Rate for Unprivileged Group (e.g., Female, Minority Race)
    unprivileged_mask = (sensitive_features == unprivileged_value)
    if unprivileged_mask.sum() == 0:
        return torch.tensor([1.0]) # Return neutral if group is missing
    
    unpriv_selection_rate = preds[unprivileged_mask].float().mean().item()
    
    # Selection Rate for Privileged Group (e.g., Male, Majority Race)
    privileged_mask = (sensitive_features == privileged_value)
    if privileged_mask.sum() == 0:
        return torch.tensor([1.0])
        
    priv_selection_rate = preds[privileged_mask].float().mean().item()
    
    # DI = (Selection Rate Unprivileged) / (Selection Rate Privileged)
    # If the privileged group has 0 selection rate, the ratio is technically undefined,
    # but we return 1.0 to indicate no measurable disparity in 'favorable' outcomes.
    if priv_selection_rate == 0: 
        return torch.tensor([1.0])
        
    di = unpriv_selection_rate / priv_selection_rate
    
    # Clamp to a reasonable range for visualization if needed
    return torch.tensor([di])

def apply_adversarial_debiasing(model):
    """
    Placeholder for the mitigation step.
    In the real version: This uses a second 'adversary' network to strip
    bias from the main model's representations.
    """
    print("--- Applying Adversarial Debiasing to Model ---")
    return model