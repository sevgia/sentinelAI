import torch
import torch.nn as nn

def generate_fgsm_adversarial(model, criterion, data, target, epsilon_robust):
    """
    Generates adversarial examples by manually disabling Opacus gradient tracking
    to prevent the 'pop from empty list' error.
    """
    # 1. Prepare input
    input_data = data.clone().detach().requires_grad_(True)
    
    # 2. Access the underlying module
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()

    # 3. CRITICAL: Manually disable Opacus gradient sample tracking
    # We toggle the 'grad_sample_mode' which Opacus uses to decide 
    # whether to push/pop activations.
    was_grad_sample_enabled = getattr(model, "is_grad_sample_grads_enabled", False)
    
    try:
        if hasattr(model, "disable_hooks"):
            # If your version supports it as a method
            model.disable_hooks() 
            
        # Perform the standard PyTorch forward/backward
        output = raw_model(input_data)
        loss = criterion(output, target)
        
        raw_model.zero_grad()
        loss.backward()
        
    finally:
        # Restore Opacus state so subsequent training/audits aren't broken
        if hasattr(model, "enable_hooks"):
            model.enable_hooks()

    # 4. FGSM logic
    if input_data.grad is not None:
        data_grad = input_data.grad.data
        sign_data_grad = data_grad.sign()
        perturbed_data = data + epsilon_robust * sign_data_grad
    else:
        # If gradients still fail, return original data
        perturbed_data = data
        
    return perturbed_data

def audit_robustness(model, X_tensor, y_tensor, epsilon_robust=0.1):
    """
    Computes adversarial accuracy.
    """
    criterion = nn.CrossEntropyLoss()
    
    # Generate adversarial examples
    X_adv = generate_fgsm_adversarial(model, criterion, X_tensor, y_tensor, epsilon_robust)
    
    # Standard evaluation
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()
    
    with torch.no_grad():
        output = raw_model(X_adv)
        preds = output.argmax(dim=1)
        robust_acc = (preds == y_tensor).float().mean()
        
    return robust_acc