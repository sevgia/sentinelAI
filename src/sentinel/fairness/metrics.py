import numpy as np
import torch

def calculate_disparate_impact(model, X_test, sensitive_features, unprivileged_value=0, privileged_value=1):
    model.eval()
    with torch.no_grad():
        X_tensor = X_test if isinstance(X_test, torch.Tensor) else torch.Tensor(X_test)
        logits = model(X_tensor)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    
    if hasattr(sensitive_features, 'values'):
        sf_arr = sensitive_features.values.flatten()
    elif isinstance(sensitive_features, torch.Tensor):
        sf_arr = sensitive_features.cpu().numpy().flatten()
    else:
        sf_arr = np.array(sensitive_features).flatten()

    # --- DIAGNOSTIC TELEMETRY ---
    unique_vals = np.unique(sf_arr)
    print(f"\n--- DIAGNOSTIC AUDIT ---")
    print(f"Unique values found in sensitive features: {unique_vals}")
    print(f"Looking for Unprivileged: {unprivileged_value} | Privileged: {privileged_value}")

    unprivileged_mask = (sf_arr == unprivileged_value)
    privileged_mask = (sf_arr == privileged_value)
    
    print(f"Unprivileged Mask Match Count: {unprivileged_mask.sum()}")
    print(f"Privileged Mask Match Count: {privileged_mask.sum()}")

    if unprivileged_mask.sum() == 0:
        print(" -> Short-circuited: Unprivileged group has 0 matches. Returning 1.0")
        return torch.tensor([1.0])
    
    if privileged_mask.sum() == 0:
        print(" -> Short-circuited: Privileged group has 0 matches. Returning 1.0")
        return torch.tensor([1.0])
        
    unpriv_selection_rate = preds[unprivileged_mask].astype(float).mean()
    priv_selection_rate = preds[privileged_mask].astype(float).mean()
    
    print(f"Unprivileged Selection Rate (Favorable Outcome %): {unpriv_selection_rate:.4f}")
    print(f"Privileged Selection Rate (Favorable Outcome %): {priv_selection_rate:.4f}")

    if priv_selection_rate == 0: 
        print(" -> Short-circuited: Privileged selection rate is 0. Returning 1.0")
        return torch.tensor([1.0])
        
    di = unpriv_selection_rate / priv_selection_rate
    print(f"Calculated Disparate Impact: {di:.4f}")
    print("-------------------------\n")
    
    return torch.tensor([di])