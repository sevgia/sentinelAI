import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import numpy as np
import warnings

# Local project imports
from sentinel.data_loader import load_adult_data
from sentinel.models import AdultIncomeMLP, FairnessAdversary
from sentinel.privacy.attacks import train_private_model, calculate_mia_score
from sentinel.fairness.metrics import calculate_disparate_impact
from sentinel.robustness.attacks import audit_robustness

# Suppress RDP overflows for high-epsilon runs
warnings.filterwarnings("ignore", category=RuntimeWarning, module="opacus")
warnings.filterwarnings("ignore", category=UserWarning, module="opacus")

def setup_mlflow(experiment_name):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

def run_audit_experiment(mitigated=False):
    """
    Unified execution pipeline. Toggles adversarial debiasing based on the `mitigated` flag
    to ensure DRY (Don't Repeat Yourself) compliance while preserving control baselines.
    """
    setup_mlflow("Sentinel_Full_Trust_Audit")

    # 1. Load Real Data and Run Integrity Verification
    (X_train, X_test, y_train, y_test, 
     gender_train, gender_test, 
     race_train, race_test) = load_adult_data()
    
    if not mitigated:
        print("\n--- DATA LOADER INTEGRITY CHECK ---")
        print(f"Train Race Unique Values: {np.unique(race_train, return_counts=True)}")
        print(f"Test Race Unique Values: {np.unique(race_test, return_counts=True)}")
        print(f"Train Gender Unique Values: {np.unique(gender_train, return_counts=True)}")
        print(f"Test Gender Unique Values: {np.unique(gender_test, return_counts=True)}")
        print("-----------------------------------\n")

    # Convert data structures to Core PyTorch Tensors
    train_ds = TensorDataset(
        torch.Tensor(X_train), 
        torch.LongTensor(y_train.values),
        torch.Tensor(gender_train),
        torch.Tensor(race_train)
    )
    
    X_test_tensor = torch.Tensor(X_test)
    y_test_tensor = torch.LongTensor(y_test.values)

    # 2. Experiment Settings
    epsilons = [1.0, 10.0, 100.0] 
    epochs = 15
    lr = 0.05
    
    # Asymmetric Bias Multipliers
    alpha_gender = 4.5  
    alpha_race = 1.5    

    for eps in epsilons:
        # Separate run naming schemas to organize MLflow outputs clearly
        run_name = f"Mitigated_Epsilon_{eps}" if mitigated else f"Epsilon_{eps}"
        
        with mlflow.start_run(run_name=run_name):
            print(f"\n>>> Starting {'Mitigated ' if mitigated else ''}Audit for Epsilon: {eps}")
            
            # Re-instantiate data loader to refresh internal dataset tracking state
            train_loader = DataLoader(train_ds, batch_size=256, drop_last=True)
            
            # Initialize Backbone Classifier
            model = AdultIncomeMLP(input_dim=X_train.shape[1])
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            # Conditional Setup for Adversarial Tracking Framework
            if mitigated:
                adv_gender = FairnessAdversary(input_dim=2).train()
                adv_race = FairnessAdversary(input_dim=2).train()
                opt_adv_g = optim.Adam(adv_gender.parameters(), lr=0.05)
                opt_adv_r = optim.Adam(adv_race.parameters(), lr=0.05)
                adv_criterion = nn.BCELoss()

            # 3. Apply Differential Privacy Layer (Opacus Wrapper)
            model, optimizer, train_loader, _ = train_private_model(
                model, train_loader, optimizer, target_epsilon=eps, epochs=epochs
            )

            # 4. Processing Execution Loop
            model.train()
            for epoch in range(epochs):
                for data, labels, g_attr, r_attr in train_loader:
                    
                    # --- PATHWAY A: ADVERSARIAL MULTI-TASK STEP (Only if mitigated) ---
                    if mitigated:
                        opt_adv_g.zero_grad()
                        opt_adv_r.zero_grad()
                        
                        with torch.no_grad():
                            logits = model(data)
                        
                        loss_adv_g = adv_criterion(adv_gender(logits), g_attr.unsqueeze(1))
                        loss_adv_r = adv_criterion(adv_race(logits), r_attr.unsqueeze(1))
                        
                        (loss_adv_g + loss_adv_r).backward()
                        opt_adv_g.step()
                        opt_adv_r.step()

                    # --- PATHWAY B: CLASSIFIER WEIGHT OPTIMIZATION STEP ---
                    optimizer.zero_grad()
                    logits = model(data)
                    clf_loss = criterion(logits, labels)
                    
                    if mitigated:
                        # Maximize adversary error using independent constraint alphas
                        f_loss_g = adv_criterion(adv_gender(logits), g_attr.unsqueeze(1))
                        f_loss_r = adv_criterion(adv_race(logits), r_attr.unsqueeze(1))
                        total_loss = clf_loss - (alpha_gender * f_loss_g) - (alpha_race * f_loss_r)
                    else:
                        total_loss = clf_loss
                        
                    total_loss.backward()
                    optimizer.step()

            # 5. Comprehensive Trust Metric Auditing
            model.eval()
            with torch.no_grad():
                preds = model(X_test_tensor).argmax(dim=1)
                acc = (preds == y_test_tensor).float().mean().item()
                
                # Sample slices to accelerate Membership Inference calculations
                leak_score = calculate_mia_score(model, torch.Tensor(X_train[:500]), X_test_tensor[:500])
                
                di_gender = calculate_disparate_impact(model, X_test, gender_test, unprivileged_value=0)
                di_race = calculate_disparate_impact(model, X_test, race_test, unprivileged_value=0)

            # Recreate clean tensor graphs specifically for verification under FGSM perturbations
            X_test_tensor_audit = torch.Tensor(X_test)
            y_test_tensor_audit = torch.from_numpy(y_test.to_numpy()).long()
            robust_score = audit_robustness(model, X_test_tensor_audit, y_test_tensor_audit, epsilon_robust=0.1)

            # 6. MLOps Parameter Recording and Logging
            mlflow.log_params({"epsilon": eps, "epochs": epochs, "lr": lr, "mitigated": mitigated})
            mlflow.log_metrics({
                "accuracy": acc,
                "privacy_leakage": leak_score.item(),
                "disparate_impact_gender": di_gender.item(),
                "disparate_impact_race": di_race.item(),
                "adversarial_robustness": robust_score.item()
            })

            with torch.no_grad():
                pos_rate = model(X_test_tensor).argmax(dim=1).float().mean().item()
                print(f"Total Positive Rate (Selection Rate): {pos_rate:.4f}")

            prefix = "MITIGATED" if mitigated else "DONE"
            print(f"{prefix} | Acc: {acc:.3f} | Leak: {leak_score.item():.3f} | DI_gender: {di_gender.item():.3f} | DI_race: {di_race.item():.3f} | Robustness: {robust_score.item():.3f}")

if __name__ == "__main__":
    # Execute structural control baseline tracking loop
    run_audit_experiment(mitigated=False)
    
    # Execute experimental multi-adversarial debiasing loop
    run_audit_experiment(mitigated=True)