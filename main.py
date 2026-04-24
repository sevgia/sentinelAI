import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import numpy as np
import warnings

# local project imports
from sentinel.data_loader import load_adult_data
from sentinel.models import AdultIncomeMLP
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

def execute_trust_audit():
    setup_mlflow("Sentinel_Full_Trust_Audit")

    # 1. Load Real Data
    X_train, X_test, y_train, y_test, raw_gender, raw_race = load_adult_data()

    # Convert training data to Tensors
    train_ds = TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train.values))
    train_loader = DataLoader(train_ds, batch_size=256, drop_last=True)
    
    # Static test tensors for accuracy
    X_test_tensor = torch.Tensor(X_test)
    y_test_tensor = torch.LongTensor(y_test.values)

    # 2. Experiment Settings
    epsilons = [1.0, 10.0, 100.0] 
    epochs = 15
    lr = 0.05

    for eps in epsilons:
        with mlflow.start_run(run_name=f"Epsilon_{eps}"):
            print(f"\n>>> Starting Audit for Epsilon: {eps}")
            
            # Initialize Neural Network
            model = AdultIncomeMLP(input_dim=X_train.shape[1])
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            # 3. Apply Differential Privacy (Opacus)
            model, optimizer, train_loader, _ = train_private_model(
                model, train_loader, optimizer, target_epsilon=eps, epochs=epochs
            )

            # 4. Training Loop
            model.train()
            for epoch in range(epochs):
                for data, labels in train_loader:
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, labels)
                    loss.backward()
                    optimizer.step()

            # 5. Post-Training Audits
            model.eval()
            
            # --- PHASE A: Accuracy & Fairness (No Gradients) ---
            with torch.no_grad():
                preds = model(X_test_tensor).argmax(dim=1)
                acc = (preds == y_test_tensor).float().mean().item()
                
                leak_score = calculate_mia_score(model, torch.Tensor(X_train[:500]), X_test_tensor[:500])
                
                di_gender = calculate_disparate_impact(model, X_test, raw_gender, unprivileged_value=0)
                di_race = calculate_disparate_impact(model, X_test, raw_race, unprivileged_value=0)

            # --- PHASE B: Robustness Audit (Requires Gradients) ---
            # We explicitly recreate tensors to ensure clean gradient tracking
            X_test_tensor_audit = torch.Tensor(X_test)
            y_test_tensor_audit = torch.from_numpy(y_test.to_numpy()).long()

            robust_score = audit_robustness(
                model, 
                X_test_tensor_audit, 
                y_test_tensor_audit, 
                epsilon_robust=0.1
            )

            # 6. Log Results
            mlflow.log_params({"epsilon": eps, "epochs": epochs, "lr": lr})
            mlflow.log_metrics({
                "accuracy": acc,
                "privacy_leakage": leak_score.item(),
                "disparate_impact_gender": di_gender.item(),
                "disparate_impact_race": di_race.item(),
                "adversarial_robustness": robust_score.item()
            })

            print(f"DONE | Acc: {acc:.3f} | Leak: {leak_score.item():.3f} | DI_gender: {di_gender.item():.3f} | DI_race: {di_race.item():.3f} | Robustness: {robust_score.item():.3f}")

if __name__ == "__main__":
    execute_trust_audit()