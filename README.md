# Sentinel-AI: A Multi-Dimensional Trust Audit Framework

**Sentinel-AI** is a research-oriented framework designed to audit and visualize the Trio of Trust in machine learning: **Privacy, Fairness, and Robustness.** This project explores the complex trade-offs between utility (accuracy) and ethical constraints. Specifically, it focuses on how **Differential Privacy (DP)** budgets ($\epsilon$) influence a model's susceptibility to adversarial attacks and its propensity for algorithmic bias in structured datasets.

---

## Core Audit Pillars

Sentinel-AI evaluates models across three critical dimensions:

* **Privacy:** Evaluates Membership Inference Attack (MIA) vulnerability using the confidence-gap method to measure how much training data "leaks" through the model's predictions.
* **Fairness:** Quantifies **Disparate Impact (DI)** across race and gender protected attributes to detect algorithmic discrimination.
* **Robustness:** Measures **Adversarial Accuracy** using the Fast Gradient Sign Method (FGSM) to test model resilience against targeted feature perturbations.

---

## Key Findings: The "Fairness Cliff"

Our audit of the UCI Adult Income dataset revealed a critical phenomenon termed the **"Fairness Cliff."** As the privacy budget ($\epsilon$) increases to maximize utility, the model's ethical alignment collapses.

### Audit Results Summary

| Metric | $\epsilon=1.0$ (Private) | $\epsilon=10.0$ (Optimal) | $\epsilon=100.0$ (Non-Private) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 75.0% | **84.2%** | 84.7% |
| **Robustness (Adv. Acc)** | 0.636 | **0.758** | 0.774 |
| **Race Fairness (DI)** | 2.961 | **0.214** | 0.100 |
| **Privacy Leakage** | 0.500 | 0.624 | 0.591 |

### Critical Analysis

1.  **The Fairness Cliff:** At $\epsilon=100$, the model achieves peak accuracy (84.7%) but at a devastating cost to fairness. The Race Disparate Impact falls to **0.100**, indicating the model is 10x less likely to predict high income for the unprivileged group compared to the privileged group.
2.  **The Robustness Paradox:** Contrary to the theory that noise always improves robustness, at $\epsilon=1.0$ the model underfits the baseline (~75% majority class), resulting in unstable decision boundaries that are easily manipulated by FGSM attacks.
3.  **The Sentinel "Sweet Spot":** The audit identifies **$\epsilon=10.0$** as the "Goldilocks" configuration, offering near-peak accuracy while maintaining significantly higher fairness and robustness compared to the unconstrained baseline.

Note: High-epsilon stress tests ($\epsilon=100$) were conducted to establish a baseline for 'natural bias' in the UCI Adult dataset, acknowledging the vacuous nature of privacy bounds at this scale.

---

##  Technical Stack

* **Core Logic:** PyTorch
* **Privacy Engine:** Opacus (DP-SGD)
* **Experiment Tracking:** MLflow
* **Data Processing:** Pandas, Scikit-learn
* **Adversarial Logic:** FGSM (Fast Gradient Sign Method)

---

## Future Work: Adversarial Debiasing

The next phase of Sentinel-AI involves implementing **Adversarial Debiasing** via min-max optimization. By introducing an adversary network during training, we aim to "bridge" the Fairness Cliff—maintaining high utility while restoring Disparate Impact scores to acceptable regulatory levels (>0.8).

---

## Academic Context

This project serves as a framework for quantifying the hidden costs of privacy-preserving machine learning in socio-technical systems.

---

## How to Run the Audit

1.  **Initialize MLflow Tracking Server:**
    ```bash
    mlflow server --host 127.0.0.1 --port 5000
    ```
2.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Execute the Pipeline:**
    ```bash
    python main.py
    ```
4.  **Analyze Results:**
    Open `http://127.0.0.1:5000` to view the multi-run comparison dashboard.

    ---

  