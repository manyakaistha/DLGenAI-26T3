# Baseline Model Training Dynamics & Convergence Diagnostics
*(Milestone 2 · Question 8 Study Guide)*

---

## 1. The 5-Epoch Baseline Training Lifecycle

With your PyTorch `SurfaceAnomalyDataset` and DataLoader implemented under **Paradigm 1 (Multi-Channel Joint Prediction)**, you are ready to train and evaluate your first baseline segmentation model (such as a 3-stage or 4-stage U-Net).

### Optimization & Evaluation Protocol
- **Optimizer:** Adam or AdamW with an initial learning rate $\eta \approx 3 \times 10^{-4}$.
- **Loss Function:** `nn.BCEWithLogitsLoss()`.
- **Evaluation Schedule:** At the end of each epoch, evaluate the loss across your validation holdout with `model.eval()` and `torch.no_grad()`.
- **Logging:** Record the average training loss and average validation loss for each of the 5 epochs.

---

## 2. Diagnosing Convergence & Generalization Regimes

Plotting and comparing the trajectories of training loss versus validation loss across epochs reveals how your neural network is learning:

### Regime 1: Underfitting (High Bias)
- **Visual Trajectory:** Both training loss and validation loss remain high and decrease very slowly, or plateau almost immediately.
- **Diagnosis:** The network has failed to extract meaningful features from the training strips.
- **Common Causes:** Learning rate is too small, model capacity is too limited, or input images were not properly normalized to $[0.0, 1.0]$.

### Regime 2: Overfitting (High Variance)
- **Visual Trajectory:** Training loss decreases steadily to low values, but validation loss bottoms out early (e.g., at Epoch 2 or 3) and then begins climbing upward.
- **Diagnosis:** The model is memorizing training noise and surface artifacts rather than learning generalizable defect signatures.
- **Common Causes:** Training for too many epochs without regularization (dropout, weight decay) or lack of data augmentation.

### Regime 3: Healthy Generalization
- **Visual Trajectory:** Training loss decreases steadily across all epochs. Validation loss decreases in parallel, tracking slightly above training loss and leveling out cleanly without diverging.
- **Diagnosis:** The network is learning robust visual representations that transfer effectively to unseen industrial strips.

---

## 3. Checkpoint Selection for Inference

In machine learning competitions and production deployments, **never blindly select the model weights from the final epoch**.

If a model begins overfitting after a few epochs, the final epoch's weights will perform worse on test data than earlier weights. You should always save and select the model checkpoint from the epoch that achieved the **lowest validation loss**.

---

## 4. Guiding Hints for Question 8

To complete Question 8:

- **Hint 1 (Record Epoch Metrics):** Tabulate your average training loss and average validation loss at the end of each epoch (Epochs 1 through 5).
- **Hint 2 (Identify Optimal Checkpoint):** Compare the validation losses across all 5 epochs to find the lowest value. That epoch is your optimal checkpoint.
- **Hint 3 (Characterize Generalization Dynamics):** Examine whether your validation loss tracks training loss downwards (healthy generalization), diverges upward (overfitting), or remains high and flat (underfitting). Justify your diagnosis in 1–2 sentences.

---

## 5. Your Assignment Task

1. Open [Milestone 2 Assessment: Question 8](milestone-2.html#question-8-baseline-model-training-dynamics-and-convergence).
2. Train your baseline U-Net model for 5 epochs, logging train and validation loss for each epoch.
3. Report your losses at Epoch 1 and Epoch 5, identify the best epoch, and provide your 1–2 sentence generalization assessment: `epoch1_(train, val), epoch5_(train, val), best_epoch, generalization_assessment`.
