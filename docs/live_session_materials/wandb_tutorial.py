# %% [markdown]
# # W&B: log a machine-learning experiment
# Iris has 150 built-in examples. The model is just a source of values to log.
# Run setup, then the demo. Use MODE="online" and sign in for a live dashboard;
# offline mode records locally. No GPU or dataset download is needed.

# %%
# W&B demo: metrics, media, tables and a versioned model.
import time
import joblib
import matplotlib.pyplot as plt
import wandb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss

MODE = "offline"  # Use "online" to see the live W&B dashboard.
if MODE == "online":
    wandb.login()  # Sign in securely; never paste a key into this file.
data = load_iris()  # 150 built-in examples; no dataset download.
X_train, X_val, y_train, y_val = train_test_split(
    data.data, data.target, test_size=0.2, stratify=data.target, random_state=42)
scaler = StandardScaler().fit(X_train)
x_train, x_val = scaler.transform(X_train), scaler.transform(X_val)
config = {"lr": 0.03, "epochs": 15, "seed": 42, "dataset": "iris"}
run = wandb.init(project="wandb-first-run", config=config, mode=MODE,
                 settings=wandb.Settings(x_disable_stats=True))
run.define_metric("epoch")
run.define_metric("val/*", step_metric="epoch")
model = SGDClassifier(loss="log_loss", learning_rate="constant",
                      eta0=run.config.lr, random_state=run.config.seed)
started = time.perf_counter()
for epoch in range(1, run.config.epochs + 1):
    model.partial_fit(x_train, y_train, classes=[0, 1, 2])
    probs = model.predict_proba(x_val)
    preds = probs.argmax(axis=1)
    run.log({"epoch": epoch, "val/loss": log_loss(y_val, probs),
             "val/accuracy": accuracy_score(y_val, preds)})
run.summary["training_seconds"] = time.perf_counter() - started
run.summary["final_accuracy"] = accuracy_score(y_val, preds)
run.log({"confidence": wandb.Histogram(probs.max(axis=1))})
table = wandb.Table(columns=["true", "predicted", "confidence"])
for truth, pred, confidence in zip(y_val, preds, probs.max(axis=1)):
    table.add_data(data.target_names[truth], data.target_names[pred], float(confidence))
run.log({"predictions": table})
fig, ax = plt.subplots(figsize=(5, 3))
ax.scatter(X_val[:, 0], X_val[:, 1], c=preds, vmin=0, vmax=2)
ax.set(xlabel=data.feature_names[0], ylabel=data.feature_names[1], title="Predicted class")
run.log({"plot": wandb.Image(fig)}); plt.close(fig)
joblib.dump({"model": model, "scaler": scaler, "config": config}, "model.joblib")
artifact = wandb.Artifact("iris-model", type="model", metadata={"accuracy": float(run.summary["final_accuracy"])})
artifact.add_file("model.joblib")
run.log_artifact(artifact, aliases=["latest"])
print("Run:", run.url if MODE == "online" else "Offline logs:", run.dir)
run.finish()  # Close the run and complete queued logging.

# %% [markdown]
# Open the run: inspect loss/accuracy, predictions, confidence, plot and Artifacts.
# Change only lr and rerun to compare experiments. The same split is used each time.
# An artifact alias such as latest can move; v0/v1 identifies a fixed version.
# Offline upload after completion: `wandb sync wandb/offline-run-*`.
