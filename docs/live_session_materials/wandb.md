# W&B in 50 lines

A tiny built-in dataset supplies the training values. Focus on the W&B calls: what they record, where the result appears, and how to reuse it.

[Open in Colab](https://colab.research.google.com/github/manyakaistha/DLGenAI-26T3/blob/main/docs/live_session_materials/wandb_tutorial.ipynb) · [Download notebook](wandb_tutorial.ipynb) · [Download Python](wandb_tutorial.py)

The notebook has one setup cell and one **50-line demo**. Set `MODE = "online"` to see the dashboard and sign in with `wandb.login()`. The default offline mode saves logs locally. The Iris dataset is included with scikit-learn; no GPU or data download is required.

## The complete demo

```python
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
```

## Find each API pattern in the run

| API | What it records | Where to inspect it |
|---|---|---|
| `wandb.init(config=...)` | One execution and its settings | Run overview and config |
| `run.define_metric(...)` | Epoch as the metric axis | Chart x-axis |
| `run.log({...})` | Loss and accuracy over time | Metric charts |
| `run.summary[...]` | Final accuracy and elapsed training time | Run summary |
| `wandb.Histogram(...)` | Distribution of prediction confidence | Histogram panel |
| `wandb.Table(...)` | True class, prediction and confidence per example | Predictions table |
| `wandb.Image(fig)` | A Matplotlib plot | Plot panel |
| `wandb.Artifact(...)`, `add_file`, `log_artifact` | A versioned saved model and preprocessing | Artifacts |
| `run.finish()` | Completes queued logging and closes the run | Finished run status |

## Try one change

Change `lr`, rerun, and compare the two accuracy/loss curves. Keep the split and other settings fixed. The table helps you inspect individual errors; a single average does not show which examples were missed.

The artifact contains the model, fitted scaler and configuration. `latest` is a movable alias; use a fixed version such as `v0` when reproducing a result. To download your saved version later:

```python
saved = wandb.Api().artifact("YOUR_ENTITY/wandb-first-run/iris-model:v0")
folder = saved.download()
```

Offline runs must be uploaded with `wandb sync wandb/offline-run-*` before their dashboards or artifacts are available online.

Official references: [logging](https://docs.wandb.ai/guides/track/log/), [tables](https://docs.wandb.ai/guides/track/log/log-tables), [artifacts](https://docs.wandb.ai/models/artifacts/download-and-use-an-artifact).
