# 3 · I already have a `.pth` file

Maybe you saved your model as `model.pth` instead of `weights.pt`. Good news: this takes about two minutes to sort out.

## First, the name does not matter

`.pth` and `.pt` are **the same thing**. The letters at the end are just a label. What matters is **what is inside the file**, and that depends on how you saved it.

Think of IKEA furniture. `weights.pt` is a bag of parts and screws. Some `.pth` files are the same bag of parts. Others are the parts **plus a note saying "built by the class called `Net`"**. Python then needs that exact class before it can open the box.

## Which kind do you have?

Find the line you used when you saved:

| How you saved it | What is inside | What to do |
| --- | --- | --- |
| `torch.save(model.state_dict(), "model.pth")` | Only the weights | **Nothing.** Rename it to `weights.pt`. You are done. |
| `torch.save({"model_state_dict": ..., "optimizer": ...}, "model.pth")` | Weights **plus** training extras | [Convert it](#convert-it) |
| Trained with `DataParallel` or multiple GPUs | Weights whose names start with `module.` | [Convert it](#convert-it) |
| `torch.save(model, "model.pth")` | The **whole model object** | [Convert it](#convert-it) |
| `torch.jit.script(model).save(...)` (TorchScript) | A frozen copy of the model | [Convert it](#convert-it) |

Not sure? Run the converter below. It looks inside and tells you.

## Easiest fix: two lines in your notebook

If the notebook or script that trained the model is still open, where `model` exists, run this:

```python
import torch
torch.save(model.state_dict(), "weights.pt")
```

That always works, whatever you did before, and you get a clean `weights.pt`. Go back to [step 2 of Prepare the files](01_prepare.html#step-2-modelpy-your-architecture).

## Convert it

If you only have the `.pth` file, use the converter.

1. [Download `convert_to_weights.py`](convert_to_weights.py).
2. Put it in the same folder as your `.pth` file. If you saved the **whole model**, also put the `.py` file that defines your model class in that folder.
3. Run it:

```bash
pip install torch
python convert_to_weights.py model.pth
```

It writes `weights.pt` and prints the first few layer names:

```text
Wrote weights.pt: 772 tensors. First names:
   encoder._conv_stem.weight (32, 1, 3, 3)
   ...
```

The script handles all five cases in the table: plain weights, training checkpoints, the `module.` prefix, TorchScript files and whole pickled models. It also moves GPU-saved tensors to the CPU. It never changes the numbers.

### Check that it worked

With your `model.py` next to `weights.pt`:

```python
import torch
from model import build_model

model = build_model()
model.load_state_dict(torch.load("weights.pt", map_location="cpu"))
print("loads correctly")
```

If you see `loads correctly`, continue with [step 3 of Prepare the files](01_prepare.html#step-3-apppy-the-whole-app). If you see an error, find it in the table below.

## What the errors mean

| Error | What happened | Fix |
| --- | --- | --- |
| `UnpicklingError ... weights_only` | The file is a whole model or has non-weight extras | Run the converter |
| `Can't get attribute 'Net' on <module '__main__'>` | The class lived in a notebook, so Python cannot find it in a file | Use the **two lines in your notebook**, above |
| `No module named '...'` when converting | Your class file is not in the same folder | Copy the `.py` file that defines the class next to the converter |
| `Unexpected key(s) ... "module."` | Saved from `DataParallel` | Run the converter. It removes the prefix |
| `Missing key(s)` or `size mismatch` | `model.py` is not the exact architecture you trained | Copy the class and its arguments from training |

## Why not upload the `.pth` straight to the Space?

You can, but only for a whole-model file, and it is fragile:

- Python must find a class with the **exact same name in an importable file**. Rename or move the class and the file breaks.
- Modern PyTorch refuses to open it unless you pass `weights_only=False`. That means *run whatever code the file contains*.
- A class defined in a notebook usually cannot be found on the server at all.

`weights.pt` has none of those problems: it is only numbers, and **you** write `model.py`, so the two always match. Convert once and move on.

> **Safety rule.** Never open a `.pth` file from someone you do not know with `weights_only=False`. It can run code on your computer. Your own files are fine.

## Check yourself

1. Your friend sends you `model.pth` and says "just load it". What do you ask them before you try?
2. You run the converter and it prints names that start with `encoder.` and `decoder.`. Where in your project should you find those same names?
3. Why is the two-line notebook fix more reliable than converting a whole-model `.pth` later?

[← Publish on Hugging Face](02_publish.html) · [← Back to the overview](index.html)
