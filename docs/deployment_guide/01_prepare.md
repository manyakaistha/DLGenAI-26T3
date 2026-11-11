# 1 · Prepare the files

Your Space is a folder with five files.

![The five files in the Space folder](figures/space_files.png)

[Download the starter folder](surface_defect_space.zip) and edit it, or copy the code below. The download holds the four text files; add your own `weights.pt`. Only `model.py`, `weights.pt` and the marked lines of `app.py` are specific to you.

## Step 1 · Save the weights

At the end of training, in your notebook or script:

```python
torch.save(model.state_dict(), "weights.pt")
```

Save the `state_dict` only. Do not save the optimizer or the whole model object. The file is smaller and loads on any machine.

**Already have a `model.pth`?** The name does not matter, but what is inside does. See [page 3 · I already have a `.pth` file](03_have_a_pth.md) for a two-minute fix and a converter.

## Step 2 · `model.py`: your architecture

Copy your model class here and add a function that builds it **without downloading anything**:

```python
import torch.nn as nn

def build_model():
    # Replace with YOUR architecture. Output: (batch, 4, 128, 800) logits.
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
        nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
        nn.Conv2d(16, 4, 1),
    )
```

The starter body is a tiny placeholder so the folder runs out of the box. Replace it with the layers you trained.

If you used a pretrained encoder, build it with `encoder_weights=None`. The trained values already live in `weights.pt`, so the app must not download ImageNet weights again.

## Step 3 · `app.py`: the whole app

```python
import gradio as gr
import numpy as np
import torch
from PIL import Image

from model import build_model          # your model class lives in model.py

try:                                   # free accounts run on ZeroGPU, which provides `spaces`
    import spaces
    gpu = spaces.GPU(duration=20)
except ImportError:                    # on your own computer this does nothing
    def gpu(fn):
        return fn

THRESHOLD = 0.5                        # use the value you tuned on validation
NAMES = ["defect_a", "defect_b", "defect_c", "defect_d"]
COLORS = np.array([[230, 57, 70], [42, 157, 143], [69, 123, 255], [244, 162, 97]], dtype=np.float32)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_model()
model.load_state_dict(torch.load("weights.pt", map_location="cpu"))
model.to(device).eval()


def preprocess(gray):
    """MUST match training. Edit this to your own preprocessing."""
    x = np.asarray(gray, dtype=np.float32) / 255.0
    x = (x - x.mean()) / max(float(x.std()), 0.05)
    return torch.from_numpy(x)[None, None]            # (1, 1, 128, 800)


@gpu
def predict(image):
    gray = image.convert("L").resize((800, 128))       # width 800, height 128
    with torch.no_grad():
        probs = torch.sigmoid(model(preprocess(gray).to(device)))[0].cpu().numpy()   # (4, 128, 800)

    canvas = np.stack([np.asarray(gray, dtype=np.float32)] * 3, axis=-1)
    for c in range(4):
        hit = probs[c] > THRESHOLD
        canvas[hit] = 0.5 * canvas[hit] + 0.5 * COLORS[c]
    return Image.fromarray(canvas.astype(np.uint8))


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Surface image"),
    outputs=gr.Image(label="Predicted masks"),
    title="Industrial surface defect segmentation",
    description="Upload a surface image. Overlay colours: "
                "🔴 defect_a · 🟢 defect_b · 🔵 defect_c · 🟠 defect_d",
)

if __name__ == "__main__":
    demo.launch()
```

### What each section does

Think of `app.py` as a small restaurant. The **setup** (sections 1 to 4) happens once, when the Space starts: unpack the ingredients and heat the oven. The **cooking** (sections 5 and 6) happens every time someone uploads an image. Sections 7 and 8 build the dining room and open the doors.

#### 1 · Imports: get the tools

| Line | In plain words |
| --- | --- |
| `gradio` | Builds the web page for us: the upload box and the result box. |
| `numpy` | Handles pictures as grids of numbers. |
| `torch` | Runs the neural network. |
| `PIL.Image` | Opens, resizes and saves images. |
| `from model import build_model` | Brings in the recipe for the empty network, from your own `model.py`. |

#### 2 · The `@gpu` marker: for the free GPU

```python
try:
    import spaces
    gpu = spaces.GPU(duration=20)
except ImportError:
    def gpu(fn):
        return fn
```

Free Hugging Face accounts run Gradio apps on **ZeroGPU**: a shared graphics card that is lent to your function only while it runs. Hugging Face needs a label on the function that may use it. This block creates that label, called `gpu`. Without it, the Space stops with *"No @spaces.GPU function detected"*.

- **On Hugging Face:** the `spaces` library exists, so `gpu` becomes the real label.
- **On your own computer:** there is no `spaces` library, so Python takes the `except` branch and `gpu` becomes a label that does nothing.

The same file therefore works in both places. You never need to change this section.

#### 3 · Settings: the things you may want to change

- **`THRESHOLD`:** the network gives every pixel a probability from 0 to 1. A pixel counts as a defect if its probability is **above this number**. Use the value you chose from validation.
- **`NAMES`:** the four defect labels, kept for reference. The code does not use it. The legend in the page description is typed by hand.
- **`COLORS`:** one colour per defect, written as red, green, blue values from 0 to 255. `[230, 57, 70]` is red. Defect `a` gets the first colour, `b` the second, and so on.

#### 4 · Load the model once

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_model()
model.load_state_dict(torch.load("weights.pt", map_location="cpu"))
model.to(device).eval()
```

This runs **once**, when the Space starts, not on every upload.

1. **`device`:** use the graphics card if there is one, otherwise the normal processor.
2. **`build_model()`:** make the empty network, like furniture with no screws yet.
3. **`torch.load("weights.pt", ...)`:** read your trained numbers from the file.
4. **`load_state_dict(...)`:** pour those numbers into the empty network. This only works if `model.py` has exactly the layers you trained.
5. **`.to(device)`:** move the model to the GPU or CPU chosen in step 1.
6. **`.eval()`:** switch to test mode, so layers behave the way they did during validation.

#### 5 · `preprocess`: shape the image the way the model expects

```python
x = np.asarray(gray, dtype=np.float32) / 255.0
x = (x - x.mean()) / max(float(x.std()), 0.05)
return torch.from_numpy(x)[None, None]
```

The model was trained on images prepared in one particular way, so at prediction time you **must** prepare them the same way. This is the most common place for hidden bugs.

- **Line 1:** turn the image into numbers and scale them from 0–255 down to 0–1.
- **Line 2:** subtract the average and divide by the spread, so every image has average 0 and spread 1. The `max(..., 0.05)` stops a division by zero on a flat grey image.
- **Line 3:** turn it into a PyTorch tensor. `[None, None]` adds two extra dimensions: **batch** (how many images) and **channel** (grayscale means 1). The shape becomes `(1, 1, 128, 800)`.

If your training did something different, edit this function to match.

#### 6 · `predict`: the part that runs for every upload

It has three stages.

**6a · Prepare**

```python
@gpu
def predict(image):
    gray = image.convert("L").resize((800, 128))
```

`@gpu` is the label from section 2. `convert("L")` makes the picture grayscale. `resize((800, 128))` forces width 800 and height 128, the size the model was trained on.

**6b · Ask the model**

```python
with torch.no_grad():
    probs = torch.sigmoid(model(preprocess(gray).to(device)))[0].cpu().numpy()
```

Read it from the inside out:

1. `preprocess(gray)`: prepare the image (section 5).
2. `.to(device)`: put it on the same device as the model.
3. `model(...)`: run the network. The output is 4 grids of raw scores, one per defect.
4. `torch.sigmoid(...)`: turn raw scores into probabilities between 0 and 1.
5. `[0]`: remove the batch dimension, since we sent one image.
6. `.cpu().numpy()`: bring the result back to ordinary numbers.

`probs` has shape `(4, 128, 800)`: for each of the 4 defects, a probability at every one of the 128 × 800 pixels. `torch.no_grad()` tells PyTorch we are only predicting, not training, which saves memory.

**6c · Paint the masks**

```python
canvas = np.stack([np.asarray(gray, dtype=np.float32)] * 3, axis=-1)
for c in range(4):
    hit = probs[c] > THRESHOLD
    canvas[hit] = 0.5 * canvas[hit] + 0.5 * COLORS[c]
return Image.fromarray(canvas.astype(np.uint8))
```

- **`canvas`:** the grayscale picture copied three times (red, green, blue) so we can draw in colour.
- **`for c in range(4)`:** go through the four defects one at a time.
- **`hit = probs[c] > THRESHOLD`:** a grid of True and False. True means "the model thinks this pixel has defect `c`".
- **`canvas[hit] = 0.5 * canvas[hit] + 0.5 * COLORS[c]`:** at the True pixels, mix **half the original picture with half the defect's colour**. That is why you can still see the surface through the colour. The False pixels are untouched.
- **`Image.fromarray(...)`:** turn the numbers back into a picture to show on the page.

#### 7 · `gr.Interface`: the whole web page in one call

Gradio reads this as: *"Show an upload box. When someone uploads, call `predict` with that image. Show what `predict` returns in an output box."*

- `fn=predict`: which function to run.
- `inputs=gr.Image(type="pil")`: an upload box that hands your function a PIL image.
- `outputs=gr.Image(...)`: an image box for the result.
- `title` and `description`: the text shown on the page, including the colour legend.

#### 8 · Start the app

`if __name__ == "__main__": demo.launch()` means "if this file is run directly, start the web server". On your own computer you run `python app.py` and open the link it prints. On Hugging Face the platform runs the file for you.

**The whole file in one line:** load the model once, then for each upload resize, predict, threshold, colour, and show the result.

### Three things to check in your own copy

1. **Preprocessing** is identical to training: same scaling, same normalisation, same size.
2. **Output meaning** is four independent channels with `sigmoid`, in the order a, b, c, d.
3. **`THRESHOLD`** is the value you chose from validation, not a default you never examined.

### If your model is different

| Your model | Change |
| --- | --- |
| Five outputs, background first, trained with softmax | `probs = torch.softmax(model(x), 1)[0, 1:].numpy()` |
| Expects 3 input channels | `x = preprocess(gray).repeat(1, 3, 1, 1)` before calling the model |
| A different threshold for each class | `THRESHOLD = [0.4, 0.5, 0.6, 0.5]` and use `probs[c] > THRESHOLD[c]` |
| Uses `segmentation_models_pytorch` | Add `segmentation-models-pytorch` to `requirements.txt` and build with `encoder_weights=None` |

## Step 4 · `requirements.txt`

```text
torch==2.8.0
numpy
pillow
```

ZeroGPU supports PyTorch **2.8.0 or newer**, so pin 2.8.0 or higher. Do **not** add a CPU-only PyTorch line: ZeroGPU needs the normal build with GPU support. Gradio is installed automatically from the version named in `README.md`; do not list it here. The `spaces` library is already installed on Hugging Face.

## Step 5 · `README.md`: the Space settings

Hugging Face reads this short header to know how to run your app. ZeroGPU only supports Python `3.12.12` and `3.10.13`, so keep the `python_version` line:

```yaml
---
title: Surface Defect Segmentation
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12.12"
app_file: app.py
pinned: false
---
```

## Step 6 · Test on your own computer first

```bash
pip install gradio==6.31.0 torch numpy pillow
python app.py
```

Open the address printed in the terminal (usually `http://127.0.0.1:7860`), upload an image and look at the result.

- The overlay lands on the defects you expect.
- The output has the same width and height as the model input (800 × 128).
- A clean image gives little or no colour.

If it works here, it will work on Hugging Face. Debugging on your laptop takes seconds; debugging a remote build takes minutes.

**Check yourself:** your model was trained on images divided by 255 only, but the starter `preprocess` also subtracts the mean and divides by the standard deviation. The app runs without errors. What do you expect to see, and why will no error message warn you?

[Next: publish on Hugging Face →](02_publish.md)
