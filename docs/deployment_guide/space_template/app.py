import gradio as gr
import numpy as np
import torch
from PIL import Image

from model import build_model          # your model class lives in model.py

THRESHOLD = 0.5                        # use the value you tuned on validation
NAMES = ["defect_a", "defect_b", "defect_c", "defect_d"]
COLORS = np.array([[230, 57, 70], [42, 157, 143], [69, 123, 255], [244, 162, 97]], dtype=np.float32)

model = build_model()
model.load_state_dict(torch.load("weights.pt", map_location="cpu"))
model.eval()


def preprocess(gray):
    """MUST match training. Edit this to your own preprocessing."""
    x = np.asarray(gray, dtype=np.float32) / 255.0
    x = (x - x.mean()) / max(float(x.std()), 0.05)
    return torch.from_numpy(x)[None, None]            # (1, 1, 128, 800)


def predict(image):
    gray = image.convert("L").resize((800, 128))       # width 800, height 128
    with torch.no_grad():
        probs = torch.sigmoid(model(preprocess(gray)))[0].numpy()   # (4, 128, 800)

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
