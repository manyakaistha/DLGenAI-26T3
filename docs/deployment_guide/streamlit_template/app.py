from pathlib import Path
import streamlit as st
import numpy as np
import torch
from PIL import Image

from model import build_model          # your model class lives in model.py

THRESHOLD = 0.5                        # use the value you tuned on validation
NAMES = ["defect_a", "defect_b", "defect_c", "defect_d"]
COLORS = np.array([[230, 57, 70], [42, 157, 143], [69, 123, 255], [244, 162, 97]], dtype=np.float32)

@st.cache_resource
def load_model():
    model = build_model()
    model.load_state_dict(torch.load(Path(__file__).with_name("weights.pt"), map_location="cpu", weights_only=True))
    return model.cpu().eval()


def preprocess(gray):
    """MUST match training. Edit this to your own preprocessing."""
    x = np.asarray(gray, dtype=np.float32) / 255.0
    x = (x - x.mean()) / max(float(x.std()), 0.05)
    return torch.from_numpy(x)[None, None]            # (1, 1, 128, 800)


def predict(image):
    model = load_model()
    gray = image.convert("L").resize((800, 128))       # width 800, height 128
    with torch.no_grad():
        probs = torch.sigmoid(model(preprocess(gray)))[0].cpu().numpy()   # (4, 128, 800)

    canvas = np.stack([np.asarray(gray, dtype=np.float32)] * 3, axis=-1)
    for c in range(4):
        hit = probs[c] > THRESHOLD
        canvas[hit] = 0.5 * canvas[hit] + 0.5 * COLORS[c]
    return Image.fromarray(canvas.astype(np.uint8))


st.title("Surface defect segmentation")
st.caption("Overlay: red = defect_a · green = defect_b · blue = defect_c · orange = defect_d")
upload = st.file_uploader("Upload a surface strip (800 × 128)", type=["png", "jpg", "jpeg"])
if upload is not None:
    image = Image.open(upload)
    if image.width * 128 != image.height * 800:
        st.error("Use a surface strip with the 800:128 aspect ratio.")
    elif st.button("Predict"):
        with st.spinner("Predicting on CPU…"):
            st.image(predict(image), caption="Predicted masks", width="stretch")
