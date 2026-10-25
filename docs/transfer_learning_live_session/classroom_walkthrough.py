# /// script
# requires-python = ">=3.10"
# dependencies = ["torch==2.8.0", "torchvision==0.23.0"]
# ///
# %% [markdown]
# # A small upgrade to the U-Net you already know
# 26 October 2026. Optional classroom demo: no MRI download, GPU or training.
# We inspect an encoder and its trainable parameters. This is not a new project.
# The dummy input below is only for checking tensor shapes, not a real MRI scan.

# %%
import os
from collections import OrderedDict
import torch
from torchvision.models import resnet18, ResNet18_Weights

torch.set_num_threads(min(4, os.cpu_count() or 1))
torch.manual_seed(42)

# %% [markdown]
# ## 1. Find the encoder inside a familiar segmentation model
# A U-Net encoder supplies features to a decoder. Here we inspect ResNet18's
# feature layers. We do not build a decoder or predict masks in this small demo.
# The complete MRI training notebook is a separate, optional exercise.

# %%
USE_PRETRAINED = os.environ.get("USE_PRETRAINED", "0") == "1"
# In a notebook you can instead set USE_PRETRAINED = True here.
# True loads actual ImageNet weights: internet or a prepared cache is required.
weights = ResNet18_Weights.IMAGENET1K_V1 if USE_PRETRAINED else None
backbone = resnet18(weights=weights)
encoder = torch.nn.Sequential(OrderedDict(list(backbone.named_children())[:-2]))
encoder.eval()
print("ImageNet encoder weights loaded" if USE_PRETRAINED else
      "Random-weight shape demo; no pretrained weights loaded")

# %% [markdown]
# ## 2. Follow a dummy image through the feature layers
# The three input channels could hold three MRI sequences in the real example.
# These zeros contain no patient information. The decreasing spatial sizes show
# the encoder's structure; they tell us nothing about prediction quality.

# %%
x = torch.zeros(1, 3, 128, 128)
print("Input:", tuple(x.shape))
with torch.no_grad():
    features = x
    for name, layer in encoder.named_children():
        features = layer(features)
        print(f"{name}:", tuple(features.shape))
assert features.shape == (1, 512, 4, 4)
print("A segmentation decoder would use spatial features to learn a mask.")

# %% [markdown]
# ## 3. Choose which parameters can learn
# Freezing stops gradients for the encoder parameters. A segmentation model's
# new decoder can still learn. BatchNorm mode and optimizer membership also
# matter in a real training loop; those details are in the optional website notes.

# %%
for parameter in encoder.parameters():
    parameter.requires_grad_(False)
encoder.eval()  # keep its BatchNorm running statistics fixed too
frozen_count = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
print("Trainable encoder parameters when frozen:", frozen_count)
assert frozen_count == 0

for parameter in encoder.parameters():
    parameter.requires_grad_(True)
trainable_count = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
print("Encoder parameters available for fine-tuning:", trainable_count)
assert trainable_count > 0
print("No optimizer step was run; enabling gradients alone does not train a model.")

# %% [markdown]
# ## 4. Describe a fair comparison
# Imagine two copies of the same ResNet-based U-Net. Use the same split, input
# preparation, decoder starting weights, loss and training settings. Change only
# the encoder's starting weights: random in A, ImageNet in B. Then compare real
# validation results. This demo does not train either arm or invent a Dice score.
#
# Quick check: if B also gets more epochs, what else could explain an improvement?
# Return to the notes: https://manyakaistha.github.io/DLGenAI-26T3/session-3/
