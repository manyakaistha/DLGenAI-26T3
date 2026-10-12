# %% [markdown]
# # Milestone 3: Transfer Learning & Stronger Segmentation Models
# ## Synthetic Educational Walkthrough (CPU-Only, Zero Competition Data)
#
# This notebook demonstrates the core concepts of Milestone 3:
# 1. Synthesizing an industrial strip ($128 \times 800$) and simulating defect masks
# 2. Extracting multi-scale feature hierarchies through an encoder-decoder
# 3. Adapting single-channel grayscale inputs for pretrained RGB backbones
# 4. Comparing 4-Sigmoid (multi-label) vs 5-Softmax (mutually exclusive) output heads
# 5. Comparing BCEWithLogitsLoss and Soft Dice Loss behavior
# 6. Executing a complete forward/backward optimization step on CPU

# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

print(f"PyTorch Version: {torch.__version__}")
print("Device: CPU (Lightweight Educational Demo)")

# %% [markdown]
# ## 1. Synthetic Strip & Mask Generation
# We synthesize a $128 \times 800$ grayscale strip with longitudinal rolling noise and two defect instances.

# %%
np.random.seed(42)
torch.manual_seed(42)

# Generate synthetic strip with rolling texture
base_strip = np.full((128, 800), 160, dtype=np.uint8)
texture = np.random.normal(0, 10, (128, 800)).astype(np.int16)

# Add synthetic defect A (longitudinal scratch) and defect C (inclusion cluster)
base_strip[30:34, 150:350] = 60    # Defect A: scratch
base_strip[70:95, 500:570] = 40    # Defect C: inclusion
synthetic_img = np.clip(base_strip + texture, 0, 255).astype(np.float32) / 255.0

# Create 4-channel ground truth mask tensor [4, 128, 800]
target_masks = np.zeros((4, 128, 800), dtype=np.float32)
target_masks[0, 30:34, 150:350] = 1.0  # Channel 0: Defect A
target_masks[2, 70:95, 500:570] = 1.0  # Channel 2: Defect C

# Convert to PyTorch tensors [B=1, C, H, W]
image_tensor = torch.from_numpy(synthetic_img).unsqueeze(0).unsqueeze(0)  # [1, 1, 128, 800]
target_tensor = torch.from_numpy(target_masks).unsqueeze(0)               # [1, 4, 128, 800]

print(f"Image Tensor Shape:  {tuple(image_tensor.shape)} (dtype: {image_tensor.dtype})")
print(f"Target Tensor Shape: {tuple(target_tensor.shape)} (dtype: {target_tensor.dtype})")

# %% [markdown]
# ## 2. Grayscale Input Adaptation (3 Strategies)
# Pretrained backbones expect 3 RGB channels. We evaluate:
# - **Strategy 1**: Repeating grayscale values across 3 channels (`x.repeat(1, 3, 1, 1)`)
# - **Strategy 2**: Modifying the first convolution weights ($W_{\text{new}} = \sum W_c$)
# - **Strategy 3**: Prepending a learned $1 \times 1$ adapter (`nn.Conv2d(1, 3, 1x1, bias=False)`)

# %%
conv_rgb = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True)
w_rgb = conv_rgb.weight.data
b_rgb = conv_rgb.bias.data

# Strategy 1: Channel Repeat
x_rep = image_tensor.repeat(1, 3, 1, 1)
out_rep = conv_rgb(x_rep)

# Strategy 2: First-Layer Conv Weight Summing
conv_gray = nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=True)
conv_gray.weight.data = w_rgb.sum(dim=1, keepdim=True)
conv_gray.bias.data = b_rgb.clone()
out_gray = conv_gray(image_tensor)

diff = torch.max(torch.abs(out_rep - out_gray)).item()
print(f"Strategy 1 vs Strategy 2 Max Diff: {diff:.2e} (Mathematically Identical!)")

# Strategy 3: Learned Adapter
adapter = nn.Conv2d(1, 3, kernel_size=1, bias=False)
adapter.weight.data = torch.ones(3, 1, 1, 1)
out_adapted = conv_rgb(adapter(image_tensor))
diff_adapt = torch.max(torch.abs(out_rep - out_adapted)).item()
print(f"Strategy 1 vs Strategy 3 Max Diff: {diff_adapt:.2e}")

# %% [markdown]
# ## 3. Feature Map Shape Tracing Across Encoder-Decoder
# We trace a $128 \times 800$ strip through a 3-stage contracting path and expanding decoder with skip connections.

# %%
class ToyEncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder stages
        self.enc1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Decoder stages (upsampling + concatenation)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = nn.Conv2d(32 + 16, 16, kernel_size=3, padding=1)

        # Head
        self.head = nn.Conv2d(16, 4, kernel_size=1)

    def forward(self, x):
        s1 = F.relu(self.enc1(x))
        p1 = self.pool1(s1)
        s2 = F.relu(self.enc2(p1))
        p2 = self.pool2(s2)
        s3 = F.relu(self.enc3(p2))
        p3 = self.pool3(s3)

        b = F.relu(self.bottleneck(p3))

        d3 = F.relu(self.dec3(torch.cat([self.up3(b), s3], dim=1)))
        d2 = F.relu(self.dec2(torch.cat([self.up2(d3), s2], dim=1)))
        d1 = F.relu(self.dec1(torch.cat([self.up1(d2), s1], dim=1)))

        return self.head(d1)

model = ToyEncoderDecoder()
with torch.no_grad():
    logits = model(image_tensor)

print(f"Encoder-Decoder Output Logits Shape: {tuple(logits.shape)}")
assert logits.shape == (1, 4, 128, 800)

# %% [markdown]
# ## 4. Output Head Semantics: 4-Sigmoid vs 5-Softmax
# We analyze how a single pixel with competing defect logits behaves under independent vs mutually exclusive heads.

# %%
# Synthetic competing pixel logits: [Background, Defect_A, Defect_B, Defect_C, Defect_D]
competing_logits = torch.tensor([[-1.0, 2.5, -4.0, 2.1, -3.0]])

# 4-Channel Sigmoid (ignores background)
sig_probs = torch.sigmoid(competing_logits[:, 1:])
print("4-Channel Sigmoid Probabilities (Defect A..D):")
print([f"{p:.4f}" for p in sig_probs[0].tolist()])
print(f"Probabilities sum: {sig_probs.sum().item():.4f} (> 1.0! Both Defect A and C active)")

# 5-Channel Softmax (includes background)
sm_probs = F.softmax(competing_logits, dim=1)
print("\n5-Channel Softmax Probabilities (Bg, Defect A..D):")
print([f"{p:.4f}" for p in sm_probs[0].tolist()])
print(f"Probabilities sum: {sm_probs.sum().item():.4f} (Exactly 1.0! Defect A wins, Defect C suppressed)")

# %% [markdown]
# ## 5. Loss Functions: BCE vs Soft Dice on Tiny Synthetic Canvas

# %%
def soft_dice_loss(probs, target, smooth=1.0):
    intersection = (probs * target).sum()
    cardinality = probs.sum() + target.sum()
    return 1.0 - (2.0 * intersection + smooth) / (cardinality + smooth)

bce_fn = nn.BCEWithLogitsLoss()
sig_probs = torch.sigmoid(logits)

bce_val = bce_fn(logits, target_tensor)
dice_val = soft_dice_loss(sig_probs, target_tensor)
compound_loss = bce_val + 1.0 * dice_val

print(f"Batch BCE Loss:       {bce_val.item():.4f}")
print(f"Batch Soft Dice Loss: {dice_val.item():.4f}")
print(f"Compound Loss:        {compound_loss.item():.4f}")

# %% [markdown]
# ## 6. Single Forward and Backward Optimization Step
# Verifying that gradients flow back through all layers on CPU.

# %%
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
optimizer.zero_grad()

logits = model(image_tensor)
loss = bce_fn(logits, target_tensor) + soft_dice_loss(torch.sigmoid(logits), target_tensor)
loss.backward()
optimizer.step()

print(f"Optimization Step Completed! Loss: {loss.item():.4f}")
print("✓ Standalone CPU verification complete.")
