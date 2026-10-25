"""Minimal toy U-Net step demonstrating the forward, loss, and backward lifecycle on CPU.

Demonstrates:
1. Encoder contracting block with MaxPool2d downsampling.
2. Bottleneck representation.
3. Decoder expanding block with bilinear upsampling and skip concatenation.
4. Output head emitting 4 independent defect logits.
5. Loss calculation (BCEWithLogitsLoss) and optimizer gradient step.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim


class DoubleConv(nn.Module):
    """Two 3x3 convolutions with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MinimalToyUNet(nn.Module):
    """Educational 2-level U-Net for strip input (1, H, W) -> logits (4, H, W)."""

    def __init__(self, in_channels: int = 1, num_classes: int = 4, base_ch: int = 16):
        super().__init__()
        # Encoder Level 1: [B, base_ch, H, W]
        self.enc1 = DoubleConv(in_channels, base_ch)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # [H/2, W/2]

        # Bottleneck: [B, base_ch*2, H/2, W/2]
        self.bottleneck = DoubleConv(base_ch, base_ch * 2)

        # Decoder Level 1: Upsample + Concatenate skip from enc1
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # in_ch = bottleneck_ch (base_ch*2) + skip_ch (base_ch) = base_ch * 3
        self.dec1 = DoubleConv(base_ch * 3, base_ch)

        # 1x1 Convolution Output Head: [B, num_classes, H, W]
        self.out_conv = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        e1 = self.enc1(x)          # [B, 16, 128, 800]
        p1 = self.pool1(e1)        # [B, 16, 64, 400]

        # Bottleneck
        b = self.bottleneck(p1)    # [B, 32, 64, 400]

        # Decoder path with skip connection
        u1 = self.up1(b)           # [B, 32, 128, 800]
        cat1 = torch.cat([u1, e1], dim=1)  # [B, 48, 128, 800]
        d1 = self.dec1(cat1)       # [B, 16, 128, 800]

        # Output logits
        logits = self.out_conv(d1) # [B, 4, 128, 800]
        return logits


def main():
    print("=== Milestone 2: Toy U-Net Optimization Lifecycle Diagnostic ===")

    # Set deterministic seed
    torch.manual_seed(42)

    # 1. Instantiate model and print parameter count
    model = MinimalToyUNet(in_channels=1, num_classes=4, base_ch=16)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Instantiated Educational U-Net with {total_params:,} trainable parameters.")

    # 2. Synthesize a batch of 2 strip images: [B=2, C=1, H=128, W=800]
    batch_images = torch.randn(2, 1, 128, 800, dtype=torch.float32)
    # Synthesize binary targets: [B=2, C=4, H=128, W=800]
    batch_targets = (torch.rand(2, 4, 128, 800) > 0.95).float()
    print(f"\n[Batch Tensor Shapes]")
    print(f"• Input images:  {tuple(batch_images.shape)}")
    print(f"• Target masks:  {tuple(batch_targets.shape)} (positive pixels={batch_targets.sum():.0f})")

    # 3. Forward Pass: compute logits
    logits = model(batch_images)
    print(f"• Output logits: {tuple(logits.shape)} (range: [{logits.min().item():.2f}, {logits.max().item():.2f}])")
    assert logits.shape == (2, 4, 128, 800), "Output logits shape mismatch!"

    # 4. Probabilities calculation
    probs = torch.sigmoid(logits)
    print(f"• Probabilities: range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")

    # 5. Loss calculation (BCEWithLogitsLoss)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, batch_targets)
    print(f"\n[Loss & Optimization]")
    print(f"• Computed BCE Loss: {loss.item():.4f}")

    # 6. Backward pass and optimizer step
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()

    # Measure gradient norm
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2)
    print(f"• Total Gradient Norm: {total_norm.item():.4f}")
    assert total_norm > 0, "Gradients should be non-zero after backward pass!"

    optimizer.step()
    print("• Optimizer step executed successfully!")
    print("\n✓ All U-Net forward, backward, and optimization lifecycle assertions passed!")


if __name__ == "__main__":
    main()
