"""Diagnostic implementation of Automatic Mixed Precision (AMP) in PyTorch 2.x.

Demonstrates:
1. Modern PyTorch 2.x torch.amp.autocast and torch.amp.GradScaler syntax.
2. Dynamic gradient scaling mechanics and underflow prevention.
3. Gradient accumulation with microbatches: loss scaling and delayed optimizer updates.
4. Correct unscaling and gradient clipping sequence: scaler.unscale_ -> clip_grad_norm_.
5. Automatic skip of optimizer step when non-finite gradients (Inf/NaN) occur.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import AdamW


class TinySegmentationModel(nn.Module):
    """Tiny convolutional net simulating a 2-level encoder-decoder."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.head = nn.Conv2d(16, 4, kernel_size=1)  # 4 defect channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        return self.head(out)


def main():
    print("=" * 75)
    print("Milestone 4: PyTorch 2.x AMP Lifecycle & Gradient Accumulation Diagnostic")
    print("=" * 75)

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # For CPU, bfloat16 is supported in autocast; for CUDA, float16 is standard
    dtype = torch.bfloat16 if device.type == "cpu" else torch.float16
    amp_enabled = True

    print(f"Hardware Environment: Device = {device.type.upper()}, AMP Dtype = {dtype}")
    print(f"PyTorch Version: {torch.__version__}")

    # Build model, optimizer, scaler
    model = TinySegmentationModel().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # PyTorch 2.x GradScaler constructor accepts device_type
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    # Simulation settings:
    # We want an effective batch size of 8, using microbatch size = 2 and accumulation = 4
    microbatch_size = 2
    accumulation_steps = 4
    effective_batch = microbatch_size * accumulation_steps
    h, w = 64, 400  # scaled synthetic strip dimensions

    print(f"Batch Architecture: Microbatch = {microbatch_size}, Accumulation = {accumulation_steps} -> Effective Batch = {effective_batch}")
    print("\nExecuting 1 Optimization Step Across 4 Accumulated Microbatches:\n")

    optimizer.zero_grad(set_to_none=True)
    criterion = nn.BCEWithLogitsLoss()

    for micro_step in range(1, accumulation_steps + 1):
        # Generate synthetic input and multi-label targets
        x = torch.randn(microbatch_size, 1, h, w, device=device)
        y = (torch.rand(microbatch_size, 4, h, w, device=device) > 0.95).float()

        # 1. Forward pass within torch.amp.autocast context
        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=amp_enabled):
            logits = model(x)
            # Loss must be divided by accumulation_steps so gradients sum to the true average!
            loss = criterion(logits, y)
            scaled_loss = loss / accumulation_steps

        # 2. Backward pass with gradient scaling
        scaler.scale(scaled_loss).backward()

        print(f"  Microbatch {micro_step}/{accumulation_steps}: "
              f"Raw Loss = {loss.item():.4f}, Logits Dtype = {logits.dtype}")

    # 3. Optimizer update sequence
    print("\nExecuting Optimizer Update Step:")
    # A. Unscale gradients before inspecting or clipping norms
    scaler.unscale_(optimizer)

    # B. Clip gradients
    max_norm = 1.0
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    print(f"  1. scaler.unscale_(optimizer): Gradients unscaled to true float values.")
    print(f"  2. clip_grad_norm_: Total gradient norm = {grad_norm.item():.4f} (max ceiling = {max_norm})")

    # C. Step optimizer through scaler
    scaler.step(optimizer)
    print(f"  3. scaler.step(optimizer): Weights updated if no Inf/NaN detected.")

    # D. Update scaler factor for next step
    scaler.update()
    print(f"  4. scaler.update(): Dynamic scale factor adjusted.")

    # E. Zero gradients
    optimizer.zero_grad(set_to_none=True)
    print(f"  5. optimizer.zero_grad(set_to_none=True): Gradient buffers freed to save VRAM.")

    print("\nKey Takeaways:")
    print("• Loss Division: Scaled loss = loss / accumulation_steps prevents gradients")
    print("  from scaling with the number of accumulated steps.")
    print("• Unscaling Order: You MUST call scaler.unscale_(optimizer) BEFORE")
    print("  clip_grad_norm_; otherwise, gradient norms will be evaluated on artificially")
    print("  scaled numbers, causing aggressive and incorrect down-scaling.")
    print("• Automatic Inf/NaN Handling: If an overflow occurs in FP16, scaler.step()")
    print("  skips the optimizer update entirely and scaler.update() halves the scale factor.")
    print("=" * 75)


if __name__ == "__main__":
    main()
