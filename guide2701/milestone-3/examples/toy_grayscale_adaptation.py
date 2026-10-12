"""Educational Toy Example: Grayscale Input Adaptation Strategies.

Demonstrates and verifies:
1. Strategy 1: Channel repeat (1 -> 3)
2. Strategy 2: First-layer conv weight collapse (sum and average)
3. Strategy 3: Learned 1x1 conv adapter
Proves exact numerical response equivalence between channel repeat and conv weight summing.
"""
import torch
import torch.nn as nn


def demonstrate_grayscale_adaptation():
    torch.manual_seed(42)

    # 1. Define synthetic 1-channel grayscale patch [B=1, C=1, H=8, W=8]
    x_gray = torch.randn(1, 1, 8, 8)

    # 2. Suppose a pretrained first convolution has weights [out_ch=4, in_ch=3, k=3, k=3] and bias [4]
    out_channels = 4
    conv3 = nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=3, padding=1, bias=True)
    w_rgb = conv3.weight.data  # shape: [4, 3, 3, 3]
    b_rgb = conv3.bias.data    # shape: [4]

    # --- Strategy 1: Channel Repeat (1 -> 3) ---
    x_repeat = x_gray.repeat(1, 3, 1, 1)  # shape: [1, 3, 8, 8]
    out_strat1 = conv3(x_repeat)

    # --- Strategy 2A: Weight Summing in 1-Channel Conv ---
    # W_sum = W_R + W_G + W_B
    conv1_sum = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=3, padding=1, bias=True)
    conv1_sum.weight.data = w_rgb.sum(dim=1, keepdim=True)  # shape: [4, 1, 3, 3]
    conv1_sum.bias.data = b_rgb.clone()
    out_strat2_sum = conv1_sum(x_gray)

    # Check exact numerical equivalence
    diff_sum = torch.max(torch.abs(out_strat1 - out_strat2_sum)).item()
    print(f"Max Absolute Difference (Repeat 3x vs. Weight Sum): {diff_sum:.2e}")
    assert diff_sum < 1e-6, "Weight sum should be mathematically identical to 3x repeat!"

    # --- Strategy 2B: Weight Averaging in 1-Channel Conv ---
    # W_avg = (W_R + W_G + W_B) / 3
    conv1_avg = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=3, padding=1, bias=True)
    conv1_avg.weight.data = w_rgb.mean(dim=1, keepdim=True)  # shape: [4, 1, 3, 3]
    conv1_avg.bias.data = b_rgb.clone()
    out_strat2_avg = conv1_avg(x_gray)
    # Scaled by exactly 1/3 (excluding bias)
    diff_avg_scale = torch.max(torch.abs((out_strat1 - b_rgb.view(1, 4, 1, 1)) / 3.0 - (out_strat2_avg - b_rgb.view(1, 4, 1, 1)))).item()
    print(f"Max Absolute Difference (Averaging scale consistency): {diff_avg_scale:.2e}")

    # --- Strategy 3: Learned 1x1 Conv Adapter ---
    # Prepend nn.Conv2d(1, 3, kernel_size=1, bias=False)
    adapter = nn.Conv2d(1, 3, kernel_size=1, bias=False)
    # Initialize adapter weights to 1.0 so adapter(x_gray) == x_repeat
    adapter.weight.data = torch.ones(3, 1, 1, 1)
    adapted_x = adapter(x_gray)
    out_strat3 = conv3(adapted_x)
    diff_adapter = torch.max(torch.abs(out_strat1 - out_strat3)).item()
    print(f"Max Absolute Difference (Repeat 3x vs. Initialized Adapter): {diff_adapter:.2e}")
    assert diff_adapter < 1e-6, "Initialized adapter must match 3x repeat!"

    print("\n✓ Verification successful: All three grayscale adaptation strategies verified on CPU!")


if __name__ == "__main__":
    demonstrate_grayscale_adaptation()
