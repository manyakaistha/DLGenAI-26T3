"""Educational Toy Example: Feature Map Shape Tracing through an Encoder-Decoder.

Demonstrates contracting path, bottleneck, expanding path, and skip connection
tensor shapes on a synthetic industrial strip tensor [B, 1, 128, 800] on CPU.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ToyEncoderDecoder(nn.Module):
    """A lightweight 3-stage educational encoder-decoder with skip connections."""
    def __init__(self, in_channels=1, num_classes=4):
        super().__init__()
        # Contracting Path (Encoder)
        self.enc1 = ConvBlock(in_channels, 32)         # [B, 32, 128, 800]
        self.pool1 = nn.MaxPool2d(2, 2)               # -> [B, 32, 64, 400]

        self.enc2 = ConvBlock(32, 64)                  # [B, 64, 64, 400]
        self.pool2 = nn.MaxPool2d(2, 2)               # -> [B, 64, 32, 200]

        self.enc3 = ConvBlock(64, 128)                 # [B, 128, 32, 200]
        self.pool3 = nn.MaxPool2d(2, 2)               # -> [B, 128, 16, 100]

        # Bottleneck
        self.bottleneck = ConvBlock(128, 256)          # [B, 256, 16, 100]

        # Expanding Path (Decoder)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False) # -> [B, 256, 32, 200]
        self.dec3 = ConvBlock(256 + 128, 128)          # Concat with Enc3 skip -> [B, 128, 32, 200]

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False) # -> [B, 128, 64, 400]
        self.dec2 = ConvBlock(128 + 64, 64)            # Concat with Enc2 skip -> [B, 64, 64, 400]

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False) # -> [B, 64, 128, 800]
        self.dec1 = ConvBlock(64 + 32, 32)             # Concat with Enc1 skip -> [B, 32, 128, 800]

        # Output Head
        self.head = nn.Conv2d(32, num_classes, kernel_size=1) # -> [B, 4, 128, 800]

    def forward(self, x):
        print(f"0. Input Strip Shape:         {list(x.shape)}")

        # Encoder forward pass
        s1 = self.enc1(x)
        p1 = self.pool1(s1)
        print(f"1. Encoder Stage 1 (Skip 1):  {list(s1.shape)} -> Pooled: {list(p1.shape)}")

        s2 = self.enc2(p1)
        p2 = self.pool2(s2)
        print(f"2. Encoder Stage 2 (Skip 2):  {list(s2.shape)} -> Pooled: {list(p2.shape)}")

        s3 = self.enc3(p2)
        p3 = self.pool3(s3)
        print(f"3. Encoder Stage 3 (Skip 3):  {list(s3.shape)} -> Pooled: {list(p3.shape)}")

        # Bottleneck
        b = self.bottleneck(p3)
        print(f"4. Bottleneck Feature Map:    {list(b.shape)}")

        # Decoder forward pass with skip concatenations
        d3 = self.up3(b)
        d3 = torch.cat([d3, s3], dim=1)
        d3 = self.dec3(d3)
        print(f"5. Decoder Stage 3 (Cat S3):  {list(d3.shape)}")

        d2 = self.up2(d3)
        d2 = torch.cat([d2, s2], dim=1)
        d2 = self.dec2(d2)
        print(f"6. Decoder Stage 2 (Cat S2):  {list(d2.shape)}")

        d1 = self.up1(d2)
        d1 = torch.cat([d1, s1], dim=1)
        d1 = self.dec1(d1)
        print(f"7. Decoder Stage 1 (Cat S1):  {list(d1.shape)}")

        logits = self.head(d1)
        print(f"8. Output Logits (Final Head): {list(logits.shape)}")
        return logits


if __name__ == "__main__":
    print("=== Educational Feature Map Shape Tracing ===")
    model = ToyEncoderDecoder(in_channels=1, num_classes=4)
    # Synthetic batch of 2 strips (128x800)
    synthetic_strip = torch.randn(2, 1, 128, 800)
    with torch.no_grad():
        out = model(synthetic_strip)
    assert out.shape == (2, 4, 128, 800), "Shape mismatch!"
    print("\n✓ Verification successful: Output dimensions exactly match input spatial canvas!")
