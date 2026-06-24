"""
lwbna_unet.py — PyTorch implementation of LWBNA-UNet
(Lightweight Bottleneck Narrowing with Attention U-Net),
Sharma et al., Scientific Reports 12:8508 (2022).

Faithful port of the reference Keras model
(github.com/parmanandsharma/Lightweight_AI):
  * fixed width: every conv block in the encoder/decoder uses `f` (=128) channels
  * channel-attention (squeeze-excitation, single FC: sigmoid(relu(W·GAP(x))))
    after every conv block
  * skip connections are ADD (not concat)
  * UpSampling (not transposed conv)
  * bottleneck "channel narrowing": f -> f/2 -> f/4 -> f/8 each with attention,
    then expand back to f and add the first bottleneck feature

Used here to segment the cerebrum on coronal STIR (binary mask), distilled from
SAM2. Input 256x256, depth 4 -> bottleneck 16x16.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """SE-style gate: x * sigmoid(relu(W . GAP(x))) (matches the reference)."""

    def __init__(self, channels: int):
        super().__init__()
        self.fc = nn.Linear(channels, channels)

    def forward(self, x):
        w = x.mean(dim=(2, 3))                 # global average pool -> (B, C)
        w = torch.sigmoid(F.relu(self.fc(w)))
        return x * w[:, :, None, None]


class ConvBlock(nn.Module):
    """Conv-BN-ReLU x2 (+ optional channel attention)."""

    def __init__(self, c_in: int, c_out: int, attn: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.attn = ChannelAttention(c_out) if attn else None

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        if self.attn is not None:
            x = self.attn(x)
        return x


class LWBNAUNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 1,
                 f: int = 128, depth: int = 4, dropout: float = 0.3,
                 attn: bool = True):
        super().__init__()
        self.depth = depth
        self.drop = nn.Dropout2d(dropout)

        # Encoder (fixed width f at every level)
        self.enc = nn.ModuleList()
        c_in = in_channels
        for _ in range(depth):
            self.enc.append(ConvBlock(c_in, f, attn))
            c_in = f

        # Bottleneck: channel narrowing f -> f/2 -> f/4 -> f/8, attention each step
        self.mid_convs = nn.ModuleList()
        self.mid_attn = nn.ModuleList()
        widths = [max(f // (2 ** i), 1) for i in range(depth)]   # 128,64,32,16
        prev = f
        for w in widths:
            self.mid_convs.append(nn.Conv2d(prev, w, 3, padding=1))
            self.mid_attn.append(ChannelAttention(w))
            prev = w
        self.mid_expand = nn.Conv2d(widths[-1], f, 3, padding=1)  # back to f, add xe1
        self.mid_post = ConvBlock(f, f, attn)

        # Decoder (UpSample + add skip + ConvBlock), fixed width f
        self.dec = nn.ModuleList([ConvBlock(f, f, attn) for _ in range(depth)])

        self.head = nn.Conv2d(f, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder: conv block at each level, save its output for the skip
        # connection, then halve the spatial size.
        skips = []
        for blk in self.enc:
            c = blk(x)
            skips.append(c)
            x = self.drop(F.max_pool2d(c, 2))

        # Bottleneck "narrowing": squeeze the channels f -> f/2 -> f/4 -> f/8
        # (attention each step) to distil the features, then widen back to f and
        # add `xe1` (the first, widest bottleneck feature) as a residual.
        xe1 = None
        for i, (conv, att) in enumerate(zip(self.mid_convs, self.mid_attn)):
            x = F.relu(conv(x))
            if i == 0:
                xe1 = x
            x = att(x)
        x = F.relu(self.mid_expand(x))
        x = x + xe1
        x = self.mid_post(x)

        # Decoder: upsample, ADD the matching encoder skip (LWBNA uses add, not
        # concat), then a conv block. Skips are consumed in reverse order.
        for i, blk in enumerate(self.dec):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = x + skips[self.depth - 1 - i]
            x = self.drop(x)
            x = blk(x)

        return self.head(x)        # logits (B, num_classes, H, W)


if __name__ == "__main__":
    m = LWBNAUNet(in_channels=1, num_classes=1)
    n = sum(p.numel() for p in m.parameters())
    y = m(torch.randn(2, 1, 256, 256))
    print(f"LWBNA-UNet params={n/1e6:.2f}M  out={tuple(y.shape)}")
