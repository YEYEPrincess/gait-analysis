import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Two consecutive Conv2D + BatchNorm + ReLU blocks
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    """
    Small U-Net for insole pressure maps

    Input  : (B, 2, 33, 15)
    Output : (B, 1, 33, 15)
    """

    def __init__(self, in_ch=2, out_ch=1, base=32):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base * 2, base * 4)

        # Decoder
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        # Output layer
        self.out = nn.Conv2d(base, out_ch, kernel_size=1)

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Decoder
        u2 = self.up2(b)
        u2 = self._align(u2, e2)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        u1 = self._align(u1, e1)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        # Output (logits)
        y = self.out(d1)

        return y


    @staticmethod
    def _align(src, ref):
        """
        Align src spatial size to ref using padding or cropping
        """
        _, _, h, w = ref.shape

        pad_h = max(0, h - src.shape[-2])
        pad_w = max(0, w - src.shape[-1])

        src = F.pad(src, (0, pad_w, 0, pad_h))

        return src[:, :, :h, :w]