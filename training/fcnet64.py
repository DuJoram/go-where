from torch import nn


class FCNet64(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super(FCNet64, self).__init__()
        self._conv = nn.Sequential(
            # (B, in_channels, 64, 64) -> (B, 4, 64, 64)
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=4,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="zeros",
            ),
            nn.ReLU(),
            # (B, 4, 64, 64) -> (B, 4, 16, 16)
            nn.MaxPool2d(
                kernel_size=4,
                stride=4,
                padding=0,
            ),
            # (B, 4, 16, 16) -> (B, 8, 16, 16)
            nn.Conv2d(
                in_channels=4,
                out_channels=8,
                kernel_size=7,
                stride=1,
                padding=3,
            ),
            nn.ReLU(),
            # (B, 8, 16, 16) -> (B, 8, 8, 8)
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
                padding=0,
            ),
            # (B, 8, 8, 8) -> (B, 16, 8, 8)
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            # (B, 16, 8, 8) -> (B, 16, 4, 4)
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
                padding=0,
            ),
            # (B, 16, 4, 4) -> (B, 32, 4, 4)
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            # (B, 32, 4, 4) -> (B, 32, 2, 2)
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
                padding=0,
            ),
            # (B, 32, 2, 2) -> (B, 64, 2, 2)
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            # (B, 64, 2, 2) -> (B, 64, 1, 1)
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
                padding=0,
            ),
            # (B, 64, 1, 1) -> (B, out_channels, 1, 1)
            nn.Conv2d(
                in_channels=64,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

    def forward(self, x):
        return self._conv(x)
