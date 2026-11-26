"""Small four-block CNN baseline trained from scratch."""

from torch import nn


class BaselineCNN(nn.Module):
    """Four Conv-BN-ReLU-pool blocks, global pooling, dropout, linear head."""

    def __init__(self, num_classes: int, dropout: float = 0.30):
        super().__init__()
        channels = (16, 32, 64, 96)
        layers = []
        in_channels = 3
        for out_channels in channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            ])
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(channels[-1], num_classes))

    def forward(self, images):
        return self.classifier(self.pool(self.features(images)))
