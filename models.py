import torch.nn as nn
from torchvision import models

from config import IMG_SIZE, DROPOUT_HEAD, USE_TRANSFER


# Q1 - linear baseline (no spatial structure)
class BaselineLinear(nn.Module):
    def __init__(self, num_classes, img_size=IMG_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * img_size * img_size, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# Q2 - one conv block, no activation
class CNNv1(nn.Module):
    def __init__(self, num_classes, filters=32, img_size=IMG_SIZE):
        super().__init__()
        pooled = img_size // 2
        self.features = nn.Sequential(
            nn.Conv2d(3, filters, 3, padding=1),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Linear(filters * pooled * pooled, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


# Q3 - one conv block + ReLU
class CNNv2(nn.Module):
    def __init__(self, num_classes, filters=32, img_size=IMG_SIZE):
        super().__init__()
        pooled = img_size // 2
        self.features = nn.Sequential(
            nn.Conv2d(3, filters, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Linear(filters * pooled * pooled, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


# Improved model used for Q4-Q10
class CNNImproved(nn.Module):
    # Three conv blocks with doubling filter counts: 32 -> 64 -> 128.
    # AdaptiveAvgPool collapses spatial dims to 1x1, making the head
    # resolution-independent and cutting parameter count vs. raw flatten.
    def __init__(self, num_classes, filters=32, dropout=DROPOUT_HEAD):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )

        self.features = nn.Sequential(
            conv_block(3, filters),
            conv_block(filters, filters * 2),
            conv_block(filters * 2, filters * 4),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(filters * 4, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


# Optional transfer learning backbone
class TransferModel(nn.Module):
    # MobileNetV2 pretrained on ImageNet-1K; feature extractor is frozen,
    # only the classification head is fine-tuned. Typically reaches 90%+
    # accuracy on this dataset after just a few epochs.
    def __init__(self, num_classes, dropout=0.4):
        super().__init__()
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        for p in backbone.features.parameters():
            p.requires_grad = False
        in_feats = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, num_classes),
        )
        self.model = backbone

    def forward(self, x):
        return self.model(x)


def build_model(num_classes, filters=32, use_transfer=USE_TRANSFER):
    if use_transfer:
        print("  Using MobileNetV2 transfer learning backbone")
        return TransferModel(num_classes)
    return CNNImproved(num_classes, filters)
