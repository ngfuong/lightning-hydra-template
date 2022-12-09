import torch.nn as nn
import torchvision

class ConvNet_VGG16bn(nn.Module):
    """EmbeddingNet using VGG16 Batch Norm."""

    def __init__(self, num_classes=4096):
        """Initialize EmbeddingNet model."""
        super(ConvNet_VGG16bn, self).__init__()

        # Everything except the last linear layer
        self.vgg16_bn = torchvision.models.vgg16_bn(pretrained=True)
        self.vgg16_bn.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.6),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.6),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        """Forward pass of EmbeddingNet."""

        out = self.vgg16_bn(x)
        return out
