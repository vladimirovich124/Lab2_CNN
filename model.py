import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()
        # self.model = models.resnet50(pretrained=True)
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)
