import torch
import torch.nn as nn
from torchvision import models


class MultiTaskModel(nn.Module):
    def __init__(self, num_classes=23, backbone="resnet18", pretrained=True):
        super(MultiTaskModel, self).__init__()

        # ----------------------------
        # Backbone Selection
        # ----------------------------
        if backbone == "resnet18":
            base = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            in_features = base.fc.in_features

        elif backbone == "resnet50":
            base = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )
            in_features = base.fc.in_features

        else:
            raise ValueError("Unsupported backbone. Use 'resnet18' or 'resnet50'.")

        # Remove final FC layer
        self.backbone = nn.Sequential(*list(base.children())[:-1])

        # ----------------------------
        # Regression Head (Days Left)
        # ----------------------------
        self.reg_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

        # ----------------------------
        # Classification Head
        # ----------------------------
        self.cls_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)

        # Flatten if needed
        if len(features.shape) == 4:
            features = torch.flatten(features, 1)

        # Outputs
        reg_output = self.reg_head(features).squeeze(1)
        cls_output = self.cls_head(features)

        return reg_output, cls_output