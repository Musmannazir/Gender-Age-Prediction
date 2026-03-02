import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG16_Weights
from torchvision import transforms


class GenderAgeVGG16(nn.Module):
    def __init__(self, dropout: float = 0.3, unfreeze_last_blocks: int = 0):
        super().__init__()
        backbone = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        for param in backbone.features.parameters():
            param.requires_grad = False

        if unfreeze_last_blocks > 0:
            maxpool_indices = [
                idx for idx, layer in enumerate(backbone.features) if isinstance(layer, nn.MaxPool2d)
            ]
            blocks_to_unfreeze = min(unfreeze_last_blocks, len(maxpool_indices))
            if blocks_to_unfreeze > 0:
                start_idx = maxpool_indices[-blocks_to_unfreeze] + 1
                for layer in backbone.features[start_idx:]:
                    for param in layer.parameters():
                        param.requires_grad = True

        backbone.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        backbone.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.backbone = backbone
        self.gender_head = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())
        self.age_head = nn.Linear(256, 1)

    def forward(self, x):
        features = self.backbone.features(x)
        pooled = self.backbone.avgpool(features)
        flattened = torch.flatten(pooled, 1)
        embedding = self.backbone.classifier(flattened)

        gender = self.gender_head(embedding)
        age = self.age_head(embedding)
        return gender, age


def build_train_transforms(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.04),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_eval_transforms(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_transforms(image_size: int = 224):
    return build_eval_transforms(image_size)


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen
