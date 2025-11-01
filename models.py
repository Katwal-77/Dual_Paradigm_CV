"""
Model architectures for Weather Classification
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights

import config


class CustomCNN(nn.Module):
    """Custom CNN architecture for weather classification"""
    
    def __init__(self, num_classes=16):
        super(CustomCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResNetClassifier(nn.Module):
    """ResNet-based classifier with transfer learning"""
    
    def __init__(self, num_classes=16, pretrained=True):
        super(ResNetClassifier, self).__init__()

        # Load pretrained ResNet50
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.model = models.resnet50(weights=weights)

        # Replace final layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


class EfficientNetClassifier(nn.Module):
    """EfficientNet-based classifier with transfer learning"""
    
    def __init__(self, num_classes=16, pretrained=True):
        super(EfficientNetClassifier, self).__init__()

        # Load pretrained EfficientNet-B0
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.model = models.efficientnet_b0(weights=weights)

        # Replace final layer
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


def get_model(model_name='resnet50', num_classes=16, pretrained=True):
    """
    Factory function to get the specified model
    
    Args:
        model_name: Name of the model ('resnet50', 'efficientnet_b0', 'custom_cnn')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights (for transfer learning models)
    
    Returns:
        PyTorch model
    """
    
    if model_name == 'resnet50':
        print("🔧 Loading ResNet50 model...")
        model = ResNetClassifier(num_classes=num_classes, pretrained=pretrained)
        
    elif model_name == 'efficientnet_b0':
        print("🔧 Loading EfficientNet-B0 model...")
        model = EfficientNetClassifier(num_classes=num_classes, pretrained=pretrained)
        
    elif model_name == 'custom_cnn':
        print("🔧 Loading Custom CNN model...")
        model = CustomCNN(num_classes=num_classes)
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    return model


if __name__ == "__main__":
    # Test models
    print("\n=== Testing Custom CNN ===")
    model1 = get_model('custom_cnn', num_classes=16, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model1(x)
    print(f"Output shape: {out.shape}\n")
    
    print("=== Testing ResNet50 ===")
    model2 = get_model('resnet50', num_classes=16, pretrained=False)
    out = model2(x)
    print(f"Output shape: {out.shape}\n")

