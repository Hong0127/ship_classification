import torch
import torch.nn as nn
import torchvision.models as models

class BaseModel(nn.Module):
    """기본 모델 클래스"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self._create_backbone()
        self.classifier = self._create_classifier()
        
    def _create_backbone(self):
        """백본 모델 생성"""
        backbone_type = self.config['model']['backbone_type']
        pretrained = self.config['model']['pretrained']
        
        if backbone_type == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
            backbone.fc = nn.Identity()  # FC 레이어 제거
        elif backbone_type == 'efficientnet_b4':
            backbone = models.efficientnet_b4(pretrained=pretrained)
            backbone.classifier = nn.Identity()
        elif backbone_type == 'resnext50':
            backbone = models.resnext50_32x4d(pretrained=pretrained)
            backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")
            
        return backbone
        
    def _create_classifier(self):
        """분류기 생성"""
        feature_dim = self.config['model']['feature_dim']
        num_classes = self.config['model']['num_classes']
        
        return nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = torch.mean(features, dim=(2, 3))  # Global Average Pooling
        output = self.classifier(features)
        return features, output
        
    def get_features(self, x):
        """특징 추출"""
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = torch.mean(features, dim=(2, 3))
        return features

class ModelFactory:
    """모델 생성 팩토리"""
    @staticmethod
    def create_model(config):
        """설정에 따른 모델 생성"""
        return BaseModel(config)