import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EnsembleModel(nn.Module):
    """앙상블 모델"""
    def __init__(self, models, device):
        super().__init__()
        self.models = models
        self.device = device
        self.weights = nn.Parameter(
            torch.ones(len(models)) / len(models)
        )
        
    def forward(self, x):
        predictions = []
        features_list = []
        
        for model in self.models:
            features, outputs = model(x)
            predictions.append(F.softmax(outputs, dim=1))
            features_list.append(features)
            
        # 가중치 정규화
        weights = F.softmax(self.weights, dim=0)
        
        # 가중 평균 예측
        ensemble_pred = torch.zeros_like(predictions[0])
        for w, pred in zip(weights, predictions):
            ensemble_pred += w * pred
            
        # 특징 벡터의 가중 평균
        ensemble_features = torch.zeros_like(features_list[0])
        for w, feat in zip(weights, features_list):
            ensemble_features += w * feat
            
        return ensemble_features, ensemble_pred
        
    def update_weights(self, val_loader, criterion):
        """검증 성능 기반 가중치 업데이트"""
        accuracies = []
        
        for model in self.models:
            acc = self.evaluate_single_model(model, val_loader, criterion)
            accuracies.append(acc)
            
        # Softmax를 통한 가중치 계산
        self.weights.data = F.softmax(torch.tensor(accuracies), dim=0)
        
    @torch.no_grad()
    def evaluate_single_model(self, model, val_loader, criterion):
        """단일 모델 평가"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            _, outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        return correct / total

class StackingEnsemble(nn.Module):
    """스태킹 앙상블"""
    def __init__(self, models, num_classes, feature_dim, device):
        super().__init__()
        self.models = models
        self.device = device
        
        # 메타 분류기
        self.meta_classifier = nn.Sequential(
            nn.Linear(len(models) * num_classes, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        base_predictions = []
        
        # 기본 모델들의 예측 수집
        for model in self.models:
            model.eval()
            with torch.no_grad():
                _, outputs = model(x)
                base_predictions.append(F.softmax(outputs, dim=1))
                
        # 예측들을 연결
        stacking_features = torch.cat(base_predictions, dim=1)
        
        # 메타 분류기로 최종 예측
        return self.meta_classifier(stacking_features)