import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, List, Tuple

class MetricsCalculator:
    """평가 메트릭 계산"""
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        """메트릭 초기화"""
        self.total = 0
        self.correct = 0
        self.predictions = []
        self.targets = []
        self.confidences = []
        
    def update(self, outputs: torch.Tensor, targets: torch.Tensor):
        """메트릭 업데이트"""
        pred = outputs.argmax(dim=1)
        conf = torch.softmax(outputs, dim=1).max(dim=1)[0]
        
        self.total += targets.size(0)
        self.correct += pred.eq(targets).sum().item()
        
        self.predictions.extend(pred.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        self.confidences.extend(conf.cpu().numpy())
        
    def get_accuracy(self) -> float:
        """정확도 계산"""
        return self.correct / self.total if self.total > 0 else 0
        
    def get_class_accuracy(self) -> Dict[int, float]:
        """클래스별 정확도 계산"""
        class_correct = np.zeros(self.num_classes)
        class_total = np.zeros(self.num_classes)
        
        for pred, target in zip(self.predictions, self.targets):
            class_total[target] += 1
            if pred == target:
                class_correct[target] += 1
                
        class_accuracy = {}
        for i in range(self.num_classes):
            if class_total[i] > 0:
                class_accuracy[i] = class_correct[i] / class_total[i]
            else:
                class_accuracy[i] = 0.0
                
        return class_accuracy
        
    def get_confusion_matrix(self) -> np.ndarray:
        """혼동 행렬 계산"""
        return confusion_matrix(
            self.targets, 
            self.predictions, 
            labels=range(self.num_classes)
        )
        
    def get_classification_report(self) -> str:
        """분류 리포트 생성"""
        return classification_report(
            self.targets,
            self.predictions,
            labels=range(self.num_classes),
            digits=4
        )
        
    def get_confidence_stats(self) -> Dict[str, float]:
        """예측 신뢰도 통계"""
        confidences = np.array(self.confidences)
        return {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences)
        }
        
    def get_all_metrics(self) -> Dict:
        """모든 메트릭 반환"""
        return {
            'accuracy': self.get_accuracy(),
            'class_accuracy': self.get_class_accuracy(),
            'confusion_matrix': self.get_confusion_matrix(),
            'classification_report': self.get_classification_report(),
            'confidence_stats': self.get_confidence_stats()
        }

class DomainMetrics:
    """도메인 적응 메트릭"""
    @staticmethod
    def calculate_mmd_distance(source_features: torch.Tensor, 
                             target_features: torch.Tensor) -> float:
        """MMD 거리 계산"""
        from models.domain_adapter import compute_mmd
        return compute_mmd(source_features, target_features).item()
        
    @staticmethod
    def calculate_domain_accuracy(domain_outputs: torch.Tensor, 
                                domain_labels: torch.Tensor) -> float:
        """도메인 분류 정확도 계산"""
        pred = (domain_outputs > 0.5).float()
        return pred.eq(domain_labels).float().mean().item()
        
    @staticmethod
    def calculate_feature_statistics(source_features: torch.Tensor, 
                                   target_features: torch.Tensor) -> Dict:
        """특징 통계 계산"""
        return {
            'source_mean': source_features.mean(dim=0).cpu().numpy(),
            'source_std': source_features.std(dim=0).cpu().numpy(),
            'target_mean': target_features.mean(dim=0).cpu().numpy(),
            'target_std': target_features.std(dim=0).cpu().numpy()
        }