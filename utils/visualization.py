import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.manifold import TSNE
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Visualizer:
    """시각화 도구"""
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
    def plot_training_history(self, history: Dict[str, List[float]], save_name: str):
        """학습 이력 플롯"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Loss', 'Accuracy',
                'Domain Loss', 'MMD Loss'
            )
        )
        
        # Loss 플롯
        fig.add_trace(
            go.Scatter(y=history['train_loss'], name='Train Loss'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(y=history['val_loss'], name='Val Loss'),
            row=1, col=1
        )
        
        # Accuracy 플롯
        fig.add_trace(
            go.Scatter(y=history['train_acc'], name='Train Acc'),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(y=history['val_acc'], name='Val Acc'),
            row=1, col=2
        )
        
        # Domain Loss 플롯
        fig.add_trace(
            go.Scatter(y=history['domain_loss'], name='Domain Loss'),
            row=2, col=1
        )
        
        # MMD Loss 플롯
        fig.add_trace(
            go.Scatter(y=history['mmd_loss'], name='MMD Loss'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Training History")
        fig.write_html(self.save_dir / f'{save_name}.html')
        
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, 
                            class_names: List[str], save_name: str):
        """혼동 행렬 시각화"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(self.save_dir / f'{save_name}.png')
        plt.close()
        
    def plot_feature_distribution(self, source_features: torch.Tensor,
                                target_features: torch.Tensor,
                                save_name: str):
        """특징 분포 시각화"""
        # TSNE로 차원 축소
        tsne = TSNE(n_components=2, random_state=42)
        combined_features = torch.cat([source_features, target_features], dim=0)
        embedded = tsne.fit_transform(combined_features.cpu().numpy())
        
        # 소스와 타겟 도메인 분리
        source_embedded = embedded[:len(source_features)]
        target_embedded = embedded[len(source_features):]
        
        plt.figure(figsize=(10, 8))
        plt.scatter(
            source_embedded[:, 0],
            source_embedded[:, 1],
            label='Source Domain',
            alpha=0.6
        )
        plt.scatter(
            target_embedded[:, 0],
            target_embedded[:, 1],
            label='Target Domain',
            alpha=0.6
        )
        plt.title('Feature Distribution (t-SNE)')
        plt.legend()
        plt.savefig(self.save_dir / f'{save_name}.png')
        plt.close()
        
    def plot_confidence_distribution(self, confidences: List[float],
                                   predictions: List[int],
                                   true_labels: List[int],
                                   save_name: str):
        """신뢰도 분포 시각화"""
        correct_conf = [conf for conf, pred, true in 
                       zip(confidences, predictions, true_labels)
                       if pred == true]
        wrong_conf = [conf for conf, pred, true in 
                     zip(confidences, predictions, true_labels)
                     if pred != true]
        
        plt.figure(figsize=(10, 6))
        plt.hist(correct_conf, bins=50, alpha=0.5, label='Correct')
        plt.hist(wrong_conf, bins=50, alpha=0.5, label='Wrong')
        plt.title('Prediction Confidence Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(self.save_dir / f'{save_name}.png')
        plt.close()
        
    def plot_ensemble_weights(self, weights_history: List[np.ndarray],
                            save_name: str):
        """앙상블 가중치 변화 시각화"""
        plt.figure(figsize=(10, 6))
        weights_array = np.array(weights_history)
        for i in range(weights_array.shape[1]):
            plt.plot(weights_array[:, i], label=f'Model {i+1}')
        plt.title('Ensemble Weights Evolution')
        plt.xlabel('Iteration')
        plt.ylabel('Weight')
        plt.legend()
        plt.savefig(self.save_dir / f'{save_name}.png')
        plt.close()