import optuna
import torch
import numpy as np
from typing import Dict, Any
import logging
from pathlib import Path
import yaml

class HyperparameterOptimizer:
    """하이퍼파라미터 최적화"""
    def __init__(self, base_config: Dict, study_name: str, n_trials: int = 100):
        self.base_config = base_config
        self.study_name = study_name
        self.n_trials = n_trials
        self.logger = logging.getLogger(__name__)
        
        # 연구 생성
        self.study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            pruner=optuna.pruners.MedianPruner()
        )
        
    def optimize(self, train_fn):
        """최적화 실행"""
        self.logger.info(f"Starting hyperparameter optimization: {self.study_name}")
        self.study.optimize(
            lambda trial: self._objective(trial, train_fn),
            n_trials=self.n_trials,
            timeout=72000  # 20시간
        )
        
        # 결과 저장
        self.save_results()
        
        return self.study.best_params, self.study.best_value
        
    def _objective(self, trial: optuna.Trial, train_fn) -> float:
        """목적 함수"""
        # 하이퍼파라미터 탐색 공간 정의
        config = self.base_config.copy()
        
        # 모델 관련 파라미터
        config['model'].update({
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'feature_dim': trial.suggest_categorical('feature_dim', [512, 1024, 2048])
        })
        
        # 학습 관련 파라미터
        config['training'].update({
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),
            'weight_decay': trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)
        })
        
        # 도메인 적응 관련 파라미터
        config['domain_adaptation'].update({
            'lambda_domain': trial.suggest_loguniform('lambda_domain', 0.01, 1.0),
            'lambda_mmd': trial.suggest_loguniform('lambda_mmd', 0.01, 1.0)
        })
        
        try:
            # 학습 실행 및 검증 정확도 반환
            val_acc = train_fn(config)
            
            # 중간 결과 보고
            trial.report(val_acc, step=0)
            
            return val_acc
            
        except Exception as e:
            self.logger.error(f"Trial failed: {str(e)}")
            raise optuna.TrialPruned()
            
    def save_results(self):
        """최적화 결과 저장"""
        save_dir = Path('optimization_results')
        save_dir.mkdir(exist_ok=True)
        
        # 최적 파라미터 저장
        best_params = {
            'best_value': self.study.best_value,
            'best_params': self.study.best_params
        }
        
        with open(save_dir / f'{self.study_name}_best_params.yaml', 'w') as f:
            yaml.dump(best_params, f)
            
    def plot_optimization_history(self):
        """최적화 과정 시각화"""
        save_dir = Path('optimization_results')
        save_dir.mkdir(exist_ok=True)
        
        # 최적화 히스토리
        fig = optuna.visualization.plot_optimization_history(self.study)
        fig.write_html(save_dir / f'{self.study_name}_history.html')
        
        # 파라미터 중요도
        fig = optuna.visualization.plot_param_importances(self.study)
        fig.write_html(save_dir / f'{self.study_name}_importance.html')
        
        # 파라미터 관계
        fig = optuna.visualization.plot_parallel_coordinate(self.study)
        fig.write_html(save_dir / f'{self.study_name}_parallel.html')

class ModelOptimizer:
    """모델 최적화 도구"""
    @staticmethod
    def get_optimizer(model: torch.nn.Module, config: Dict) -> torch.optim.Optimizer:
        """옵티마이저 생성"""
        optimizer_name = config['training'].get('optimizer', 'adamw').lower()
        lr = config['training']['learning_rate']
        weight_decay = config['training']['weight_decay']
        
        if optimizer_name == 'adamw':
            return torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adam':
            return torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
            
    @staticmethod
    def get_scheduler(optimizer: torch.optim.Optimizer, 
                     config: Dict) -> torch.optim.lr_scheduler._LRScheduler:
        """스케줄러 생성"""
        scheduler_name = config['training'].get('scheduler', 'cosine').lower()
        
        if scheduler_name == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=config['training']['scheduler']['T_0'],
                T_mult=config['training']['scheduler']['T_mult'],
                eta_min=config['training']['scheduler']['eta_min']
            )
        elif scheduler_name == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.1,
                patience=5,
                verbose=True
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")