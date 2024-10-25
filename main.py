import argparse
import yaml
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from data.dataset import SpectrogramDataset, DomainDataset
from data.augmentation import SpectrogramAugmentation
from models.backbone import ModelFactory
from models.domain_adapter import DomainAdapter
from models.ensemble import EnsembleModel, StackingEnsemble
from trainers.trainer import BaseTrainer
from trainers.domain_trainer import DomainTrainer
from utils.metrics import MetricsCalculator
from utils.visualization import Visualizer
from utils.optimizer import HyperparameterOptimizer, ModelOptimizer

def setup_logging(config):
    """로깅 설정"""
    log_dir = Path(config['paths']['log_dir'])
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'train.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """설정 파일 로드"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def prepare_data(config):
    """데이터 준비"""
    transform = SpectrogramAugmentation(
        img_size=config['data']['img_size'],
        p=config['data']['augment_prob']
    )
    
    # 학습 데이터셋
    train_dataset = DomainDataset(
        source_dir=config['paths']['train_data'],
        target_dir=config['paths']['test_data'],
        transform=transform
    )
    
    # 검증 데이터셋
    val_dataset = SpectrogramDataset(
        data_dir=config['paths']['test_data'],
        transform=transform,
        is_train=False
    )
    
    # 데이터 로더
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    return train_loader, val_loader

def train(config: dict):
    """학습 실행"""
    logger = setup_logging(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 데이터 로더 준비
    train_loader, val_loader = prepare_data(config)
    
    # 모델 생성
    models = []
    optimizers = []
    schedulers = []
    
    for model_config in config['ensemble']['models']:
        # 기본 모델 생성
        base_model = ModelFactory.create_model(config)
        
        # 도메인 적응 모델 생성
        model = DomainAdapter(
            backbone=base_model,
            feature_dim=model_config['feature_dim'],
            num_classes=config['model']['num_classes']
        ).to(device)
        
        # 옵티마이저와 스케줄러 생성
        optimizer = ModelOptimizer.get_optimizer(model, config)
        scheduler = ModelOptimizer.get_scheduler(optimizer, config)
        
        models.append(model)
        optimizers.append(optimizer)
        schedulers.append(scheduler)
    
    # 앙상블 모델 생성
    ensemble = EnsembleModel(models, device)
    
    # 트레이너 초기화
    trainer = DomainTrainer(
        model=ensemble,
        optimizer=optimizers[0],  # 첫 번째 옵티마이저 사용
        scheduler=schedulers[0],  # 첫 번째 스케줄러 사용
        criterion=torch.nn.CrossEntropyLoss(),
        domain_criterion=torch.nn.BCEWithLogitsLoss(),
        device=device,
        config=config
    )
    
    # 시각화 도구 초기화
    visualizer = Visualizer(config['paths']['save_dir'])
    
    # 학습 실행
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=config['training']['num_epochs']
    )
    
    # 학습 결과 시각화
    visualizer.plot_training_history(history, 'training_history')
    
    return trainer.best_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--optimize', action='store_true')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.optimize:
        # 하이퍼파라미터 최적화
        optimizer = HyperparameterOptimizer(
            base_config=config,
            study_name='ship_classification',
            n_trials=100
        )
        best_params, best_value = optimizer.optimize(train)
        print(f"Best accuracy: {best_value:.4f}")
        print("Best parameters:", best_params)
        
        # 최적화 결과 시각화
        optimizer.plot_optimization_history()
    else:
        # 일반 학습
        best_acc = train(config)
        print(f"Best accuracy: {best_acc:.4f}")

if __name__ == '__main__':
    main()