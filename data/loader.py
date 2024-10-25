from torch.utils.data import DataLoader
from .dataset import SpectrogramDataset, DomainDataset
from .augmentation import SpectrogramAugmentation, MixupAugmentation

class DataLoaderFactory:
    """데이터 로더 생성 팩토리"""
    def __init__(self, config):
        self.config = config
        self.transform = SpectrogramAugmentation(
            img_size=config['data']['img_size'],
            p=config['data']['augment_prob']
        )
        self.mixup = MixupAugmentation(alpha=0.2)
        
    def create_train_loader(self, data_dir):
        """학습 데이터 로더 생성"""
        dataset = SpectrogramDataset(
            data_dir=data_dir,
            transform=self.transform,
            is_train=True
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True
        )
        
    def create_test_loader(self, data_dir):
        """테스트 데이터 로더 생성"""
        dataset = SpectrogramDataset(
            data_dir=data_dir,
            transform=self.transform,
            is_train=False
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True
        )
        
    def create_domain_loader(self, source_dir, target_dir):
        """도메인 적응을 위한 데이터 로더 생성"""
        dataset = DomainDataset(
            source_dir=source_dir,
            target_dir=target_dir,
            transform=self.transform
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True
        )
        
    def apply_mixup(self, batch, labels):
        """Mixup 적용"""
        return self.mixup(batch, labels)