import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class SpectrogramDataset(Dataset):
    """스펙트로그램 데이터셋"""
    def __init__(self, data_dir, transform=None, is_train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        
        # 데이터 파일 목록과 레이블 생성
        self.samples = []
        self.labels = []
        self._load_data()
        
    def _load_data(self):
        """데이터 파일 로드"""
        class_dirs = sorted(os.listdir(self.data_dir))
        for class_idx, class_dir in enumerate(class_dirs):
            class_path = os.path.join(self.data_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
                
            for file_name in os.listdir(class_path):
                if file_name.endswith(('.png', '.jpg')):
                    self.samples.append(os.path.join(class_path, file_name))
                    self.labels.append(class_idx)
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        image_path = self.samples[idx]
        image = Image.open(image_path)
        
        if self.transform:
            image = self.transform(image)
            
        if self.is_train:
            return image, self.labels[idx]
        return image

class DomainDataset(SpectrogramDataset):
    """도메인 적응을 위한 데이터셋"""
    def __init__(self, source_dir, target_dir, transform=None):
        super().__init__(source_dir, transform, is_train=True)
        self.target_dataset = SpectrogramDataset(
            target_dir, transform, is_train=False
        )
        
    def __getitem__(self, idx):
        # 소스 도메인 데이터
        source_img, source_label = super().__getitem__(idx)
        
        # 타겟 도메인 데이터 (랜덤 선택)
        target_idx = torch.randint(len(self.target_dataset), (1,)).item()
        target_img = self.target_dataset[target_idx]
        
        return {
            'source': {
                'image': source_img,
                'label': source_label
            },
            'target': {
                'image': target_img
            }
        }