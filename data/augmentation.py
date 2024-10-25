import torch
import torch.nn as nn
import torchvision.transforms as transforms
import random
import numpy as np

class SpectrogramAugmentation:  # 클래스 이름 일치
    """스펙트로그램 데이터 증강"""
    def __init__(self, img_size=224, p=0.5):
        self.p = p
        self.img_size = img_size
        self.training = True  # training mode flag 추가
        
        # 기본 변환
        self.basic_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # 증강 변환
        self.augment_transform = transforms.Compose([
            transforms.RandomApply([
                transforms.Lambda(lambda x: self.time_masking(x))
            ], p=p),
            transforms.RandomApply([
                transforms.Lambda(lambda x: self.freq_masking(x))
            ], p=p),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomApply([
                transforms.Lambda(lambda x: self.add_gaussian_noise(x))
            ], p=p)
        ])
        
    def time_masking(self, img, max_width=50):
        """시간축 마스킹"""
        if isinstance(img, torch.Tensor):
            c, h, w = img.shape
            width = random.randint(0, max_width)
            start = random.randint(0, w - width)
            mask = torch.ones((c, h, w), device=img.device)
            mask[:, :, start:start+width] = 0
            return img * mask
        return img
        
    def freq_masking(self, img, max_height=20):
        """주파수축 마스킹"""
        if isinstance(img, torch.Tensor):
            c, h, w = img.shape
            height = random.randint(0, max_height)
            start = random.randint(0, h - height)
            mask = torch.ones((c, h, w), device=img.device)
            mask[:, start:start+height, :] = 0
            return img * mask
        return img
        
    def add_gaussian_noise(self, img, std=0.01):
        """가우시안 노이즈 추가"""
        if isinstance(img, torch.Tensor):
            noise = torch.randn_like(img) * std
            return torch.clamp(img + noise, 0, 1)
        return img
        
    def __call__(self, img):
        img = self.basic_transform(img)
        if self.training:
            img = self.augment_transform(img)
        return img

class MixupAugmentation:
    """Mixup 증강"""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        
    def __call__(self, batch, labels):
        """배치 단위 mixup"""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        batch_size = len(batch)
        index = torch.randperm(batch_size)
        
        mixed_batch = lam * batch + (1 - lam) * batch[index, :]
        label_a, label_b = labels, labels[index]
        
        return mixed_batch, label_a, label_b, lam