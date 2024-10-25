import torch
import torch.nn as nn
from tqdm import tqdm
import logging
import time
from pathlib import Path

class BaseTrainer:
    """기본 학습 클래스"""
    def __init__(self, model, optimizer, scheduler, criterion, device, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.config = config
        
        self.start_epoch = 0
        self.best_acc = 0
        self.logger = logging.getLogger(__name__)
        
        # 체크포인트 저장 경로
        self.save_dir = Path(config['paths']['save_dir'])
        self.save_dir.mkdir(exist_ok=True)
        
    def train(self, train_loader, val_loader, num_epochs):
        """학습 실행"""
        for epoch in range(self.start_epoch, num_epochs):
            # 학습
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # 검증
            val_loss, val_acc = self.validate(val_loader)
            
            # 학습률 조정
            self.scheduler.step()
            
            # 로깅
            self.logger.info(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
            )
            
            # 최상의 모델 저장
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_checkpoint(epoch, val_acc, is_best=True)
                
            # 정기적인 체크포인트 저장
            if (epoch + 1) % self.config['training'].get('save_interval', 5) == 0:
                self.save_checkpoint(epoch, val_acc)
                
    def train_epoch(self, train_loader, epoch):
        """한 에폭 학습"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Mixup 적용 (설정된 경우)
            if self.config['training'].get('use_mixup', False):
                inputs, targets_a, targets_b, lam = self.mixup(inputs, targets)
                
            self.optimizer.zero_grad()
            
            # 순전파
            features, outputs = self.model(inputs)
            
            # 손실 계산
            if self.config['training'].get('use_mixup', False):
                loss = self.mixup_criterion(
                    outputs, targets_a, targets_b, lam
                )
            else:
                loss = self.criterion(outputs, targets)
                
            # 역전파
            loss.backward()
            self.optimizer.step()
            
            # 통계 업데이트
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            if not self.config['training'].get('use_mixup', False):
                correct += predicted.eq(targets).sum().item()
                
            # 진행률 표시 업데이트
            pbar.set_postfix({
                'loss': total_loss/(batch_idx+1),
                'acc': 100.*correct/total if not self.config['training'].get('use_mixup', False) else 0
            })
            
        return total_loss/len(train_loader), correct/total
        
    def validate(self, val_loader):
        """검증"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 순전파
                features, outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # 통계 업데이트
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        return total_loss/len(val_loader), correct/total
        
    def mixup_criterion(self, pred, y_a, y_b, lam):
        """Mixup 손실 계산"""
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)
        
    def mixup(self, x, y):
        """Mixup 적용"""
        lam = np.random.beta(0.2, 0.2)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
        
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
            'val_acc': val_acc
        }
        
        # 일반 체크포인트 저장
        save_path = self.save_dir / f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, save_path)
        
        # 최상의 모델 저장
        if is_best:
            best_path = self.save_dir / 'model_best.pth'
            torch.save(checkpoint, best_path)
            
    def load_checkpoint(self, checkpoint_path):
        """체크포인트 로드"""
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_acc = checkpoint['best_acc']
        
        return checkpoint['val_acc']