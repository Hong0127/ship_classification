import torch
import torch.nn as nn
from .trainer import BaseTrainer
import numpy as np
from tqdm import tqdm

class DomainTrainer(BaseTrainer):
    """도메인 적응 학습 클래스"""
    def __init__(self, model, optimizer, scheduler, criterion, domain_criterion, 
                 device, config):
        super().__init__(model, optimizer, scheduler, criterion, device, config)
        self.domain_criterion = domain_criterion
        
    def train_epoch(self, train_loader, epoch):
        """도메인 적응 학습 에폭"""
        self.model.train()
        total_loss = 0
        cls_losses = 0
        domain_losses = 0
        correct = 0
        total = 0
        
        # 진행률에 따른 alpha 값 계산
        p = float(epoch) / self.config['training']['num_epochs']
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        for batch_idx, data in enumerate(pbar):
            # 소스 도메인 데이터
            source_inputs = data['source']['image'].to(self.device)
            source_labels = data['source']['label'].to(self.device)
            
            # 타겟 도메인 데이터
            target_inputs = data['target']['image'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # 소스 도메인 처리
            s_features, s_outputs, s_domain = self.model(source_inputs, alpha)
            
            # 타겟 도메인 처리
            t_features, _, t_domain = self.model(target_inputs, alpha)
            
            # 도메인 라벨 생성
            source_domain_labels = torch.ones(source_inputs.size(0), 1).to(self.device)
            target_domain_labels = torch.zeros(target_inputs.size(0), 1).to(self.device)
            
            # 손실 계산
            cls_loss = self.criterion(s_outputs, source_labels)
            domain_loss, _, mmd_loss = self.domain_criterion(
                s_features, t_features,
                torch.cat([s_domain, t_domain]),
                torch.cat([source_domain_labels, target_domain_labels])
            )
            
            loss = cls_loss + domain_loss
            
            # 역전파
            loss.backward()
            self.optimizer.step()
            
            # 통계 업데이트
            total_loss += loss.item()
            cls_losses += cls_loss.item()
            domain_losses += domain_loss.item()
            
            _, predicted = s_outputs.max(1)
            total += source_labels.size(0)
            correct += predicted.eq(source_labels).sum().item()
            
            # 진행률 표시 업데이트
            pbar.set_postfix({
                'loss': total_loss/(batch_idx+1),
                'cls_loss': cls_losses/(batch_idx+1),
                'domain_loss': domain_losses/(batch_idx+1),
                'acc': 100.*correct/total
            })
            
        return total_loss/len(train_loader), correct/total
        
    def validate(self, val_loader):
        """도메인 적응 검증"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 순전파 (도메인 적응 없이)
                features, outputs, _ = self.model(inputs, alpha=0)
                loss = self.criterion(outputs, targets)
                
                # 통계 업데이트
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        return total_loss/len(val_loader), correct/total