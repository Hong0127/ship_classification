import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class GradientReversalFunction(Function):
    """그래디언트 반전 함수"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversalLayer(nn.Module):
    """그래디언트 반전 레이어"""
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class DomainAdapter(nn.Module):
    """도메인 적응 모델"""
    def __init__(self, backbone, feature_dim, num_classes):
        super().__init__()
        self.backbone = backbone
        
        # 도메인 분류기
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )
        
        self.grl = GradientReversalLayer()
        
    def forward(self, x, alpha=1.0):
        self.grl.alpha = alpha
        
        # 특징 추출
        features, class_output = self.backbone(x)
        
        # 도메인 분류
        domain_features = self.grl(features)
        domain_output = self.domain_classifier(domain_features)
        
        return features, class_output, domain_output

def compute_mmd(source_features, target_features, kernel_mul=2.0, kernel_num=5):
    """MMD 손실 계산"""
    batch_size = source_features.size(0)
    kernels = gaussian_kernel(
        source_features, 
        target_features,
        kernel_mul=kernel_mul, 
        kernel_num=kernel_num
    )
    
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    
    loss = torch.mean(XX + YY - XY - YX)
    return loss

def gaussian_kernel(source, target, kernel_mul, kernel_num):
    """가우시안 커널 계산"""
    n_samples = source.size(0) + target.size(0)
    total = torch.cat([source, target], dim=0)
    
    total0 = total.unsqueeze(0).expand(n_samples, n_samples, -1)
    total1 = total.unsqueeze(1).expand(n_samples, n_samples, -1)
    
    L2_distance = ((total0-total1)**2).sum(2)
    
    bandwidth_list = [torch.mean(L2_distance.detach()) * 
                     (kernel_mul ** (i - kernel_num//2))
                     for i in range(kernel_num)]
    
    kernel_val = [torch.exp(-L2_distance / bandwidth) for bandwidth in bandwidth_list]
    
    return sum(kernel_val)

class DomainLoss(nn.Module):
    """도메인 적응 손실"""
    def __init__(self, lambda_domain=0.1, lambda_mmd=0.1):
        super().__init__()
        self.lambda_domain = lambda_domain
        self.lambda_mmd = lambda_mmd
        self.domain_criterion = nn.BCEWithLogitsLoss()
        
    def forward(self, source_features, target_features, 
                domain_output, domain_labels):
        # 도메인 분류 손실
        domain_loss = self.domain_criterion(domain_output, domain_labels)
        
        # MMD 손실
        mmd_loss = compute_mmd(source_features, target_features)
        
        # 전체 손실
        total_loss = (self.lambda_domain * domain_loss + 
                     self.lambda_mmd * mmd_loss)
                     
        return total_loss, domain_loss, mmd_loss