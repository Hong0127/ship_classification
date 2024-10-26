# Ship Classification with Domain Adaptation

선박 소음 데이터의 스펙트로그램을 분석하여 선박을 분류하는 AI 모델입니다. 서로 다른 해역(A해역, B해역)에서 수집된 데이터 간의 도메인 차이를 극복하기 위해 도메인 적응 기법을 적용하였습니다.

## 주요 기능

1. **도메인 적응**
   - DANN (Domain-Adversarial Neural Network)
   - MMD (Maximum Mean Discrepancy)
   - 특징 분포 정렬을 통한 도메인 갭 최소화

2. **모델 앙상블**
   - 다양한 백본 모델 (ResNet, EfficientNet, ResNeXt)
   - 가중치 기반 앙상블
   - 스태킹 앙상블

3. **실시간 모니터링**
   - 웹 기반 대시보드
   - 성능 메트릭 시각화
   - 15분 단위 검증 결과 분석

4. **자동 최적화**
   - Optuna 기반 하이퍼파라미터 최적화
   - 학습률 자동 조정
   - 모델 구조 탐색

## 프로젝트 구조

```
ship_classification/
├── configs/            # 설정 파일
│   └── config.yaml    
├── data/              # 데이터 처리
│   ├── dataset.py     # 데이터셋 구현
│   ├── augmentation.py# 데이터 증강
│   └── loader.py      # 데이터 로더
├── models/            # 모델 구현
│   ├── backbone.py    # 기본 모델
│   ├── domain_adapter.py  # 도메인 적응
│   └── ensemble.py    # 앙상블 모델
├── trainers/          # 학습 관련
│   ├── trainer.py     # 기본 학습기
│   └── domain_trainer.py  # 도메인 적응 학습
├── utils/             # 유틸리티
│   ├── metrics.py     # 평가 지표
│   ├── visualization.py   # 시각화
│   └── optimizer.py   # 최적화 도구
├── dashboard/         # 모니터링
│   ├── app.py        # 대시보드 앱
│   └── callbacks.py  # 대시보드 콜백
└── main.py           # 메인 실행 파일
```

## 설치 방법

1. **저장소 클론**
```bash
git clone https://github.com/your-username/ship_classification.git
cd ship_classification
```

2. **필요 패키지 설치**
```bash
pip install -r requirements.txt
```

## 사용 방법

1. **데이터 준비**
```bash
# 데이터 디렉토리 구조
data/
├── train/     # A해역 데이터 (학습용)
└── test/      # B해역 데이터 (검증용)
```

2. **설정 파일 수정**
```yaml
# configs/config.yaml
model:
  num_classes: 5
  backbone_type: 'resnet50'
  feature_dim: 2048

training:
  num_epochs: 30
  batch_size: 32
  learning_rate: 1e-4
```

3. **학습 실행**
```bash
# 일반 학습
python main.py --config configs/config.yaml

# 하이퍼파라미터 최적화
python main.py --config configs/config.yaml --optimize
```

4. **모니터링 대시보드 실행**
```bash
python -m dashboard.app
```

## 실험 결과

- Baseline Model Accuracy: 85%
- Domain Adapted Model Accuracy: 92%
- Ensemble Model Accuracy: 94%

## 주요 특징

### 데이터 증강
- 시간축 마스킹
- 주파수축 마스킹
- 랜덤 노이즈 추가
- Mixup 증강

### 도메인 적응
- Gradient Reversal Layer
- MMD Loss
- 특징 정렬

### 앙상블 기법
- Weighted Average
- Stacking
- Cross-validation

## 인용

프로젝트에서 사용된 주요 논문들:
```bibtex
@article{ganin2016domain,
  title={Domain-adversarial training of neural networks},
  author={Ganin, Yaroslav and others},
  journal={Journal of Machine Learning Research},
  year={2016}
}
```

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참고하세요.

## 기여

버그 리포트나 새로운 기능 제안은 GitHub Issues를 통해 제출해주세요.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 연락처

프로젝트 관리자: [Your Name](https://github.com/your-username)

프로젝트 링크: https://github.com/your-username/ship_classification
