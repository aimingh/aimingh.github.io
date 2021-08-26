---
title: "[boostcamp AI Tech] 학습기록 day16 (week4)"
date: 2021-08-24 19:51:43 -0400
categories:
use_math: true
---

# Dataset
* 전체 과정에서 데이터 수집하고 전처리 하는것이 80% 이상
* Bounding Box
    * 우리가 원하는 물체의 영역을 제외한 나머지는 학습의 노이즈일 수 있다.
    * 4개의 값을 이용하여 bounding box 표현 가능
        * (x, y, width, hight), (x1, y1, x2, y2) 등
* Resize
    * 원본 데이터를 그대로 쓰기에는 너무 많은 연산력이 필요
    * 적당한 크기, 적당한 해상도라도 충분한 성능을 발휘

* Example) APPTOS Blindness Detection
    * 눈에서 질병을 예측하는데 사용하는 데이터셋
    * 특히 의료 이미지 같은 경우 전처리를 했을 경우 성능의 차이가 크게 나타난다.
    * 도메인, 데이터 형식에 따라 다양한 전처리가 다르게 적용 될 수 있다.

* Generalization
    * Bias and Variance
        * High Bias: 데이터가 학습되지 않아 충분히 fitting되지 못한 경우 (underfitting)
        * High Variance: 데이터의 분포가 너무 fitting되어 개개의 데이터에 너무 최적화 되는 경우 (overfitting)

* Train / Validation
    * 학습데이터셋에서 일부를 분리하여 검증하는데 사용
    * 테스트를 하기 학습할 때 학습을 평가하기 위해 학습에 포함되지 않은 데이터셋이 필요

* Data Augmentation
    * 동일한 데이터라도 다양항 상황에 대한 case, status로 다양한 형태로 가공
    * 테스트 데이터에는 학습에 사용된 원본 영상과 다른 여러가지 노이즈가 포함될 수 있다.
        * 눈, 밤, 비, 이동, 회전, 반전 등
    * random crop, flip, rotation, shearing, gaussian noise, ...
    * Albumentations library
        * torchvision보다 말고 사용해 볼만한 augmentation library
        * torchvision보다 성능이 좋다.

# Data generation
* 데이터 생성 능력
    * 병목현상
    * Data generation: 10 batch/s, model 20 batch/s -> 10 batch/s
    * Data generation: 30 batch/s, model 20 batch/s -> 20 batch/s

# torch.utils
* [torch.utils.data](https://pytorch.org/docs/stable/data.html?highlight=torch%20utils%20data#torch.utils.data.DataLoader)
    * Dataset의 기본구조
    * vanilla data를 원하는 형태로 출력
```
import torch.utils,data import Dataset

class MyDataset(Dataset)        # Dataset class를 상속받는다
    def __init__(self):         # 처음 선언되었을 때 필요한 것들
        pass

    def __len__(self):          # 전체 길이를 반환
        pass

    def __getitem__(self, idx): # index 위치의 데이터와 라벨을 반환
        pass
```

* [torch.utils.data.DataLoader](https://pytorch.org/docs/stable/data.html?highlight=torch%20utils%20data#torch.utils.data.DataLoader)
    * 데이터를 불러오는데 도움을 주는 기능들을 미리 묶어논 data loader
    * 데이터를 효율적으로 사용
```
import torch.utils,data import DataLoader

data_loader = DataLoader(
                train_data,                 # 학습데이터    
                batchsize=batchsize,        # batch size
                num_workers=num_workers,    # 쓰레드의 수
                drop_last=True              # batch size와 마지막 batch의 크기가 맞지 않을경우 버림
)
```






# [피어세션 - 팀회고록](https://hackmd.io/qRNCtIf-Rv6qw3PwWZ63eg)

# 후기
