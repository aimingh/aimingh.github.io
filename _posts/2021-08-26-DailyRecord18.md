---
title: "[boostcamp AI Tech] 학습기록 day17 (week4)"
date: 2021-08-26 20:24:37 -0400
categories:
use_math: true
---

#
## Loss
* 신경망의 출력과 정답간의 에러를 정의한다.
* pytorch에서 losss도 nn.Mudule Family여서 학습에서 역전파에 바로 관여한다.
```
output = model(input)
loss = criterion(output, labels)
loss.backward()
optimizer.step()
```
* required_frad=false 옵션을 준 대상의 파라미터는 업데이트 되지 않고 frozen 된다.

* Focal loss
    * classimbalance 문제가 있는 경우 특정 class에 loss를 더 높게 부여

* Label smoothing loss
    * class target label을 onehot표현으로 사용하기보다 smoothing하여 일반화 성능을 높임

## Optimizer
* 가중치에대한 에러의 변화율은 방향을 정의합니다.
* optimizer에서 학습률 (learning rate)은 얼마나 움직일지를 정의합니다.
* LR scheduler
    * learning rate를 동적으로 조절할 수 있는 방법
    * step LR: 특정 step이 지날떄마다 LR을 단계별로 줄여줌
    ```
    schedular = torch.optim.lr_scheduler.StepLR(optimizer,'min)
    ```
    * CosineAnnealingLR: cosine 함수처럼 LR을 급격히 변경, 급격한 변화로 local minima등에 대하여 탈출
    ```
    schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,'min)
    ```
    * ReduceLROnPlateau: 더이상 성능이 향상이 없을 때 LR을 감소
    ```
    schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min)
    ```

## Metric
* 모델을 객관적으로 평가하기 위한 지표
* Classification
    * Accuracy, f1-score, precision, recall, ROC&AUC
* Regression
    * MAE, MSE
* Ranking
    * MRR, NDCG, MAP

## Training Process
```
model.train()
optimizer.zero_grad()

output = model(input)
loss = criterion(output, labels)
loss.backward
optimizer.step()
```
* Gradient Accumulation
    * batch size를 적게 하더라도 gradient를 일정 쌓아서 한번에 업데이트 하는 기법

## Inference Process
```
# pseudo
model.eval()
with torch.no_grad():
    validation
    checkpoint.save()
```

## Appendix
* pytorch lightning

# [피어세션 - 팀회고록]()

# 후기
