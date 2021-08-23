---
title: "[boostcamp AI Tech] 학습기록 day14 (week3)"
date: 2021-08-20 23:23:27 -0400
categories:
use_math: true
---

# Pytorch
## Multi-GPU
* Model Parallel
    * multi-GPU를 사용하기 위해 학습을 분산하는 방법
    * 모델 나누기 - ex) AlexNet
        * 모델의 병목, 파이프라인의 어려움등으로 고난이도 
    * 데이터 나누기 (Data parallel)
        * 데이터를 나눠 GPU에 할당
        * 결과의 평균을 사용
        * DataParallel 
            * 단순히 데이터를 분배한 후 평균을 사용
            * GPU 사용 불균현, batch size 감소, GPU 병목

        * DistributedDataParallel
            * 각 CPU마다 프로세스를 생성하여 GPU에 할당
            * DataParallel을 기본으로 하나하나 개별적으로 연산을 평균

## Hyperparameter Tuning
1. Hyperparameter
    * 모델 스스로 학습하지 못하는 값
    * learning rate, model size, optimizer

2. Grid vs random

3. Ray
    * multi-node multi processing 지원 모듈
    * ML/DL 병렬처리
    * hyperparameter 탐색에 필요한 여러 모듈 제공

## Troubleshooting
* OOM (out of memory)
* GPU에 따른 model size
* 가용 메모리를 확보하기 위해 torch.cuda.empty_cache()
* tensor에 축적되는 변수들을 확인
* 필요 없어진 변수를 삭제할 필요가 있음
* batch size 변경
* torch.no_grad() 사용

# [피어세션 - 팀회고록](https://hackmd.io/@ai17/HkOHZy6eF)

# 후기
이번 주도 디난주처럼 선택과제가 미뤄지게 되어서 아쉬운 느낌이었다. 다만 저번주와 달리 이번주는 과제가 꽤 무거워서 다시 마음을 다잡게 된 주이기도 했다. 과제는 이정도 되야 과제인 느낌이 대학교 졸업 이후에 오랜만에 느끼는 기분이었다.