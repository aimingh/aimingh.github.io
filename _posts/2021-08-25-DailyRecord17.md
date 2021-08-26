---
title: "[boostcamp AI Tech] 학습기록 day17 (week4)"
date: 2021-08-25 14:12:43 -0400
categories:
use_math: true
---

# Model
* 모델이란
    * 모델이란 일반적으로 물체, 사람, 시스템 등의 정보적인 대표입니다.
    * 인공지능 분야에서 모델은 사용하는 인공신경망 그 자체를 의미합니다.
    * 본질적으로 인공신경망을 구성하는 뉴런들과 가중치, 바이어스, 그외 다른 파라미터들에 대한 정보들의 집합입니다.

* pytorch
    * Low-level
    * pythonic
    * flexibility

* nn.Module
    * pytorch의 모든 layer의 기본
    * 모든 레이어는 nn.Module 클래스를 상속받아서 사용된다.
    ```
    import torch.nn as nn

    class MyModule(nn.Module):                  # nn.Module 클래스를 상속
        def __init__(self):             
            super(MyModule, self).__init__()
            pass                                # 추가될 레이어 같은 모듈을 구성 ex) nn.Conv2d
        
        def forward(self, x):
            pass                                # 모듈이 x를 받을 때 실행되는 함수 MyModule(x) == MyModule.forward(x)
    ```
* parameter 
    * model에서 pytorch에서는 nn.Module이 가지고 있는 계산에 사용되는 가중치나 바이어스등의 값들입니다.
    * Parameter 클래스의 가장 큰 특징은 Tensor와 다르게 파라미터는 자동미분의 대상이 된다는 겁니다. (requires_grad)

* Pretrained Model
    * ImageNet과 같은 대용량의 검증된 데이터로 미리 학습된 모델 -> 우리의 목적에 맞도록 fine-tunning
    * 미리 학습된 모델을 자신의 목적에 맞게 사용하는 것입니다.
    * 미리 학습된 모델을 사용하명 학습이 빨라져 시간적으로 매우 효율적입니다.
    * torchvision.models를 이용하여 pretrained model을 쉽게 가져올 수 있습니다.
    ```
    import torchvision.models as models

    alexnet =  models.alexnet(pretrained=True)
    vgg16 =  models.vgg16(pretrained=True)
    resnet18 =  models.resnet18(pretrained=True)
    densenet =  models.densenet161(pretrained=True)
    squeezenet =  models.squeezenet1_0(pretrained=True)
    googlenet =  models.googlenet(pretrained=True)
    monilenet =  models.monilenet_v2(pretrained=True)
    inception =  models.inception_v3(pretrained=True)
    ```

* Transfer Learning
    * CNN base 모델 구조
    * 데이터와 모델의 유사성
    * Case by Case
    * Input -> CNN Backbone -> Classifier -> Output(1000)
    * 비슷하고 많은데이터 -> Classifier update
    * 비슷하지 않고 많은데이터 -> CNN Backbone, Classifier update
    * 적은데이터 -> Classifier update

# [피어세션 - 팀회고록](https://hackmd.io/@ai17/BJYTbOQ-t)

# 후기
