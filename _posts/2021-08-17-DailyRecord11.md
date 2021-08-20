---
title: "[boostcamp AI Tech] 학습기록 day11 (week3)"
date: 2021-08-17 23:11:05 -0400
categories:
use_math: true
---

# Pytorch
## Intoroduction
1. pytorch의 특징
    * Define by Run - pythonic code
    * GPU support, Good API, Good community
    * 간편한 사용법
    * 반면에 TensorFlow는 production과 scalability에서 장점

2. Numpy + AutoGrad + Function
    * Numpy 구조의 Tensor 객체
    * 자동미분 지원으로 Deep learing 연산
    * 다양한 형태의 Deep learning 함수와 모델 지원

## Pytorch Basic
1. Tensor
    * 다차원 array를 표현하는 pyTorch 클래스
    * numpy의 ndarray나 TensorFlow의 Tensor와 동일한 기능
    * Tensor를 생성하는 함수 지원
    ```
    import numpy as np
    import torch

    n_array = np.arange(10).reshape(2,5)
    t_array = torch.FloatTensor(n_array)    # ndarray로 tensor 생성

    data = [[3, 5],[10, 5]]
    x_data = torch.tensor(data)             # list를 이용해도 가능
    nd_array_ex = np.array(data)
    tensor_array = torch.from_numpy(nd_array_ex)    # 다시 ndarray로 변환
    ```
    * pytorch에서 tensor는 ndarray에서 지원하는 사용법이 대부분 적용
    * 가장 큰 차이점은 tensor는 GPU에서 계산을 지원

    * handling
        * view
        * squeeze
        * unsqueeze

    * Operations
        * +, -, *, /, @ 등 기존 연산 지원
        * nn module을 통해 다양한 수식 지원

2. AutoGrad
    * 자동미분으로 backward (backpropagation) 연산 가능
```
import torch

w = torch.tensor(1.0, requires_grad = True)
y = w**2
z = 10*y + 2
z.backward()
w.grad
```

# 과제
## 필수과제1 custom model 제작
* 부덕이와 함께 custom model에 대해 알아보는 유익한 시간
* 양이 엄청 많다
* pytorch의 문서 이용
* pytorch의 기본 연산자들의 사용
* nn.Module을 상속박은 class로 커스텀 모듈 이용
* module_list와 module_dict의 사용
* 3d gather

# [피어세션 - 팀회고록](https://hackmd.io/@ai17/BJyTHJYxY)

# 후기
생각보다 과제가 길었다. 먼가 셀이 숨어있으니까 러시아의 마트료시카 인형이 생각나는 과제였다. 필수과제가 2개니 이번과제를 2번에 나누고 다음과제는 상황에 맞게 나눠서 수행해야겠다. 3d gather를 사용하는게 알 것 같으면서도 모르겠는 느낌에 시간을 너무 오래 사용하였다.