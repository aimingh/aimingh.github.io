---
title: "[boostcamp AI Tech] 학습기록 day12 (week3)"
date: 2021-08-18 20:34:05 -0400
categories:
use_math: true
---

# Pytorch
## AutoGrad & Optimizer
* layer and block
* torch.nn.Module
    * 딥러닝을 구성하는 layer의 기본 class
    * input, output, forward, backward, parameter 정의

* torch.nn.Parameter
    * Tensor 객체의 상속개체
    * nn.Modulr의 attribute가 되면 자동으로 자동미분 대상이 되는 Tensor

* Backward
    * forward의 결과값을 이용한 loss를 기준으로 layer의 있는 parameter들의 미분
    * 자동미분된 값으로 parameter update

## Dataset and dataloader
1. Dataset class
    * 데이터의 입력을 정의

2. DataLoader
    * Data에서 batch를 생성
    학습 직전에 데이터들을 알맞게 변환
    * Tensor변환 및 batch 처리

# 과제
## 필수과제1 custom model 제작
* 어제 못한 hook과 apply를 수행
###  hook
    * 패키지화 된 코드에서 다른이들이 custom 코드를 패키지화 된 코드 중간에서 실행 시킬 수 있는 인터페이스라고 한다.
    * pytorch에서는 module과 tensor 두 가지에서 hook을 제공
```
model = Model()

def module_hook(module, grad_input, grad_output):
    output = []
    total = torch.sum(torch.cat(grad_input))
    for grad_i in grad_input:
        output.append(torch.div(grad_i,total))
    return output

model.register_full_backward_hook(module_hook)
```
* 모듈이 작동될때 전후 언제 작동하는 hook인지 주의가 필요할 것으로 보인다.
* 잘못하면 만들었다가 전체 학습이 꼬이는 결과를 만들겠지만 잘 사용하면 중간값을 모니터링하거나 커스텀 하는게 가능해 보였다.

### apply
* module의 파라미터 초기화 같이 모델 전체에 커스텀 함수를 적용할 때 사용
```
# 바이어스를 추가
def add_bias(module):
    module_name = module.__class__.__name__
    function_name = ['Function_A', 'Function_B', 'Function_C', 'Function_D']
    if module_name in function_name:
        module.b = Parameter(torch.rand(2, 1))

# 파라미터 초기화
def weight_initialization(module):
    module_name = module.__class__.__name__

    if module_name.split('_')[0] == "Function":
        module.W.data.fill_(1.)
        module.b.data.fill_(1.)

# forward를 linear 함수로 변경
def linear_transformation(module):
    module_name = module.__class__.__name__

    def forward(self, x):
        return x @ self.W.T + self.b

    if module_name.split('_')[0] == "Function":
        module.forward = partial(forward, module)


returned_module = model.apply(add_bias)
returned_module = model.apply(weight_initialization)
returned_module = model.apply(linear_transformation)
```
* 바이어스를 추가하고 w 초기화 forward 변경이 한번에 들어간다.
* 파라미터를 초기화하는데 많이 사용된다고 한다.
 


# [피어세션 - 팀회고록](https://hackmd.io/@ai17/BJzCHG5eY)

# 후기
다른것보다 과제를 통해 hook과 apply에 대하여 배웠는데 새로운 개념이었다. 모델이 커스텀이 완성된 시점에서도 적용할 수 있는 도구가 더 생긴 느낌이었다. 이번주는 강의보다 과제가 주가되어 파이토치에 익숙해지는 주간인 느낌이다.