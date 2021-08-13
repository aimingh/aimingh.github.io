---
layout: post
title: "[boostcamp AI Tech] 학습기록 day05"
date: 2021-08-06 22:24:05 -0400
categories:
use_math: true
---

# AI Math
## 9. CNN 기초
* CNN (Convolutional Neural Network)이란?
* 기본적인 다층신경망(MLP)과 달리 커널에 의해 완전히 연결된 (fully connected) 구조가 아니라 커널 (kernel)이 움직이며 컨볼루션 (convolution) 연산하는 convolutional layer를 이용하는 신경망입니다.

* 컨볼루션 연산
    * 컨볼루션 연산은 로컬한 영역의 신호를 필터를 이용하여 증폭하거나 축소합니다.
    $$
    \begin{aligned}
    &{[f * g](x)=\int_{\mathbb{R}^{d}} f(z) g(x-z) \mathrm{d} z=\int_{\mathbb{R}^{d}} f(x-z) g(z) \mathrm{d} z=[g * f](x)} \\
    &{[f * g](i)=\sum_{a \in \mathbb{Z}^{d}} f(a) g(i-a)=\sum_{a \in \mathbb{Z}^{d}} f(i-a) g(a)=[g * f](i)}
    \end{aligned}
    $$
    * CNN에서 사실 컨볼루션이 아니라 cross-correlation을 사용합니다. 
        * convolution 연사을 하게 되면 필터/커널을 반전하여 적용해야 하는데 학습하는 과정에서 어차피 반전 요소가 학습되기 때문입니다.
        * 그래서 반전을 하는 과정을 생략한 cross-correlation을 사용하여 학습합니다.

    * 다른 차원에서의 컨볼루션 연산
    $$
    \begin{aligned}
    &[f * g](i)=\sum_{p=1}^{d} f(p) g(i+p)\\
    &[f * g](i, j)=\sum_{p, q} f(p, q) g(i+p, j+q)\\
    &[f * g](i, j, k)=\sum_{p, q, r} f(p, q, r) g(i+p, j+q, k+r)
    \end{aligned}
    $$

    * 입력크기 $(H, W)$, 커널크기 $\left(K_{H}, K_{W}\right)$, 출력크기 $\left(O_{H}, O_{W}\right)$의 관계는 아래와 같습니다.
    $$
    \begin{aligned}
    &O_{H}=H-K_{H}+1 \\
    &O_{W}=W-K_{W}+1
    \end{aligned}
    $$

* Convolution 연산의 역전파
    * 모든 층에서 컨볼루션 연산이 사용되므로 역전파를 계산할때도 컨볼루션 연산이 이용됩니다.
    $$
    \begin{aligned}
    \frac{\partial}{\partial x}[f * g](x) &=\frac{\partial}{\partial x} \int_{\mathbb{R}^{d}} f(y) g(x-y) \mathrm{d} y \\
    &=\int_{\mathbb{R}^{d}} f(y) \frac{\partial g}{\partial x}(x-y) \mathrm{d} y \\
    &=\left[f * g^{\prime}\right](x)
    \end{aligned}
    $$

    * 각 커널에 들어오는 모든 그레디언트를 더하면 convolution 연산의 형태가 나옵니다.
    $$
    \frac{\partial \mathcal{L}}{\partial w_{i}}=\sum_{j} \delta_{j} x_{i+j-1}
    $$

## 10. RNN 기초
* RNN (Recurrent Neural Networks)이란?
    * 이미지와 같은 데이터와 달리 시퀀스 데이터를 처리하기 위하여 순환 구조를 활용한 신경망입니다.
    * 시퀀스 데이터란?
        * 소리,  문자열 등 순서가 결과에 영향을 주는 데이터 형태입니다.
        * 순서가 바뀌거나 과거 정보에 손실이 바뀌면 데이터의 확률분포가 바뀔 수 있습니다.

* 조건부 확률
    * 이전 시퀀스의 정보로 앞으로의 정보를 다루기 때문에 이용합니다.
    $$
    \begin{aligned}
    P\left(X_{1}, \ldots, X_{t}\right) &=P\left(X_{t} \mid X_{1}, \ldots, X_{t-1}\right) P\left(X_{1}, \ldots, X_{t-1}\right) \\
    &=P\left(X_{t} \mid X_{1}, \ldots, X_{t-1}\right) P\left(X_{t-1} \mid X_{1}, \ldots, X_{t-2}\right) \times \\
    &=\prod_{s=1}^{t} P\left(X_{s} \mid X_{s-1}, \ldots, X_{1}\right)
    \end{aligned}
    $$

* RNN
    * 이전 순서의 결과가 현재 입력과 함께 결과를 내는 형태입니다.
    $$
    \begin{aligned}
    \mathbf{O}_{t} &=\mathbf{H}_{t} \mathbf{W}^{(2)}+\mathbf{b}^{(2)} \\
    \mathbf{H}_{t} &=\sigma\left(\mathbf{X}_{t} \mathbf{W}_{X}^{(1)}+\mathbf{H}_{t-1} \mathbf{W}_{H}^{(1)}+\mathbf{b}^{(1)}\right)
    \end{aligned}
    $$

    * Backpropagation Through Time (BPTT)
    $$
    \partial_{w_{h}} h_{t}=\partial_{w_{h}} f\left(x_{t}, h_{t-1}, w_{h}\right)+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t} \partial_{h_{j-1}} f\left(x_{j}, h_{j-1}, w_{h}\right)\right) \partial_{w_{h}} f\left(x_{i}, h_{i-1}, w_{h}\right)
    $$

* 문제점
    * 시퀀스의 길이가 길어지는 경우 BPTT를 통해 역전파 알고리즘을 계산하면 gradient vanishing 등의 문제가 발생할 수 있습니다.
    * 이러한 문제를 해결하기 위해 LSTM이나 GRU와 같은 모델들이 등장하게 됩니다.

# 과제
## 선택과제 2
* RNN의 역전파 알고리즘을 이해할 수 있는 문제였다. RNN의 경우 이전 출력이 다음 출력에 영향을 주기 떄문에 gradient를 $W_{rec}$에 대해서도 고려해야 했다.
* 수식을 유도하여 최종적으로 다음과 같이 구현하였다.
```
def backward_gradient(X, S, grad_out, wRec):
    """
    X: input
    S: 모든 input 시퀀스에 대한 상태를 담고 있는 행렬
    grad_out: output의 gradient
    wRec: 재귀적으로 사용되는 학습 파라미터
    """
    # grad_over_time: loss의 state 에 대한 gradient 
    # 초기화
    grad_over_time = np.zeros((X.shape[0], X.shape[1]+1))
    grad_over_time[:, -1] = grad_out
    # gradient accumulations 초기화
    wx_grad = 0
    wRec_grad = 0
    '''
    TODO
    '''
    for k in range(-1, -X.shape[1]-1, -1):
        wx_grad = wx_grad + np.sum(grad_over_time[:,k] @ X[:,k])/X.shape[0]
        wRec_grad = wRec_grad + np.sum(grad_over_time[:,k] @ S[:,k-1])/S.shape[0]
        grad_over_time[:,k-1] = grad_over_time[:,k] * wRec

    return (wx_grad, wRec_grad), grad_over_time
```
* 최종 테스트 코드에서 아무 에러도 나지 않고 잘 동작함을 확인하였다.

# [피어세션](https://hackmd.io/@ai17/HyHlrP5kK)

# 후기
선택과제2를 하면서 삽질을 오래하여 좀 머리가 아픈 하루였다. 아래  TODO 1에 있는 수식으로만 하다가 먼다 잘못 했음을 느겼을때는 너무 시가닝 오래 지난 후였다.

어느세 부캠을 시작한지 1주일이 지났는데 새로운 열정을 받은 느낌이었다. 혼자 집에서 할 떄는 날이 갈수록 지치고 서서히 게으름에 짓눌려 노력도 열정도 풍화되는 느낌이었는데 단기적인 목적과 장기적인 목적을 가지고 부캠을 시작하고 다른 사람들과 같은 목적을 향해 공부하는 것은 다른 느낌이 들었다. 