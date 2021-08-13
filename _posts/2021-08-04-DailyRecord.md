---
title: "[boostcamp AI Tech] 학습기록 day03"
date: 2021-08-04 22:58:18 -0400
categories:
---

# AI Math
## 3. 미분
* 미분이란?
    * 미분 (differentiation)은 변수에 따른 함수값의 변화량입니다.
    $$
    f^{\prime}(x)=\lim _{h \rightarrow 0} \frac{f(x+h)-f(x)}{h}
    $$
    * 수학적으로 미분은 함수 $f$에 주어진 한 점 $(x, f(x))$에서의 기울기를 의미합니다.

* 편미분이란?
    * 변수가 여러개일 경우 편미분(partial$differentiation)을 이용하여 기울기를 구합니다.  
    $$
    \partial_{x_{i}} f(\mathbf{x})=\lim _{h \rightarrow 0} \frac{f\left(\mathbf{x}+h \mathbf{e}_{i}\right)-f(\mathbf{x})}{h}
    $$
    * 편미분을 이용하면 각 변수의 그레디언트 벡터를 이용하여 경사하강법 등에 적용이 가능합니다.


## 4. 경사하강법
* 경사하강법이란?
    * 경사하강법 (gradient descent)은 기울기를 이용하여 함수의 최적의 값을 반복적(iterative)으로 찾아내는 방법입니다.

    * 앞의 미분을 이용하면 함수에서 한접에서 기울기를 알면 어느 방향으로 $x$를 움직여야 함수값이 증가/감소하는지 알 수 있습니다. 이 것을 이용하여 함수의 극소값이나 극대값의 위치를 찾을 수 있습니다.

    * 알고리즘을 수도코드로 보면 다음과 같습니다.
    ```
    # init: 시작점, lr: 학습률, epsilon: 아주 작은 값
    x = init
    grad = gradient(x)
    while norm(grad) > epsilon:
        x = x - lr * grad
        grad = gradient(x)
    ``` 

* 선형회귀
    * 지난번에 배웠던 유사역행렬로 선형모델을 이용한 선형회귀식을 찾을 수 있습니다.
    $$
    \begin{aligned}
    \left[\begin{array}{c}
    -\mathbf{x}_{1}- \\
    -\mathbf{x}_{2}- \\
    \vdots \\
    -\mathbf{x}_{n}-
    \end{array}\right]\left[\begin{array}{c}
    \beta_{1} \\
    \beta_{2} \\
    \vdots \\
    \beta_{m}
    \end{array}\right] \neq\left[\begin{array}{c}
    y_{1} \\
    y_{2} \\
    \vdots \\
    y_{m}
    \end{array}\right],   
    \mathbf{X} \beta=\hat{\mathbf{y}} \approx \mathbf{y}
    \end{aligned}
    $$
    * 이 선형회귀를 경사하강법 (gradient descent)으로 구할 수 있다.
    * 선형회귀의 목적식: $\|\mathbf{y}-\mathbf{X} \beta\|_{2}$
    * 목적식을 최소화하는 $\beta$를 찾기 위해 $\beta$에 대한 그레디언트를 구한다.
    $$
    \nabla_{\beta}\|\mathbf{y}-\mathbf{X} \beta\|_{2} = 
    -\frac{\mathbf{X}^{\top}(\mathbf{y}-\mathbf{X} \beta)}{n\|\mathbf{y}-\mathbf{X} \beta\|_{2}}
    $$
    * 그레디언트를 구했으면 목적식을 최소화하도록 경사하강법을 이용하여 $\beta$를 업데이트 한다. 여기서 $\lambda$는 학습률(learning rate)를 의미한다.
    $$
    \beta^{(t+1)} =\beta^{(t)}-\lambda \nabla_{\beta}\left\|\mathbf{y}-\mathbf{X} \beta^{(t)}\right\|
    $$
    $$
    \beta^{(t+1)} = \beta^{(t)}+\frac{\lambda}{n} \frac{\mathbf{X}^{\top}\left(\mathbf{y}-\mathbf{X} \beta^{(t)}\right)}{\left\|\mathbf{y}-\mathbf{X} \beta^{(t)}\right\|}
    $$
    * 실제 구현에서는 연산의 문제로 목적식의 제곱을 이용하여 그레디언트를 구해 최소화 한다. 
    $$
    \begin{aligned}
    \nabla_{\beta}\|\mathbf{y}-\mathbf{X} \beta\|_{2}^{2}     
    &=-\frac{2}{n} \mathbf{X}^{\top}(\mathbf{y}-\mathbf{X} \beta)
    \end{aligned}
    $$
    $$
    \begin{aligned}
    \beta^{(t+1)} &= \beta^{(t)}+\frac{2 \lambda}{n} \mathbf{X}^{\top}\left(\mathbf{y}-\mathbf{X} \beta^{(t)}\right)
    \end{aligned}
    $$
    * 이를 수도코드로 나타내면 다음과 같다.
    ```
    # lr: 학습률, T: 학습횟수
    for t in range(T):
        error = y - X @ beta
        grad = -X.T @ error
        beta = beta - lr * grad
    ```

* 경사하강법의 문제
    * 경사하강법은 미분가능하고 볼록한 함수에 대해서 학습률을 적절히 선택하고 학습 횟수를 충분히 하면 수렴이 보장된니다. ex) 선형회귀문제

    * 하지만 비선형회귀의 경우 볼록하지 않을 수 있으므로 수렴이 보장되지 않는다는 점 유의해야 합니다.

* 확률적 경사하강법
    * 확률적 경사하강법 (Stochastic Gradient Descent, SGD)는 모든 데이터를 사용해서 업데이트 하는 것이 아니라 일부의 데이터만을 이용하여 업데이트 하는 방법을 말합니다.
    $$
    \beta^{(t+1)} = \beta^{(t)}+\frac{2 \lambda}{n} \mathbf{X}^{\top}\left(\mathbf{y}-\mathbf{X} \beta^{(t)}\right) \stackrel{O\left(d^{2} n\right) \rightarrow O\left(d^{2} b\right)}{\longrightarrow} 
    \beta^{(t+1)} = \beta^{(t)}+\frac{2 \lambda}{b} \mathbf{X}_{(b)}^{\top}\left(\mathbf{y}_{(b)}-\mathbf{X}_{(b)} \beta^{(t)}\right)
    $$
    * 위에서 볼록이 아닌 경우 경사하강법은 수렴을 보장하지 않는 문제가 있었는데 SGD는 그러한 비선형 문제에서 실증적으로 더 좋은 성능을 보이고 있는게 검증되었다.
    * SGD는 또한 일부의 데이터만 활용하므로 연산자원을 효율적으로 그리고 빠르게 결과를 얻을 수 있다는 장점이 있다.
    * 흔히 SGD에 사용하는 일부 대이터를 미니배치 (mini batch)라고 부른다.
    * 경사하강법은 확률적으로 데이터를 선택하기 때문에 목적식의 모양이 전체 데이터 (full batch)를 사용할 떄와 모양이 계속 변하며 학습하게 됩니다.

## 5. 딥러닝 기초
* 딥러닝에서는 앞에서 배웠던 선형모델이 아니라 비선형모델인 신경망(neural network)를 이용하게 됩니다.
* 신경망이란?
    * 신경망은 선형모델과 활성함수가 합성된 함수들이 여러 층을 이루며 합성된 함수입니다.
    * 신경망에서 층이 여러개인 이유는 층이 깊을 수록 목적함수를 근사하는데 필요한 노드의 숫자가 줄어들어 효율적으로 학습할 수 있습니다.


* 활성화 함수
    * 활성화 함수 (activation function)는 신경망에서 계산된 결과를 입력에 따라 0/1로 활성화 할지 결정하는 함수입니다.
    * 비선형 함수로 앞에서 배운 선형 모델에 활성화 함수가 결합하면서 비선형모델로써 기능합니다.
    * 시그모이드(sigmoid) 함수나 tanh 함수를 전통적으로 많이 썼지만 현재 딥러닝에서는 ReLU가 가장 많이 사용된다.
    $$
    \sigma(x)=\frac{1}{1+e^{-x}}, \quad \tanh (x)=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}, \quad \operatorname{ReLU}(x)=\max \{0, x\}
    $$

* 소프트맥스
    * 소프트맥스 (softmax) 함수는 출력을 확률로 해석 할 수 있도록 해줍니다.
    $$
    \operatorname{softmax}(\mathbf{o})=\left(\frac{\exp \left(o_{1}\right)}{\sum_{k=1}^{p} \exp \left(o_{k}\right)}, \ldots, \frac{\exp \left(o_{p}\right)}{\sum_{k=1}^{p} \exp \left(o_{k}\right)}\right)
    $$
    * 분류문제를 풀 때 선형보델에 소프트 맥스 함수를 결합하여 학습을 합니다.

* 역전파 
    * 역전파 (backpropagation) 알고리즘을 사용하면 각층에 사용된 패러미터를 학습 할 수 있습니다.
    * 역전파 알고리즘은 미분의 연쇄법칙 (chain-rule)을 기반으로 각 층의 그레디언트를 구할 수 있습니다.
    * 이렇게 계산된 그레디언트로 각 층의 파라미터를 경사하강법을 이용하여 학습합니다.


# 과제
## 과제 4
코딩할 떄 많이 하는 야구게임을 구현하는 문제다. 기본적인 룰에 필요한 함수들을 먼저 짜고 메인함수에서 야구 게임 룰을 구현하는 것이었다. 메인은 크게 본게임(입력 종료여부, 게임점수판별, 결과디스플레이), 게임이 끝난 후 게임 재시작 여부 판별하는 부분으로 나누어 작성하였다. 기능에 대한 함수를 나누어서 틀을 주셨기 때문에 빨리 끝낼 수 있었다.
## 과제 5
모스부호를 인코딩/디코딩하는 코드를 짜는 과제였다. 입력만 제어해야 했던 야구게임과 달리 입력을 모스부호 테이블에 따라 출력을 결정하는 부분이 야구게임과 큰 차이였다. 과제4도 그랬지만 문자열을 제어하는게 중요에러의 주요인이었다.

# [피어세션](https://hackmd.io/@ai17/HkgUDav1t)

# 후기
오늘은 딥러닝에서 필요한 기본적인 개념들을 다시 복습하고 과제4, 5를 구현해보았다. 예전에도 공부했지만 이렇게 정리하니까 더 좋은 것 같았다. 과제는 기본적으로 문자열을 제어하는게 가장 문제였던 느낌이었는데 다른 질문 게시판등을 보니까 이런 문제를 정규표현식을 이용하여 해결하는 분들도 있었다. 정규표현식은 제대로 공부해보지 못했는데 중간에 짬이 나면 공부가 좀 필요할 것 같았다.