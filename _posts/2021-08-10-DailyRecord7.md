---
title: "[boostcamp AI Tech] 학습기록 day07 (week2)"
date: 2021-08-10 21:23:25 -0400
categories:
use_math: true
---

# Deep Learning Basic
## 3.Optimization
경사하강법 (Gradient Descent)은 local minimum을 찾기 위한 일차 미분을 이용한 반복적인 최적화 알고리즘입니다. 
딥러닝에서 최적화는 경사하강법을 이용하여 local minima를 찾습니다.

* Important Concepts in Optimization
    * Generalization
        * 일반화 (Generalization)은 학습된 모델이 학습에 사용하지 않은 데이터에 대해 잘동작하게 만드는 것입니다.
        * 일반적으로 학습을 하면 training error가 줄어듭니다.
        * 하지만 training error가 줄어든다고 test error가 줄어드는 것이 아닙니다.
        * 이러한 차이를 generalization gap이라고도 합니다.

    * Under-fitting vs. over-fitting
        * over-fitting: 학습데이터를 잘 맞추지만 테스트 데이터는 정확도가 낮아서 generalization이 잘 안된것을 말합니다.
        * under=fitting: 네트워크가 너무 작고 간단하거나 학습이 부족하여 학습데이터도 잘 못맞추는 경우를 말합니다.

    * Cross-validation
        * 테스트 데이터에 대해 강건한게 (robust) 학습하기 위해서 주어진 데이터를 나눕니다.
        * cross-validation은 독립된 테스트 데이터셋에 모델을 일반화을 평가하기 위래 사용하는 검증 기법입니다.
        * 학습데이터가 적을 때 k개의 fold로 나누고 각 fold에 대해 validation fold를 돌아가면서 학습을 합니다.
        * 데이터를 다 사용하기 위해서는 cross-validation을 사용하여 하이퍼 파라미터를 찾아내고 찾아낸 데이터를 이용하여 모든 fold를 학습에 사용합니다.
        * 절대 테스트 데이터셋을 학습에 사용해서는 안됩니다.

    * Bias-variance tradeoff
        * Bias: 평균적으로 봤을 때 출력이 정답에 접근하는가를 말합니다.
        * Variance: 출력이 얼마나 일관적으로 나오는가를 말합니다. (over-fitting과도 관련된다고 볼 수 있습니다.)
        * 학습데이터에 noise가 있다면 학습하는 데이터에 대해 minimize를 할 수 있습니다. 이 경우 3가지 부분으로 나눌 수 있습니다.
        $$
        \text { Given } \mathcal{D}=\left\{\left(x_{i}, t_{i}\right)\right\}_{i=1}^{N}, \text { where } t=f(x)+\epsilon \text { and } \epsilon \sim \mathcal{N}\left(0, \sigma^{2}\right)
        $$
        $$
        \begin{aligned}
        \mathbb{E}\left[(t-\hat{f})^{2}\right] &=\mathbb{E}\left[(t-f+f-\hat{f})^{2}\right] \\
        &=\mathbb{E}\left[\left(f-\mathbb{E}[\hat{f}]^{2}\right)^{2}\right]+\mathbb{E}\left[(\mathbb{E}[\hat{f}]-\hat{f})^{2}\right]+\mathbb{E}[\epsilon]
        \end{aligned}
        $$
        * 그러므로 하나가 작아지면 다른 하나가 커질 수 밖에 없는 trade off가 일어납니다.

    * Bootstrapping
        * random sampling을 적용하여 가설 검증 (test)을 하거나 매트릭 (metric)을 검증하는 것을 말합니다.
        * 랜덤한 데이터 샘플링을 반복하여 여러 학습 데이터를 만들고 그것으로 여러 모델을 만들어 사용합니다.

    * Bagging and Boosting
        * Bagging (Bootstrapping aggregating)
            * 여러개의 모델을 램덤 샘플링을 이용하여 만들고 평균으로 결과를 얻습니다. 일반적으로 이러한 기법을 앙상블이라고 부르기도 합니다. 
            * 한개의 모델을 사용하는 것보다 여러개의 모델을 만들어 사용하는 것이 성능이 일반적으로 좋습니다.
        * Boosting
            * 특정한 sample에 대해서 집중하는 분류기를 만들고 이러한 분류기들을 여러개 연속으로 (sequential) 연결하여 하나의 분류기를 만드는 방식입니다.

* Practical Gradient Descent Methods
    * Gradient Descent Methods according to batch-size
        * Stochastic gradient descent
            * 하나의 샘플을 계산하여 그래디언트를 구하여 업데이트 하는 방식입니다.

        * Mini-batch gradient descent
            * 일부 샘플들을 계산하여 그래디언트를 구하여 업데이트 하는 방식입니다.
            * 가장 많이 활용됩니다.

        * Batch gradient descent
            * 모든 샘플을 계산하여 그래디언트를 구하여 업데이트 하는 방식입니다.
    * Batch-size Matters
        * Keskar, N. S., Mudigere, D., Nocedal, J., Smelyanskiy, M., & Tang, P. T. P. (2016). On large-batch training for deep learning: Generalization gap and sharp minima. arXiv preprint arXiv:1609.04836
            * large batch size를 활용하게 되면 sharp minimizer에 도달하게 되고 small batch size를 활용하면 flat minimizer에 도달합니다.
            * sharp minimizer보다 flat minimizer에 도달하는 것이 좋다
            * training function과 test function을 비교하면 flat minimum에서 조금 차이나도 결과가 큰 차이가 나지 않지만 sharp minimum에서 큰 차이가 나면 결과가 큰 차이가 나서 generalization gap이 커질 수 있다.

    * Gradient Descent Methods according
        * (Stochastic) gradient descent
            * learning rate를 잡는 것이 어렵다는 문제가 있습니다.
        $$
        W_{t+1}= W_{t}-\eta g_{t}
        $$
        
        * Momentum
            * 이전 그레디언트를 관성적으로 일정 비율을 포함시켜주어 업데이트 합니다.
            * minima로 관성이 움직이기 때문에 좀더 빨리 학습된다.
        $$ a_{t+1}= \beta a_{t}+g_{t} $$
        $$ W_{t+1} = W_{t}-\eta a_{t+1} $$

        * Nesterov accelerated gradient
            * momentum과 달리 먼저 이동하고 이동한 자리의 그래디언트를 누적시킵니다.
            * mimentum은 작은 localminima의 경우 관성으로 빠져나갈 수 있는데 이 방법은 이동후 그래디언트를 계산하기 때문에 minima로 잘 찾아갈 수 있다.
        $$
        a_{t+1} \quad \beta a_{t}+\nabla \mathcal{L}\left(W_{t}-\eta \beta a_{t}\right)
        $$
        $$
        W_{t+1} \quad W_{t}-\eta a_{t+1}
        $$

        * Adagrad
            * 파라미터가 얼마나 업데이트 되었는지를 학습에 포함시킵니다.
            * 파라미터가 지금까지 많이 변했다면 더 적게 변화시키고 적게 변했다면 더 많이 변화시킵니다.
            * 그래디언트가 누적되어 포함되기 때문에 학습이 진행될 수록 학습이 되지 않는 문제가 있다.
        $$
        W_{t+1}=W_{t}-\frac{\eta}{\sqrt{G_{t}}+\epsilon}g_{t}
        $$

        * Adadelta
            * Adagrad가 가지는 그래디언트의 누적에 의해 학습률이 감소되는 것을 개선한 방법입니다.
            * 현재의 그래디언트를 적용할 때 지수 평균을 이용하고 누적 윈도우로 제약을 주어 학습률이 monotonically하게 감소하는 것을 예방합니다.
            * 특징으로는 학습률을 하이퍼파라미터로 지정하지 않는다는 것입니다.
        $$
        \begin{aligned}
        G_{t} &=\gamma G_{t-1}+(1-\gamma) g_{t}^{2} \\
        W_{t+1} &=W_{t}-\frac{\sqrt{H_{t-1}+\epsilon}}{\sqrt{G_{t}+\epsilon}} g_{t} \\
        \quad H_{t} &=\gamma H_{t-1}+(1-\gamma)\left(\Delta W_{t}\right)^{2}
        \end{aligned}
        $$

        * RMSprop
            * Adagrad에 지수 평균을 이용하여 그래디언트를 계산합니다.
        $$
        \begin{aligned}
        {G_{b}} &=\gamma G_{t-1}+(1-\gamma) g_{t}^{2} \\
        W_{t+1} &=W_{t}-\frac{\eta}{\sqrt{G_{t}+\epsilon}} g_{t}
        \end{aligned}
        $$

        * Adam
            * Adaptive Moment Estimation (Adam)으로 모멘텀과 RMSprop을 같이 적용했다고 볼 수 있습니다.
            * 이전 그래디언트 정보에 해당하는 모멘텀 두개를 합친 것입니다.
            * 하이퍼파라미터로 4개가 있고 값을 잘 조절하는 것이 중요합니다.
        $$
        \begin{aligned}
        &m_{t}=\beta_{1} m_{t=1}+\left(1-\beta_{1}\right) g_{t}\\
        &{v_{t}}=\beta_{2} v_{t-1}+\left(1-\beta_{2}\right) g_{t}^{2}\\
        &W_{t+1}=W_{t}-\frac{\eta}{\sqrt{v_{t}+\epsilon}} \frac{\sqrt{1-\beta_{2}^{t}}}{1-\beta_{1}^{t}} {m}_{t}
        \end{aligned}
        $$


* Regularization
    * Early stopping 
        * over-fitting이 일어나기 전에 validation dataset을 이용하여 validation error를 확인하여 generalization gap이 커지기 전에 학습을 멈추는 방법입니다.

    * Parameter norm penalty 
        * 네트워크의 weight를 작게하는 텀을 추가합니다.
        * function space에서 함수를 부드러운 함수로 만급니다. (부드러운 함수가 많으면 generalization gap이 작아진다고 가정)

    * Data augmentation
        * 데이터를 늘리기 위한 방법입니다
        * 정답이 바뀌지 않는 선에서 영상을 여러가지 방법으로 변화시켜줍니다.
            * 반전, 회전 등 

    * Noise robustness 
        * 입력 데이터나 weight들에 노이즈를 추가합니다

    * Label smoothing 
        * 2학습에서 2개의 데이터를 섞어주는 기법입니다
        * 분류문제에서 decision boundaty를 찾는 거이 목적인데 이 방법을 쓰면 boundary를 부드럽게 만들어주는 효과가 있습니다.
        * Mixup, Cutout, CutMix

    * Dropout 
        * 네트워크의 랜덤으로 일부 뉴런을 0으로 하고 학습합니다.
        * 각각의 뉴런들이 robust한 feature들을 학습 할 수 있습니다.

    * Batch normalization
        * 각 레이어의 통계값을 정규화 시키는 방법입니다.
        * 다른 normalization 방법들도 있습니다.
            * Layer Norm, Instance Norm, Group Norm
    $$
    \begin{aligned}
    \mu_{B} &=\frac{1}{m} \sum_{i=1}^{m} x_{i} \\
    \sigma_{B}^{2} &=\frac{1}{m} \sum_{i=1}^{m}\left(x_{i}-\mu_{B}\right)^{2} \\
    \hat{x}_{i} &=\frac{x_{i}-\mu_{B}}{\sqrt{\sigma_{B}^{2}+\epsilon}}
    \end{aligned}
    $$

# 과제
## 필수과제2 Regression with Different Optimizers
* 위에서 여러 optimizer에 대하여 배웠고 이러한 optimizer들이 구현되었을 때 어떤 차이를 가지는지 직접 실습으로 확인하였습니다.
```
class Model(nn.Module):
    def __init__(self,name='mlp',xdim=1,hdims=[16,16],ydim=1):
        super(Model, self).__init__()
        self.name = name
        self.xdim = xdim
        self.hdims = hdims
        self.ydim = ydim

        self.layers = []
        prev_hdim = self.xdim
        for hdim in self.hdims:
            self.layers.append(nn.Linear(prev_hdim, hdim, bias=True))  # FILL IN HERE
            self.layers.append(nn.Tanh())  # activation
            prev_hdim = hdim
        # Final layer (without activation)
        self.layers.append(nn.Linear(prev_hdim,self.ydim,bias=True))

        # Concatenate all layers 
        self.net = nn.Sequential()
        for l_idx,layer in enumerate(self.layers):
            layer_name = "%s_%02d"%(type(layer).__name__.lower(),l_idx)
            self.net.add_module(layer_name,layer)

        self.init_param() # initialize parameters
```
* 처음에는 모델을 정의하는 것부터 시작하였습니다. 이번에도 간단한 모델을 구성하는데 for문을 사용하였기의 앞뒤 맥락에 맞게 입력과 출력의 사이즈를 고려하여 채워넣었습니다.

```
LEARNING_RATE = 1e-2
# Instantiate models
model_sgd = Model(name='mlp_sgd',xdim=1,hdims=[64,64],ydim=1).to(device)
model_momentum = Model(name='mlp_momentum',xdim=1,hdims=[64,64],ydim=1).to(device)
model_adam = Model(name='mlp_adam',xdim=1,hdims=[64,64],ydim=1).to(device)
# Optimizers
loss = nn.MSELoss()
optm_sgd = optim.SGD(model_sgd.parameters(), lr=LEARNING_RATE) # FILL IN HERE
optm_momentum = optim.SGD(model_momentum.parameters(), lr=LEARNING_RATE, momentum=0.9) # FILL IN HERE
optm_adam = optim.Adam(model_adam.parameters(), lr=LEARNING_RATE)# FILL IN HERE
print ("Done.")
```
* 모델의 파라미터를 다른 종류의 optimizer에 넣어 구성하였습니다.

# [피어세션](https://hackmd.io/@ai17/HkGJAjkgF)

# 후기
optimizer의 수식을 다시 보며 의미와 장점 등을 살펴 볼 수 있어 좋았습니다.