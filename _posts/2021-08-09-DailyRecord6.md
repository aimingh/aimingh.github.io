---
title: "[boostcamp AI Tech] 학습기록 day06 (week2)"
date: 2021-08-09 20:54:05 -0400
categories:
use_math: true
---

# Deep Learning Basic
## 1. Historical Review
* 딥러닝
    * AI(ML(DL))
    * key components
        1. Data
            * 학습할 데이터
            * 학습을 하기 위해 기반이 될 데이터가 필요합니다.
            * 이러한 데이터는 우리가 풀고자 하는 문제에 의존적입니다.
                * Classification: 분류할 이미지들
                * Sementic segmentation: 이미지와 각 영영에 대한 정답
                * Detection: 이미지와 정답에 대한 위치와 카테고리
                * Pose estimation: 이미지와 골격에 대한 정보
                * Visual QnA: 이미지와 문장

        2. Model
            * 데이터를 학습할 모델
            * 데이터가 주어졌을 때 정답으로 바꿔주는 것이 모델입니다
                * ex) AlexNet, GoogLeNet, LSTM, Deep AitoEncoders, GAN
            
        3. Loss
            * 모델 학습에 필요한 손실함수 (loss function)
            * 학습에서 우리가 얼마나 학습이 되었는지에 대한 지표입니다. 
            <!-- 수식 추가 -->
                * Regression Task: Mean Square Error (MSE)
                * Classification Task: Cross Entropy (CE)
                * Probabilistic Task: Maximum Likelifood Estimation (MLE)

        4. Optimization Algorithm
            * 손실을 최소화하면 모델의 파라미터를 최적화할 알고리즘
            * 데이터와 모델, 손실함수가 정해졌을 때 모델의 파라미터를 최적화하는 알고리즘입니다.
            * 일반적으로 first order method로 1차 미분한 정보를 활용합니다.
            * SGD를 기본으로 Adaptive SGD, Momentum, Adam 등을 사용합니다.
            * 정규화
                * 정규화 (Regularization)는 학습이 잘 안되게 하는 역할을 합니다.
                * 단순하게 손실함수를 줄이는게 하니라 학습되지 않은 데이터에 대해서도 강건하게 하기 위해 사용됩니다.
                <!-- * ex) Dropout, ~~~ -->

* [Historical Review](https://dennybritz.com/blog/deep-learning-most-important-ideas)
    * 2012 AlexNet
    * 2013 DQN
    * 2014 - Encoder/Decoder, Adam
    * 2015 - GAN, ResNet
    * 2017 - Transformer
    * 2018 - Bert
    * 2019 - Big Language Models (GPT-X)
    * 2020 - Self-Supervised Learning


## 2. 뉴럴네트워크 - MLP (Multi-Layer Perceptron)
* Neural Networks
    * 생물학적 신경망에서 영감을 얻어서 만들어진 컴퓨팅 시스템입니다.
    * 비선형 변환을 따르기 위해 여러개의 어파인 변환을 쌓아 만든 비선형 연산이라고도 정의할 수 있습니다.

* Linear Neural Networks
    * 가장 간단한 신경망으로 볼 수 있습니다.
    $$
    \begin{aligned}
    &\text { Data: } \mathcal{D}=\left\{\left(x_{i}, y_{i}\right)\right\}_{i=1}^{N} \\
    &\text { Model: } \hat{y}=w x+b \\
    &\text { Loss: } \operatorname{loss}=\frac{1}{N} \sum_{i=1}^{N}\left(y_{i}-\hat{y}_{i}\right)^{2}
    \end{aligned}
    $$
    * 손실함수를 정의하고 손실함수를 작게 만들어야 합니다. 
    $$
    \begin{aligned}
    \frac{\partial \operatorname{loss}}{\partial w} &=\frac{\partial}{\partial w} \frac{1}{N} \sum_{i=1}^{N}\left(y_{i}-\hat{y}_{i}\right)^{2} \\
    &=\frac{\partial}{\partial w} \frac{1}{N} \sum_{i=1}^{N}\left(y_{i}-w x_{i}-b\right)^{2} \\
    &=-\frac{1}{N} \sum_{i=1}^{N}-2\left(y_{i}-w x_{i}-b\right) x_{i}
    \end{aligned}
    $$
    $$
    \begin{aligned}
    \frac{\partial \operatorname{loss}}{\partial b} &=\frac{\partial}{\partial b} \frac{1}{N} \sum_{i=1}^{N}\left(y_{i}-\hat{y}_{i}\right)^{2} \\
    &=\frac{\partial}{\partial b} \frac{1}{N} \sum_{i=1}^{N}\left(y_{i}-w x_{i}-b\right)^{2} \\
    &=-\frac{1}{N} \sum_{i=1}^{N}-2\left(y_{i}-w x_{i}-b\right)
    \end{aligned}
    $$
    * 그러므로 기울기를 구해서 작아지는 방향을 찾아 파라미터를 업데이트 하면 loss를 감소 시킬 수 있습니다.
    $$
    \begin{aligned}
    &w \leftarrow w-\eta \frac{\partial \operatorname{loss}}{\partial w} \\
    &b \leftarrow b-\eta \frac{\partial l o s s}{\partial b}
    \end{aligned}
    $$
    * $\eta$가 중요한 요소로 너무 작거나 크면 학습이 잘 되지 않습니다.
    
    * 위의 과정은 행렬을 이용하여 더 많은 차원의 입력과 출력을 다룰 수 있습니다.
    $$
    \mathbf{y}=W^{T} \mathbf{x}+\mathbf{b}
    $$

    * 이러한 linear neural network를 여러단 쌓는다면 다음과 같이 표현 항 수 있는데 이러함 모델은 단순히 선형식을 반복하므로 하나의 선형식이라고 볼 수 있습니다. 
    $$
    y=W_{2}^{T} \mathbf{h}=W_{2}^{T} W_{1}^{T} \mathbf{x}
    $$
    * 그래서 딥러닝으로의 확장은 중간에 활성화 함수같은 비선형 함수를 이용하여 표현의 범위를 확징시켜 사용합니다. 
    $$
    y=W_{2}^{T} \mathbf{h}=W_{2}^{T} \rho W_{1}^{T} \mathbf{x}
    $$

    * 히든레이어가 하나 있는 뉴럴네트워크는 우리가 일반적으로 생각할 수 있는 대부분의 연속적인 함수를 근사할 수 있다.  
        * 다만 존재성만을 보장하고 내가 학습하는 함수가 그런 함수를 얻을 것 이라는 보장은 없다.


* Multi-Layer Perceptron
    * 위에서 보았던 식을 확장한 것이 Multi-Layer Perceptron 구조입니다.
    $$
    y=W_{3}^{T} \mathbf{h}_{2}=W_{3}^{T} \rho\left(W_{2}^{T} \mathbf{h}_{1}\right)=W_{3}^{T} \rho\left(W_{2}^{T} \rho\left(W_{1}^{T} \mathbf{x}\right)\right)
    $$

    * Loss function
        * Regression    
            * 바운더리의 문제로 무조건 MSE가 잘 학습되는 것은 아닙니다. 상황에 맞게 특성을 보며 절대값을 loss function 등 방법을 고려해야 합니다.
        $$\mathrm{MSE}=\frac{1}{N} \sum_{i=1}^{N} \sum_{d=1}^{D}\left(y_{i}^{(d)}-\hat{y}_{i}^{(d)}\right)^{2}$$
        * Classification  
            * 신경망 출력에서 해당 정답에 대한 출력만 높이는 방향으로 학습하기 때문에 classification 문제에 적합한 loss fucntion입니다.
        $$\mathrm{CE}=-\frac{1}{N} \sum_{i=1}^{N} \sum_{d=1}^{D} y_{i}^{(d)} \log \hat{y}_{i}^{(d)}$$
        * Probabilistic Task
            * 확률에 대한 정보를 활용할 떄 사용하는 loss function입니다.
        $$
        \mathrm{MLE}=\frac{1}{N} \sum_{i=1}^{N} \sum_{d=1}^{D} \log \mathcal{N}\left(y_{i}^{(d)} ; \hat{y}_{i}^{(d)}, 1\right) \quad(=\mathrm{MSE})
        $$


# 과제
## 필수과제1 Multilayer Perceptron (MLP) 
* pytorch를 이용해서 MLP를 구현하는데 중간 중간 비워져 있는 함수를 채우는 숙제였다.
```
class MultiLayerPerceptronClass(nn.Module):
    """
        Multilayer Perceptron (MLP) Class
    """
    def __init__(self,name='mlp',xdim=784,hdim=256,ydim=10):
        super(MultiLayerPerceptronClass,self).__init__()
        self.name = name
        self.xdim = xdim
        self.hdim = hdim
        self.ydim = ydim
        self.lin_1 = nn.Linear(self.xdim, self.hdim) # FILL IN HERE
        self.lin_2 = nn.Linear(self.hdim, self.ydim) # FILL IN HERE
        self.init_param() # initialize parameters
        
    def init_param(self):
        nn.init.kaiming_normal_(self.lin_1.weight)
        nn.init.zeros_(self.lin_1.bias)
        nn.init.kaiming_normal_(self.lin_2.weight)
        nn.init.zeros_(self.lin_2.bias)

    def forward(self,x):
        net = x
        net = self.lin_1(net)
        net = F.relu(net)
        net = self.lin_2(net)
        return net
```
* 처음에는 모델을 초기화 하는 부분이었는데 기본적인 linear layer 2개를 쌓아 사용하는 것을 볼 수 있었다. 안쪽의 입력과 출력이 맞게 작성하였다.

```
def func_eval(model,data_iter,device):
    with torch.no_grad():
        model.eval() # evaluate (affects DropOut and BN)
        n_total,n_correct = 0,0
        for batch_in,batch_out in data_iter:
            y_trgt = batch_out.to(device)
            model_pred = model(batch_in.view(-1,28*28).to(device)) # FILL IN HERE
            _, y_pred= torch.max(model_pred.data,1)
            n_correct += (y_trgt==y_pred).sum().item() # FILL IN HERE
            n_total += batch_in.size(0)
        val_accr = (n_correct/n_total)
        model.train() # back to train mode 
    return val_accr
print ("Done")
```
* 여기서는 만든 모델을 이용하여 evaluation function을 구성하는 부분이었고 들어온 데이터를 batch size만큼 입력되도록 view (numpy의 reshape하고 같은 기능) 함수를 이용하여 재배열하여 입력하였고 평가 하기 위하여 y_trgt==y_pred로 일치하는 출력의 수를 전체 데이터의 수로 나누어 validation accuracy를 측정하도록 하였다.

```
print ("Start training.")
M.init_param() # initialize parameters
M.train()
EPOCHS,print_every = 10,1
for epoch in range(EPOCHS):
    loss_val_sum = 0
    for batch_in,batch_out in train_iter:
        # Forward path
        y_pred = M.forward(batch_in.view(-1, 28*28).to(device))
        loss_out = loss(y_pred,batch_out.to(device))
        # Update
        optm.zero_grad()    # FILL IN HERE      # reset gradient 
        loss_out.backward() # FILL IN HERE      # backpropagate
        optm.step()         # FILL IN HERE      # optimizer update
        loss_val_sum += loss_out
    loss_val_avg = loss_val_sum/len(train_iter)
    # Print
    if ((epoch%print_every)==0) or (epoch==(EPOCHS-1)):
        train_accr = func_eval(M,train_iter,device)
        test_accr = func_eval(M,test_iter,device)
        print ("epoch:[%d] loss:[%.3f] train_accr:[%.3f] test_accr:[%.3f]."%
               (epoch,loss_val_avg,train_accr,test_accr))
```
* 마지막 부분은 학습 부분에서 update하는 부분을 채워 넣는 것이 었고 gradient를 0으로 초기화 하고 역전파, optimizer를 update하는 순으로 구성하였다.

# [피어세션](https://hackmd.io/@ai17/rJn_cLA1K)

# 후기
파이토치를 다음주에 하는데 실습 과제로 파이토치를 이용하는 걸 보고 당황했지만 일단 일부 내용만 채우는 형식으로 현재 딥러닝 기본 과정의 감을 잡으라는 의미로 받아들였습니다. 전체적인 흐름과 구현이 어떻게 이루어지는지를 중점적으로 보고 파이토치에 대한 문법은 감만 잡는 선으로 이번주는 해야 될 것 같다고 느꼈습니다.
