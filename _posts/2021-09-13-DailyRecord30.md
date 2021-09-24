---
title: "[boostcamp AI Tech] 학습기록 day30 (week7)"
date: 2021-09-13 20:00:00 -0400
categories:
use_math: true
---

# CNN Visualization
## CNN visualization?
* CNN을 시각화 하는 방법들을 말합니다.
* 내부의 파라미터 등이 어떻게 동작하는지 직관적으로 이해하기 위해 시각화합니다.

## Simple visualization
### Filter visalization
* 기본적인 filter와 filter의 출력을 시각화
* 하지만 깊은 층의 필터는 고차원의 filter가 학습되므로 직접적으로 시각화 하는 것은 큰 의미가 없다.

## 목차
1. Parameter examination
2. Feature analysis
3. Sensitivity analysis
4. Decomposition
* 위의 방법일수록 model에 초점
* 아래 방법들은 data에 초점

## 1. Parameter examination
### 1) Embedding feature analysis1 (Nearest Neighbors (NN) in feature space)
* feature space의 입력 데이터와 가장 가까운 거리의 data들을 모아서 확인
* 의미론적으로 유사한 개념의 이미지들이 clustering

#### CNN에서 NN
* 학습된 convnet의 conv layer들만 준비
* DB의 모든 특징점을 추출
* feature space에 분포
* test 영상의 가까운 feature들의 영상을 검색

## 2. Feature analysis
### 1) Embedding feature analysis2 (Dimensionality reduction, t-SNE)
* 이해하기 어려운 고차원을 이해하기 쉬운 저차원 분포로 차원 축소

### 2) Activation investigation1 (layer activation)
* layer의 activation function을 분석
* 특정 layer의 특정 channel의 activation function을 thresholding 후 masking
* hidden node들의 역할을 확인 할 수 있다.

### 3) Activation investigation2 (maximally activating patches)
* layer activation을 분석하는 방법중 하나로 patches 사용
* hidden node에서 가장 큰 값을 가지는 부분을 patch로 생성
* patch들의 공통된 특징으로 hidden node들을 분석하는데 사용
* 중간 layer의 과정을 보는데 적합

### 4) Activation investigation3 (class visulization)
* model이 어떤 영상들을 상상하고 class를 결정하는지 볼 수 있다.
* 목적함수를 최적화 하여 생성
$$
I^{*}=\underset{I}{\arg \max } f(I)-\underset{\text { Regularization term }}{\operatorname{Reg}(I)}
$$
$$
I^{*}=\underset{I}{\arg \max } f(I)-\underset{\text { Regularization term }}{\lambda\|I\|_{2}^{2}}
$$
* gradient ascent를 사용하여 최대화
* 음수의 Regularization term를 사용하여 최대화 하면 할수록 0에 가깝게 최적화
* 과정
    * 임의의 랜덤 dummy image로 prediction score 계산
    * 입력 이미지의 class score를 최대화 하면서 backpropagation
    * 현재 이미지 업데이트
    * 위의 과정을 반복


## 3. Sensitivity analysis
* 모델이 특정 입력을 어디를 보고 있는가
### 1) Saliency test (Oclusion map)
* 여러 위치에 Oclusion을 넣었을 때 prediction score를 평가
* Oclusion patch에 따라 predictio scoer map을 생성
* model이 어디를 보고 prediction하는지 볼 수 있다.

### 2) Saliency test (via Backpropagation)
* 특정 이미지의 gradient ascent를 사용
* 과정
    * 입력영상의 inference (class score)
    * backpropagation을 하고 최종적으로 나온 gradient를 절대값이나 제곱을 이용하여 magnitude map 생성
    
### 3) Backpropagation-based saliency (guided backpropagation with Rectified unit)
* backward를 할때도 긍정적으로 작용하는 양수값만 남기고 음수는 ReLU로 막아준다.


### 4) Class activation mapping (CAM)
* heatmap의 형태로 어디를 참고하였는지 시각화
* 출력 구조를 변환
    * conv layer를 지나 나온 featur map 생성
    * Global Average pooling (GAP)
    * fc layer 하나 통과
* 변환된 구조로 재학습 (CAM architectur 학습)
$$
\begin{aligned}
&S_{c}=\sum_{k} w_{k}^{c} F_{k} \\
&\stackrel{G A P}{=} \sum_{k} w_{k}^{c} \sum_{(x, y)} f_{k}(x, y) \\
&=\sum_{(x, y)} \sum_{k} w_{k}^{c} f_{k}(x, y)
\end{aligned}
$$
* $\sum_{k} w_{k}^{c} f_{k}(x, y)$ (CAM)을 시각화
* ResNet이나 GoogLeNet은 바로 적용할 수 있다는 장점이 있다.

* 제약사항
    * CAM 구조를 적용 가능해야 한다
    * 재학습하므로 성능이 변할 수 있다.

### 5) Class activation mapping (Grad-CAM)
* CAM의 제약 사항을 없애서 구조나 재학습을 안하고 결과를 얻도록 제안
* backbone이 CNN이기만 하면 사용가능 (일반화된 tool)

* $\sum_{k} w_{k}^{c} f_{k}(x, y)$의 $w_{k}^{c}$를 어떻게 구할 것인가?
    * conv layer들만 backprop
    $$\alpha_{k}^{c}=\frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^{c}}{\partial A_{i j}^{k}}$$
    * backprop한 gradient들을 Global average pooling하여 $w_{k}^{c}$대신 사용
    $$
    L_{G r a d-C A M}^{c}=\operatorname{Re} L U\left(\sum_{k} \alpha_{k}^{c} A^{k}\right)
    $$
    * conv layer를 통과한 feature map $A$에 각 채널별로 곱하고 더한다음 ReLU를 통과하여 Grad-CAM을 생성

### 5) Class activation mapping (Guided Grad-CAM)
* Grad-CAM과 Guided Backprop을 결합
* rough했던 guided backprop에서 특정 class에 관련된 texture map만 추출

## 4. Decomposition
### 1) DeepLIFT
### 2) LRP


# 과제

# 피어세션
[ 2021년 9월13일  월요일 회의록 ]

✅ 오늘의 피어세션 (모더레이터: 조준희)

1. 강의 요약
    - 발표자: 임성민
    - 내용: Object Detection (SSD부터 나머지)
    - 발표자: 조준희
    - 내용: CNN Visualization

📢 토의 내용

- 논문리뷰 도전!
    - 다음주 목,금
    - 논문:  멘토님 추천받아서 (난이도는 비슷하게 부탁)
    - 형식: 자유
    - 곽지윤: U-net
    - 백종원: YOLOv4
    - 김현수: R-CNN
    - 이양재: VGGNet
    - 임성민: AlexNet
    - 조준희: Grad-CAM or other segmentation method

📢 내일 각자 해올 것

1. 모더레이터:  곽지윤 캠퍼님
2. 필수 과제 리뷰, 질문