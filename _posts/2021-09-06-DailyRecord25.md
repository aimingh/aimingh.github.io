---
title: "[boostcamp AI Tech] 학습기록 day25 (week6)"
date: 2021-09-06 20:00:00 -0400
categories:
use_math: true
---

# Image Classification
## Visual perception
### Overview
* AI (Artificial Intelligence): 지성적인 능력(사고하고 분석하는 능력) 외에도 시각이나 소리 등에 대한 지각 능력 또한 컴퓨터 시스템이 수행하게 하는 것
* 사람은 지각능력을 통해 인지능력이 발달
* 지각능력은 사고에 대한 input, output의 데이터에 관련한다.
* 시각의 중요성: 인간은 시각의 비중이 크고 뇌에서 많은 처리를 하고 있다.
* 시각 인식의 과정
    1. Visual World
    2. Sensing - 사람:눈, 컴퓨터:카메라
    3. Interpreting - 사람:뇌, 컴퓨터:GPU&Algorithm
    4. Interperetation - Representation, High-level description, 자료구조


### Computer vision and computer graphics
    |                                   |Input                 |Output                |
    |-----------------------------------|----------------------|----------------------|
    |Computer vision (Inverse rendering)|Image(visual)         |Symbol(representation)|
    |Computer graphics (Rendering)      |Symbol(representation)|Image(visual)         |

### Visual perception의 종류
    * Color perception
    * Motion percception - Motion detection
    * 3D perception
    * Sementic-level perception - Sementic segmentation
    * Social perception (emotion perception) - Emotion classification
    * Visuomotor perception

### 시각의 이해
* CV는 사람의 생물학적 특성을 이해하고 컴퓨터 알고리즘으로 적용하는 것 또한 포함된다.
    * 대처 환각
        * 눈이나 코, 입의 위치가 뒤집힌 상태에 불편함을 느낌
        * 사람의 인식 또한 어느정도 bias가 있다고 볼 수 있다.

## Machine learning and Deep learning
* Machine learning: Input -> Feature extraction -> Classification -> Output
* Deep learning: Input -> Feature extraction and Classification -> Output
* 전통적인 Machine learning에서 Feature extraction은 사람이 필요하다고 생각되는 feature들을 알고리즘을 고안해서 직접 추출한다.
* 하지만 사람의 주관이 개입된 Feature extraction은 학습에 정말 필요한 정보를 제공하는지 못하는지 알 수 없다.
* Deep learning은 Feature extraction을 컴퓨터에게 선입견 없이 필요한 부분을 학습하게 한다.

## Image classification
### Classifier
* Input -> Classifier -> Output
* Input에 대한 정보를 output class로 분류하는 맵핑


### K-NN: k개의 가장 가까운 거리정보의 이미지에 따라 이미지를 분류하는 접근
    * 영상간의 유사도를 정의하는 것에 어려움
    * data의 크기에 따라 time complexity, memory complexity 증가

### Convolutional Neural Network (CNN)
* Fully-connected to every pixel
    * 특정 class의 w를 시각화 하면 학습 데이터 특정 class의 이미지들의 평균 영상과 같은 모습을 보입니다. 
    * 문제점
        1. 단순한 layer 구성으로 평균 이미지와 유사한 이미지 외에는 분류가 되지 않음
        2. 적용 시점 즉 위치에 따라 결과가 다르게 나옴 (같은 class 다른 패턴에 대해 정확도가 떨어짐)

* CNN
    * Locally connected를 이용 
        * 이미지의 공간적 특성을 고려하여 지역적인 특징들을 학습
        * 다른 위치의 같은 feature들에 대하여 같은 filter들을 사용하여 추출 가능
    * backbone
        * 이러한 local feature들을 학습해놓은 CNN model

## CNN architecture for image classification
1. AlexNet
    * LeNet-5에서 발전
        * 7-layer, 605k neurons, 60M param
        * ImageNet dataset 학습
        * ReLU, Dropout 사용
    * GPU memory의 부족을 해결하기 위하여 2개의 GPU 사용
    * Deprecated components 현재는 사용 되지 않는 AlexNet의 특징
        * first layer에서 11x11 filer size 사용 (큰 filter size)
        * Local Response Normalization (LRN)

2. VGG
    * Deeper architecture - 16, 19 layer
    * simple architecture - 3x3 conv, 2x2 max pooling, ReLU
    * better performance and generalization
    * 기본적인 구조는 AlexNet과 유사
        * 224x224 input
        * 학습 데이터의 RGB 각 채널의 평균을 빼서 학습
    * 3x3 conv with stride 1, 2x2 max pooling
        * 큰 receptive field size를 깊은 layer로 구현

# 과제

# 피어세션
## Ground Rule
- 공유 피피티를 활용해 자유롭게 익명으로 질문을 남긴다. - 보류
- 지각시 선택과제 리뷰를 담당한다.
- 모더레이터가 지정한 사람이 과제 리뷰
- 과제 제출기한이 끝나고 나면 팀 github 에 업로드
- 자유롭게 수업 정리 링크를 공유한다.
- 피어세션시간: 4:30~6:00
- 모더레이터
    - 이름순으로 모더레이터 진행
    - 곽지윤 → 김현수 → 백종원 → 이양재 → 임성민 → 조준희
    - 역할 : 당일 수업내용(시각화 수업 포함) 리뷰 / 회의록 작성 및 업로드

# 후기