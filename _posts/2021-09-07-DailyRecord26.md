---
title: "[boostcamp AI Tech] 학습기록 day26 (week6)"
date: 2021-09-07 20:00:00 -0400
categories:
use_math: true
---

# Image Classification
## Problem by deeper layer
* Deeper network, better performance
    * Large receptive field
    * More capacity and non-linearity
* Gradient vanishing / exploding

* Computationally complex

* Degradation problem
    * 처음에는 표현력이 과하게 좋아서 Overfitting이 일어날것이라고 판단
    * 현재는 Degradation problem라는 것이 발견

## CNN architecture for image classification
3. GoogLeNet
    * Inception module
        * Architecture
            * 1x1, 3x3, 5x5 conv
            * 3x3 pooling
            * concat outputs along channel axis
        * 1x1 conv
            * 많은 필터로 인하여 계산복잡도가 증가
            * 1x1 conv를 이용하여 channel demension 축소
            * 계산 복잡도 감소
    * stem network: vanila CNN
    * stacked inception modules
    * Auxiliary classifiers
        * Gradient vanishing 문제 해결
        * lower layer에 새로운 gradient를 추가
        * test에서는 사용하지 않음

4. ResNet
    * Deeper layer로 좋은 성능을 보여줌
    * Degradation problem
        * 학습을 함에 따라 training error와 test error가 saturation됨
        * overfitting 문제라면 training error가 deeper CNN이 lower CNN보다 낮아져야함
        * 최적화의 문제이다! (like Gradient vanishing, exploding)
    * Residual Block
        * Residual function: $H(x)=F(x)+x$
        * Target function: $F(x)=H(x)-x$
        * $H(x)$를 바로 학습하게 하는것이 아니라 $H(x)=F(x)+x$로 두어 residual만 학습함으로써 깊어져도 학습할 수 있도록 한다.
        * shortcut or skip connection을 통해 Residual function 구현
        * Analysis
            * $2^{n}$의 경우의 수가 gradient가 지나가는 input ouput을 만드는 방법이다.
            * 다양한 경로를 통해서 복잡한 mapping을 학습할 수 있다.
    * He initialization: Residual block에 적합한 initilization 방법

5. DenseNet
    * Dense block
        * 바로 직전 block의 정보 뿐만 아니라 더 이전 layer의 정보를 dense하게 전달한다.
        * 더 복잡한 mapping에 학습
        * Channel axis로 concat
            * + 신호의 결합 (ResNet)
            * Concat: 정보를 그대로 보존
    * Vanishing gradient 문제를 감소
    * Feature propagation을 강화
    * Feature의 재사용

6. SENet
    * 현재 주어진 activation간의 관계가 명확해지도록 chanel간의 관계를 모델링, 중요도를 파악, 중요한 특징을 attention
    * SE
        * Squeeze: global avg pooling을통해 공간정보 제거 후, 분포로 변환
        * Excitation: FC layer를 이용하여 체널의 attention score를 생성
        * Attention score를 이용하여 중요도에 따라 weight를 곱한다.

7. EfficientNet
    * 성능을 높이는 요소
        * Width scaling: channel 축 확장
        * Depth scling: deeper
        * Resolution scaling: input image resoultion 증가
    * compund scaling
        * width, depth, resulution scling을 종합

8. Deformable conv
    * irregular conv
        * 사람이나 자동차등 방향이나 움직임에 따라서 상대적인 형태가 변하는 object에 대하여 deformable한 것을 고려하기 위해 제안
    * 전형적인 conv에 대하어 offset filed에 따라서 w를 벌려주게 되고 irregularly sampling

### Summary
1. AlexNet: Simple architectur, lower layers, heavy memory size, low accuracy
2. VGGNet: simple 2x2 conv architecture, deeper than AlexNet, highest memory, heaviest computation
3. GoogLeNet: inception module and auxilary classifier
4. ResNet: Deeper layer with residual block, moderate efficiency
* CNN backnones
    * GoogLeNet이 다른 model 보다 효율적이지만 구조의 복잡함으로 사용하기 어려움
    * 심플한 VGGNet이나 ResNet 등을 backbone으로 사용

# 과제

# 피어세션
✅ 오늘의 피어세션 (모더레이터: 김현수)

1. 강의 요약
    - 발표자: 김현수
    - 내용: Image classification - 1
    - cnn from scratch : [https://setosa.io/ev/image-kernels/](https://setosa.io/ev/image-kernels/)
    - haar cascading

📢 내일 각자 해올 것

1. 모더레이터: 백종원 - 강의 2 강, Data Viz
2. 필수 과제 리뷰, 질문

📢 내일 우리 팀이 해야 할 일

1. 톡방 이용한 질문 확인

📢 Ground rule 수정사항

- 공유 피피티를 활용해 자유롭게 익명으로 질문을 남긴다. - 익명 질문톡방에 질문을 남긴다
    - 익명 카톡방 : [https://open.kakao.com/o/gLZi2Cyd](https://open.kakao.com/o/gLZi2Cyd)
    - !익명 카톡방 : [https://open.kakao.com/o/gQh02Cyd](https://open.kakao.com/o/gQh02Cyd)

# 후기