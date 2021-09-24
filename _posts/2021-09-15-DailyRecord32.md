---
title: "[boostcamp AI Tech] 학습기록 day32 (week7)"
date: 2021-09-15 20:00:00 -0400
categories:
use_math: true
---

# Conditional Generative Model
* Image generation 분야의 generative model에 condition을 주어서 다루는 모델
* generative model은 확률 분포를 modeling하는데 condition을 어떻데 주냐에 따라 출력을 control
    * generative model $y=P(x)$
    * conditional generative model $y=P(x|c)$

## Generative Adversarial Network (GAN)
* Generator
* Discriminator

## Example
* Low resolution audio -> High resolution audio
* Chinese sentence -> English sentence
* An article's title and subtitle -> A full article

* Translation from image to image
    * Input image -> Monet, Van Gogh, Cezanne image

* Super Resolution
    * Low resolution image -> high resolution image

## Pix2pix
* image to image translation task
    * Label to Stree Scene
    * Aerial to Map
    * Label to Facade
    * BW to Color
    * Day to Night
    * Edge to Photo
    * ...

* Loss
    * Total loss =  GAN loss + L1 loss
    $$
    G^{*}=\arg \min _{G} \max _{D} \mathcal{L}_{c G A N}(G, D)+X \mathcal{L}_{L 1}(G)
    $$
    $$
    \begin{aligned}
    \mathcal{L}_{c G A N}(G, D)=& \mathbb{E}_{x, y}[\log D(x, y)]+\\
    & \mathbb{E}_{x, z}[\log (1-D(x, G(x, z))]
    \end{aligned}
    $$
    $$
    \mathcal{L}_{L 1}(G)=\mathbb{E}_{x, y, z}\left[\|y-G(x, z)\|_{1}\right]
    $$

## CycleGAN
* pix2pix의 경우 pariwise data가 반드시 필요
* CycleGAN은 non-pariwise data으로 학습하는 방법

* CycleGAN loss = GAN loss (both direction) + Cycle consistency loss
    $$
    L_{G A N}(X \rightarrow Y)+L_{G A N}(Y \rightarrow X)+L_{\text {cycle }}(G, F)
    $$
    * GAN loss (both direction)
        $$
        L\left(D_{X}\right)+L\left(D_{Y}\right)+L(G)+L(F)
        $$
        * Mode Collapse 문제가 발생 (input에 상관없이 하나의 output만이 출력되는 형태)
    * Cycle consistency loss
        * Mode Collapse를 해결하기 위해서 도입
        * X -> Y -> X
        * Y -> X -> Y
        * 복원된 이미지가 원본과 같도록 학습
        * self-supervised loss

## Perceptual loss
* GAN은 학습하기 어려움 (alternating training이 필요)
* GAN 말고 high-quality image를 얻을 수 있는 방법
* 학습된 loss를 측정하기 위해 pre-trained model이 필요하다는 제약이 있다.
* 반면에 GAN loss는 pre-trained model이 필요 없어서 다양한 applications에 적용이 가능하다.

    |                  | GAN loss (Adversarial loss) |       Perceptual loss       |
    |------------------|-----------------------------|-----------------------------|
    |  train and code  |       Relatively hard       |            Simple           |
    |Pre-trained model |        Don't require        |           Require           |

* pre-trained model의 filter를 보면 사람의 시각적 인식과 유사하다.
    * Perception of pre-trained model: image -> perceptual space (transform)
* GAN loss를 사용하지 않고 Style transfer와 같은 task를 잘 학습

###  Model with perceptual loss
* Image Transform Net: Input -> Output (tansform)
* Loss Net
    * 생성된 이미지와 타겟 (style target, content target) 사이의 style and feature loss를 계산
    * 자주 사용되는 모델: pre-trained VGG model with ImageNet
    * Frozen (fix parameter during training)

    * feature recontruction loss
        * output과 contents target의 feature map사이의 L2 loss 
        * Semantic한 contents를 인식하고 비슷해야 한다.
    * style recontruction loss
        * output과 contents target의 feature map을 gram matrix 생성
        * gram matrices사이의 L2 loss 
        * gram matrix로 공간의 위치가 아니라 이미지 전반적인 통계적 특성을 비교
        * gram matrices
            * 전반적인 style을 보기 위해 pooling을 통해 공간 정보를 압축한다.
            * (C, H, W) -> (C, H x W)
            * (C, H x W)(C, H x W)^T = (C, C)
            * channel들 간의 상대적인 관계 (correlation)를 포함
            <!-- * trained model의 channel은 어떤 symbol을 detection
            * 각각의 symbol이 동시에 발생하는지에 대한 정보가 gram matrix에 저장 (style들의 동시발생)-->

## Various GAN applications
1. Deepfake
* 사람의 비디오나 영상에서 얼굴, 목소리를 다른 사람의 얼굴, 목소리로 변환
* 안좋은 영향이 발생할 수 있다.
    * Deepfake Detection challenge
    * 범죄에 악용될 수 있기 때문에 윤리적인 부분도 고려해야한다.

2. Face de-identification
* 프라이버시를 보호하기 위해 사람의 얼굴을 약간 변형
* 사람은 비슷하게 인식할 수 있지만 컴퓨터는 같은 사람이라고 인식하기 어려움

3. Face anonymization with passcode
* password를 이용하여 익명화
* password를 통해 암호화와 복호화가 가능

4. Video translation (manipulation)
* Pose transfer
    * CG와 같은 효과
* video to video translation
    * semantic image -> real image
* vedio to game
    * 비디오 정보를 게임처럼 콘트롤할 수 있는 영상으로 변환

# 과제

# 피어세션
[ 2021년 9월15일  수요일 회의록 ]

✅ 오늘의 피어세션 (모더레이터: 김현수)

1. 강의 요약
    - 발표자: 김현수
    - 내용 : Conditional Generative Model

📢 토의 내용

1. Pix2Pix loss 의 GAN loss 에서 x, y?
2. cGAN에서 L1 Loss를 사용하는 이유?
3. Super Resolution - HR (원본) 이미지와 LR (down sampling) 이미지를 cGAN에 학습시켜 super resolution 가능하도록 한다. 
4. 멘토링 시간 9/17(금) 4:30pm  & 금요일 피어세션 시간 4:00 pm에 팀회고 및 강의 리뷰

📢 내일 각자 해올 것

1. 모더레이터: 백종원  캠퍼님