---
layout: post
title: "[boostcamp AI Tech] 학습기록 day10 (week2)"
date: 2021-08-13 20:42:05 -0400
categories:
use_math: true
---

# Deep Learning Basic
## Generative Models
* Generative model은 주어진 학습데이터를 학습하여 원래 데이터의 분포와 유사한 새로운 데이터를 생성하는 모델입니다. 
* generative model은 두가지 기능을 가집니다.
    * Generation: 학습데이터의 분포에 따라 , 학습데이터에 없지만 학습데이터의 분포와 유사한 데이터를 생성
    * Density estimation: 주어진 데이터가 어떤 데이터의 분포인지 판별

* generative model의 종류
    explicit model: 입력이 주어졌을 때, 입력에 대한 확률값을 얻을 수 있는 모델 (generation + density estimation)
    implicit model: 단순히 generation만 하는 모델 (generation)

<!-- 통계부분 충분히 이해가 안간다... 좀더 공부하고 정리할 부분... 주말 작업? -->
<!-- * Basic discrete distributions
    * Bernoulli distribution: 2개의 카테고리를 가지는 분포입니다. 하나의 확률로 표현 가능합니다.
    * categorical distribution: m개의 카테고리가 있을 때 m-1개희 확률값으로 표현 할 수 있습니다.

* Strucctire Through Independence
    *  
    
-->

* Auto-regressive Model
    * 28x28 이진 픽셀 숫자 이미지가 있다고 가정하면
    * 우리의 목표는 $p(x)=p\left(x_{1}, \ldots, x_{784}\right)$ over $x \in\{0,1\}^{784}$를 학습하는 것입니다.
    * chain rull을 이용하여 표현하여 만든 모델이 autoregressive model입니다.

* NADE: Neural Autoregressive Density Estimator
    
    * explicit model로 생성 뿐만 아니라 입력된 데이터 분포의 확률을 알 수 있습니다.
    $$
    p\left(x_{1}, \ldots, x_{784}\right)=p\left(x_{1}\right) p\left(x_{2} \mid x_{1}\right) \cdots p\left(x_{784} \mid x_{1: 783}\right)
    $$

* Pixel RNN
    * RNN을 Auto-regressive Model를 정의하는데 사용한 모델입니다.
    * odering에 따라 row LSTM과 Diafonal BiLSTM으로 나뉩니다.

<!-- * Latent Variable Models 추천?-->


* Variational Auto-encoder
    * Variational inference (VI)
        * posteriot distribution을 찾기위한 variational distribution을 최적화하는 것이 목적입니다.
            * posteriot distribution: observation이 주어졌을 때 관심있어하는 확률 분포
            * variational distribution: posteriot distribution를 학습할 수 있는, 근사할 수 있는 확률 분포
        * KL Divergence를 이용하여 최소화합니다.
    
    * ELBo (Evidence Lower Bound) trick
        * ELBO를 최대화 함으로써 반대급부로 KL Divergence를 최소화 합니다.
        * 결과론적으로 임의의 posteriot distribution과 계산하려고 하는 variational distribution의 거리 (KL Divergence)를 줄일 수 있습니다.
        $$
        \begin{aligned}
        \ln p_{\theta}(D) &=\mathbb{E}_{q_{\phi}(z \mid x)}\left[\ln p_{\theta}(x)\right] \\
        &=\mathbb{E}_{q_{\phi}(z \mid x)}\left[\ln \frac{p_{\theta}(x, z)}{p_{\theta}(z \mid x)}\right] \\
        &=\mathbb{E}_{q_{\phi}(z \mid x)}\left[\ln \frac{p_{\theta}(x, z) q_{\phi}(z \mid x)}{q_{\phi}(z \mid x) p_{\theta}(z \mid x)}\right] \\
        &=\mathbb{E}_{q_{\phi}(z \mid x)}\left[\ln \frac{p_{\theta}(x, z)}{q_{\phi}(z \mid x)}\right]+\mathbb{E}_{q_{\phi}(z \mid x)}\left[\ln \frac{q_{\phi}(z \mid x)}{p_{\theta}(z \mid x)}\right] \\
        &=\underbrace{\mathbb{E}_{q_{\phi}(z \mid x)}\left[\ln \frac{p_{\theta}(x, z)}{q_{\phi}(z \mid x)}\right]}_{\text {ELBO } \uparrow}+\underbrace{D_{K L}\left(q_{\phi}(z \mid x) \| p_{\theta}(z \mid x)\right)}_{\text {Objective } \downarrow}
        \end{aligned}
        $$

        *  ELBO를 나눠보면 Reconstruction term과 Prior fitting term으로 나뉩니다.
            * Reconstruction term가 auto-encoder에서 Reconstruction loss에 해당합니다.
        $$
        \begin{aligned}
        {\mathbb{E}_{q_{\phi}(z \mid x)}\left[\ln \frac{p_{\theta}(x, z)}{q_{\phi}(z \mid x)}\right]}_{\text {ELBO } \uparrow}&=\int_{\ln \frac{p_{\theta}(x \mid z) p(z)}{q_{\phi}(z \mid x)} q_{\phi}(z \mid x) d z} \\
        
        &=\underbrace{\mathbb{E}_{q_{\phi}(z \mid x)}\left[p_{\theta}(x \mid z)\right]}_{\text {Reconstruction Term }}-\underbrace{D_{K L}\left(q_{\phi}(z \mid x) \| p(z)\right)}_{\text {Prior Fitting Term }}
        \end{aligned}
        $$

        * key limitation
            * intractable model: likelihood를 계산하기 어렵습니다.
            * Prior Fitting Term은 미분 가능해야 하고, 그래서 다양한 latent prior distribution을 사용하기 어렵습니다.
            $$
            D_{K L}\left(q_{\phi}(z \mid x) \| \mathcal{N}(0, I)\right)=\frac{1}{2} \sum_{i=1}^{D}\left(\sigma_{z_{i}}^{2}+\mu_{z_{i}}^{2}-\ln \left(\sigma_{z_{i}}^{2}\right)-1\right)
            $$

* Adversarial Auto-enccoder
    * Variational Auto-encoder의 Prior fitting term을 GAN (Generative Adversarial Network) objective로 바꾼 모델입니다.


* GAN (generative adversarial network)
    * 진짜같은 가짜를 만드는 generator와 진짜와 가짜를 discriminator를 경쟁시켜 성능을 향상시키는 모델입니다.
    $$
    \min _{\Omega} \max _{\Pi} V(D, G)=\mathbb{E}_{\boldsymbol{x} \sim p_{\text {data }}(\boldsymbol{x})}[\log D(\boldsymbol{x})]+\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}[\log (1-D(G(\boldsymbol{z})))]
    $$
    * discriminator
        $$
        \max _{D} V(G, D)=E_{\mathbf{x} \sim p_{\text {data }}}[\log D(\mathbf{x})]+E_{\mathbf{x} \sim p_{G}}[\log (1-D(\mathbf{x}))]
        $$
    * generator
        $$
        \min _{G} V(G, D)=E_{\mathbf{x} \sim p_{\text {data }}}[\log D(\mathbf{x})]+E_{\mathbf{x} \sim p_{G}}[\log (1-D(\mathbf{x}))]
        $$

* DCGAN
    * image 도메인에서 활용되는 GAN입니다.

* info-GAN: 
    * 생성기로 이미지만 만드는것이 아니라 랜덤한 원핫벡터를 학습에 이용합니다.
    * GAN이 condition 벡터에 집중하게 하여 multimodal distribution을 학습하는 것을 c라는 벡터를 통해 잡아주는 효과가 있다.

* Text2Image
    * 문장을 이미지로 바꿔주는 모델

* Puzzle-GAN
    * 이미지 안의 sub patch를 통해 원래 이미지를 복원하는 모델

* cycleGAN
    * 2개의 GAN을 이용하여 도메인을 바꿀 수 있는 모델
    * cycle-consistency

* star-GAN
    * 다른 도메인으로 바꾸면서 생성할 이미지의 특정 요소를 컨트롤 할 수 있는 모델입니다.

* Progressive GAN
    * 저차원부터 고차원으로 늘려가며 학습


# [피어세션 - 팀회고록](https://hackmd.io/@ai17/r1GtwsXeY)

# 후기
