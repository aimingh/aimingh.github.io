---
title: "[boostcamp AI Tech] 학습기록 day15 (week4)"
date: 2021-08-23 20:42:52 -0400
categories:
use_math: true
---

# Image Classification and EDA
## 문제 해결 과정
1. Domain Understanding
2. Data mining
3. Data Analysis
4. Data Processing
5. Modeling
6. Training
7. Deploy

## Image Classification
* 이미지 영상을 특정한 class로 분류하는것
* Input: Image
* Output: Class

## EDA (Exploraoty Data Analysis)
* 탐색적 데이터 분석으로 데이터를 이해하기위한 노력을 의미합니다.
* python, Exel 등
* 떠오르는 아이디어를 적용하기 전에 EDA를 통해 데이터를 탐색하는 과정이 반복됩니다.

## 이미지
* 2차원 평면 위에 그려진 시각적 표현물 (https://ko.wikipedia.org/wiki/%EC%98%81%EC%83%81)
* 컴퓨터는 이미지를 표현하기 위해 x축의 위치, y축의 위치에 화소(주로 RGB)에 대한 정보를 가지는 행렬식으로 표현됩니다.
* gray image는 빛에 대한 intensity를 하나의 값으로 표현하기 때문에 (width, hight)의 행렬
* color image는 RGB 하나의 값으로 표현하기 때문에 (width, hight, channel)의 행렬

## Model
* output = Model(input)
* input을 받아 output을 출력으로 내는 task를 의미합니다.
* Image Classification Model
    * class = ClassificationModel(Image)
    * 위의 형태처럼 이미지를 입력으로 class를 출력으로 가지는 model을 의미합니다.

# [피어세션 - 팀회고록](https://hackmd.io/@ai17/r15fJRlWt)

# 후기
