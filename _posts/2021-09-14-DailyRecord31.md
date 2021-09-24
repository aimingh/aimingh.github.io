---
title: "[boostcamp AI Tech] 학습기록 day31 (week7)"
date: 2021-09-14 20:00:00 -0400
categories:
use_math: true
---

# Instance/Panoptic Segmentation
## Instance/Panoptic Segmentation이란?
* Instance Segmentation은 sementic segmentation과 달리 instance의 구분이 가능합니다.
* instance Segmentation = object segmentation (Semantic segmentation without background) + instance information

* Panoptic segmentation = Instance Segmentation + background information

## Mask R-CNN
* faster R-CNN에서 개선
* Mask R-CNN = Faster R-CNN + Mask branch
* Region proposal network에서 ROI pooling (정수 좌표만) -> ROIAlign (interpolation을 사용하여 소숫점 좌표)
* key point branch를 만들어서 skeleton도 추정 가능

## YOLACT (You Only Look At CoefficienTs)

## YolactEdge

## UPSNet

## VPSNet (for video)


# Landmark Localization
## Landmark Localization이란?
* 얼굴이나 사람의 포즈등의 모양을 정의할 수 있는 point 즉 landmark의 위치를 추정하는 application


## Coordinate regression vs Heatmap classification
*  Coordinate regression
    * 2N개의 landmark들을 regression
    * 부정확하거나 바이어스될 수 있습니다.
* Heatmap classification
    * 모든 class 즉 N개의 score map을 추정하는 모델
    * 성능이 더 좋아지지만 계산량이 많아진다는 단점이 있습니다.
    * Landmark location to Gaussian heatmap
        * 어느 한점에서 goussian 분포를 이용하여 heatmap의 형태로 변환
    $$
    G_{\sigma}(x, y)=\exp \left(-\frac{\left(x-x_{c}\right)^{2}+\left(y-y_{c}\right)^{2}}{2 \sigma^{2}}\right)
    $$

## Hourglass network
* stacked hourglass modules
    * 여러개의 U-net과 유사한 구조가 반복되어 stack되는 형태
        * Unet과 달리 concat이 아닌 +로 연결 (FCN과 유사하다고 볼 수도 있다.)
    * 여러번 반복하며 정교하게 결과를 추정

## DensePose
* landmark를 세분화하여 3D 정보를 얻을 수 있는 모델
* UV map을 사용
    * UV map과 3D mesh와 mapping
* DensePose R-CNN = Faster R-CNN + 3D surface regression branch

## RetinaFace
* RetinaFace = FPN + Multi-task branch

# Detecting object as keypoint
* Object detection을 할 때 bbox가 아니라 keypoint의 형태로 detection

## CornerNet
* 성능보다는 속도


## CenterNet
* center point의 중요성을 강조, 학습에 추가
* bounding box = (W, H, x_center, y_center)



# 과제

# 피어세션
* 추천 영상 및 자료
    * [일반인을 대상으로 한 computer vision](https://www.ted.com/talks/fei_fei_li_how_we_re_teaching_computers_to_understand_pictures)
    * [cs 231](http://cs231n.stanford.edu/)
    * [cs 231 번역](https://github.com/visionNoob/CS231N_17_KOR_SUB)