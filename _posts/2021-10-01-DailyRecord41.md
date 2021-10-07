---
title: "[boostcamp AI Tech] 학습기록 day41 (week9)"
date: 2021-10-01 20:00:00 -0400
categories:
use_math: true
---

# Object detection overview
## overview
* object detection = classification + localization
* 자율주행, OCR, 의료영상 등에 이용
* history
    * R-CNN - Fast R-CNN - Faster R-CNN - YOLOv1 - SSD - YOLOv2
    * FPN - RetinaNet - YOLOv3 - PANet - EfficientDet - Swin-T

## Evaluation 
* mAP (mean average precision): 각 클래스당 평균 ap
* Confusion matrix
    * TP (True Positive): 검출 되어야할 것이 검출됨
    * FN (False Negative): 검출되어야 할 것이 검출되지 않음
    * FP (False Positive): 검출 되지 말아야 할 것이 검출됨 
    * TN (True Negative): 검출되지 말아야할 것이 검출되지 않았음

|  | Predicted Positive | Predicted Negative |
|----|----|----|
| 정답 Positive | TP  | FN |
| 정답 Negative | FP  | TN |

* Precision
$$
\lim_{x\to 0}{\frac{e^x-1}{2x}}
\overset{\left[\frac{0}{0}\right]}{\underset{\mathrm{H}}{=}}
\lim_{x\to 0}{\frac{e^x}{2}}={\frac{1}{2}}
$$
$$
\text { Precision }=\frac{T P}{T P+F P}=\frac{T P}{\text { All Detections }}
$$

* Recall
$$
\text { Recall }=\frac{T P}{T P+F N}=\frac{T P}{\text { All Ground truths }}
$$

* PR Curve
    * confidence 순으로 나열하여 각각의 precision(x축)과 recall(y축)로 표시 
* AP (average precision)
    * PR curve를 계단처럼 감소하도록 다듬어주고 curve 아래의 면적을 의미한다.
* mAP
$$
m A P=\frac{1}{n} \sum_{k=1}^{k=n} A P_{k}
$$
$A P_{k}$: k class의 AP
&n&: class의 수

* IOU (Intersection Over Union)
$$
I O U=\frac{\text { overlapping region }}{\text { combined region }}
$$

* FPS (frame per second)
* FLOPs(Floating Point Operations)

## Library
* MMDetection
* Detectron2
* YOLOv5
* EfficientDet

# 2 stage detectors
* 입력이미지 -> model 연산 -> Localization (1st stage) -> Classification (2nd stage)

## R-CNN
1. input image
2. Extract region proposal
    * sliding window
    * selective search
3. Extract CNN feature
4. classification

## SPPNet
1. input image
2. Extract CNN feature
3. spatial pyramid pooling
4. classification

## Fast R-CNN
1. image를 한번에 CNN으로 특징 추출
2. RoI projection으로 RoI 계산
3. RoI Pooling
4. classification

## Faster R-CNN
* RPN을 통하여 RoI계산
* RPN
    * Anchor box
* RPN을 통한 모든 과정을 딥러닝 이용하여 연산속도 개선

# Neck
* 다양한 크기의 객체를 탐지하기 위해서 상위레벨의 특징과 하위레벨의 특징을 교환

## FPN
* 여러 scale의 물체를 찾도록 설계
* 여러 level의 특징들을 사용

## PANet
* FPN에서 실제 CNN 구조는 깁은 구조여서 Top down layer를 통과하면서 많은 정보를 잃어버림
* Bottom up path을 추가정보의 손실을 방지

## DetectoRS
* 반복구조를 통해 성능 향상

## BiFPN
* FPN에서처럼 단순 summation라는 것이 아니라 특징별로 가중치를 부여하여 summation

## NASFPN
* FPN 구조를 NAS를 통해 탐색
* 데이터셋에 특화되어 범용적이지 못함
* 구조를 찾는데 많은 비용 발생

## AugFPN
* Bottom up까지 하면서 잃어버린 semantic feature들의 정보를 전달

# 1 stage Detectors
* 2 stage 방식과 달리 localization과 classification을 동시에 진행
* 영역을 추출하지 않고 전체 이미지를 사용하기 때문에 객체에 대한 이해가 높음
    * 배경을 잘 구분

## YOLOv1
* region proposal 단계 삭제
* 전체 이미지에서 bbox와 클래스를 동시에 예측
* faster R-CNN보다 6배 빠른 속도 (real time detection)
* 다른 real time detector보다 높은 정확도
* 전체 이미지를 이용하여 객체의 맥락적 정보를 파악하고 일반화된 특징을 학습

## SSD
* YOLOv1은 사용하는 그리드보다 작은 영역에서 물체 검출 불가능
* 마지막 feature만 사용하여 정확도 하락 
* 여러 scale의 feature map을 사용하여 성능 향상
* 낮은 level의 feature map을 사용하여 작은 물체의 검출 성능을 향상

## YOLO follow-up
### YOLOv3
* 성능향상
    * batch nomalization
    * high resoluition
    * Convolution with anchor boxes
    * Fine grained features
    * Multi scale training
* 속도 향상
    * darknet-19 사용
* 더 많은 class 사용
    * word tree 형태의 계층적인 class 추정

### YOLOv4
* darknet-53
* Multi scale Feature maps

## RetinaNet
* 기존의 문제점
    * Class imbalance
    * Anchor Box 대부분 Negative Samples

* focal loss
    * cross entropy loss scaling factor
    * 쉬우면 작은 기울기, 어려운 문제에는 큰 기울기
    * Object Detection 외에도 Class imbalance가 심한 Dataset을 사용할떄 focal loss를 사용

## MMDetection office hour
<img src="/assets/image/level2_p/mmdet-logo.png" width="90%" height="90%" title="mmdetection_logo"/> 
![mmdetection_logo](/assets/image/level2_p/mmdet-logo.png)


# 과제
## Special Mission 1 - mAP metric code
## Special Mission 2 - Faster RCNN3
## Special Mission 3 - Baseline


# 피어세션
* https://www.notion.so/20210927-c778a219f55b44c196987e00936a6f5e
* https://www.notion.so/20210928-2277e84141a44e078092b8f27533a08d
* https://www.notion.so/20210929-8af813d7fa114cd884408da3092ac9f9
* https://www.notion.so/20210930-0d21277f7ef54dad9f1e6af47ab8dc10
* https://www.notion.so/20211001-a6d1bfc2c6194e33872f5742361359ca

# 학습회고 
프로젝트 첫주 이론과 베이스라인 코드에 집중했다. 다만 이론을 자세하게는 정리하지 못했는데 프로젝트와 병행하면서 좀더 정리할 수 있음녀 좋을 것 같다.