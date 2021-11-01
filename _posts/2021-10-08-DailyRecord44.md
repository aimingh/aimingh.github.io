---
title: "[boostcamp AI Tech] 학습기록 day45 (week10)"
date: 2021-10-01 20:00:00 -0400
categories:
use_math: true
---
# Object Detection
# MMDetection
![mmdetection_logo](/assets/image/level2_p/mmdet-logo.png)

* [MMDetection](https://github.com/open-mmlab/mmdetection)
* [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [MMdetection doc](https://mmdetection.readthedocs.io/en/latest/)

# Install
[get_started](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md)


#  Configuration
detection에 대하여 config system으로 모듈들을 통합하여 편리하게 관리
1. Model config: model architecture 
2. Dataset config: Dataset, Augmentation
3. Scheduler config: Optimizer, learning rate, scheduler
4. Runtime config: log, checkpoint save etc.

# 1. Model
## 2 stage model config
model.py의 기본적인 형태는 다음과 같이 구성
```
# fast_rcnn_r50)fpn.py의 예
model = dict(
    type='FastRCNN',
    backbone=dict(),
    neck=dict(),
    roi_head=dict(),
    train_cfg=dict(),
    test_cfg=dict()
```
1. type: detector의 이름
2. backbone: backbone을 정의
    * backbone의 종류:     
        * 'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet',
    'MobileNetV2', 'Res2Net', 'HourglassNet', 'DetectoRS_ResNet',
    'DetectoRS_ResNeXt', 'Darknet', 'ResNeSt', 'TridentResNet', 'CSPDarknet',
    'SwinTransformer', 'PyramidVisionTransformer', 'PyramidVisionTransformerV2'
```
# ResNet backbone의 예
backbone=dict(  # The config of backbone
        type='ResNet',  # backbone의 type
        depth=50,  # backbone의 깊이, resnet같은 경우 50이나 101
        num_stages=4,  # backbone의 stages 수
        out_indices=(0, 1, 2, 3),  # feature map을 추출할 stage의 각 index
        frozen_stages=1,  # frozen할 stage의 index
        norm_cfg=dict(  # normalization layers 정의
            type='BN',  # norm layer의 type (BN or GN)
            requires_grad=True),  # BN에서 beta와 gamma의 학습 유무를 결정
        norm_eval=True,  # BN의 statistics의 freeze 유무
        style='pytorch'， # backbone의 스타일, 
                          # 'pytorch'는 stride 2 layers are in 3x3 conv, 
                          # 'caffe'는 stride 2 layers are in 1x1 convs.
    	init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),  # ImageNet으로 pretrained된 backbone을 로드
```
3. neck: neck modeul를 정의
```
neck=dict(
    type='FPN', # neck의 type
    in_channels=[256, 512, 1024, 2048], # 각각의 feature map의 입력 채널의 수
    out_channels=256,   # 출력 채널의 수
    num_outs=5),
```
4. rpn_head
```
rpn_head=dict(
    type='RPNHead',
    in_channels=256,
    feat_channels=256,
    anchor_generator=dict(
        type='AnchorGenerator',
        scales=[8],
        ratios=[0.5, 1.0, 2.0],
        strides=[4, 8, 16, 32, 64]),
    bbox_coder=dict(
        type='DeltaXYWHBBoxCoder',
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0]),
    loss_cls=dict(
        type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
    loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
```
4. roi_head
```
roi_head=dict(
    type='StandardRoIHead',
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='Shared2FCBBoxHead',
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=80,
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0., 0., 0., 0.],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        reg_class_agnostic=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
```
5. train_cfg
```
train_cfg=dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            match_low_quality=True,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_pre=2000,
        max_per_img=1000,
        nms=dict(type='nms', iou_threshold=0.7),
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            match_low_quality=False,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False)),
```
6. test_cfg
```    
test_cfg=dict(
    rpn=dict(
        nms_pre=1000,
        max_per_img=1000,
        nms=dict(type='nms', iou_threshold=0.7),
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100)
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
))
```

# 2. Dataset
dataset.py의 기본 요소(coco_detection.py의 예)
## 기본 구성
```
# dataset settings
dataset_type = 'CocoDataset'
data_root = ''
img_norm_cfg = dict()
train_pipeline = []
test_pipeline = []
data = dict()
evaluation = dict()
```
1. dataset_type: dataset의 형식
```
dataset_type = 'CocoDataset'
```
2. data_root: dataset의 root path
```
data_root = 'data/coco/'
```
3. img_norm_cfg: dataset의 normalization을 위한 평균, 분산 (아래 pipeline에서 사용)
```
dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
```
4. train_pipeline: 학습 augmentation같은 입력 pipeline을 설정
```
train_pipeline = [
    # file load
    dict(type='LoadImageFromFile'), # file로부터 이미지를 load
    dict(type='LoadAnnotations', with_bbox=True), # annotation load (detection이기 때문에 bbox에 대한 annotation만 불러온다. segmentation같은 경우에는 with_mask=True를 추가)
    # augmentation
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True), # resize 설정
    dict(type='RandomFlip', flip_ratio=0.5), # random flip
    dict(type='Normalize', **img_norm_cfg), # normalize (위의 img_norm_cfg를 이용하여 image를 normalization)
    dict(type='Pad', size_divisor=32), # padding 설정
    dict(type='DefaultFormatBundle'), # pipeline에서 데이터를 모으기 위한 default format
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']), # pipeline에서 통과시켜야할 데이터의 key값을 설정
]
```
5. test_pipeline: train_pipeline과 같이 테스트에 대한 pipeline을 설정
```
test_pipeline = [
    dict(type='LoadImageFromFile'), # file load
    # train과 달리 annotation을 load하지 않음
    dict(   # test augmentation을 캡슐화
        type='MultiScaleFlipAug',   # 
        img_scale=(1333, 800),  # img size 결정
        flip=False, # flip 유무
        transforms=[
            dict(type='Resize', keep_ratio=True), # resize
            dict(type='RandomFlip'),    # 추가되어있지만 flip=False으로 인해 사용 X
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']), # img를 tensor로 converting
            dict(type='Collect', keys=['img']),
        ])
]
```
6. data: data 입력 pipeline에 대한 option을 설정
```
data = dict(
    samples_per_gpu=2,  # gpu당 batch_size의 크기를 결정 
    workers_per_gpu=2,  # 각 gpu당 data load에 대한 worker 수
    train=dict(     # train dataset 설정
        type=dataset_type,  # type 결정
        ann_file=data_root + 'annotations/instances_train2017.json',    # annotation file path list 파일의 path
        img_prefix=data_root + 'train2017/',    # img path 앞에 붙을 prefix
        pipeline=train_pipeline),   # train pipeline
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
```

# 3. scheduler
schedule.py의 구성 (schedule_1x.py의 예)  
Optimizer와 학습 scheduler의 설정
```
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# scheduler
lr_config = dict(
    policy='step',      # scheduler policy
    warmup='linear',    # warmup policy
    warmup_iters=500,   # warmup하는 iter의 수
    warmup_ratio=0.001, # warmup에 사용되는 starting lr의 비율
    step=[8, 11]) # lr을 감소시킬 step
runner = dict(type='EpochBasedRunner', max_epochs=12)   # epoch 기준으로 학습, epoch의 수 설정
```

# 4. Runtime
그 외적으로 필요한 option들의 설정
```
checkpoint_config = dict(interval=1)    # checkpoint, 1 interval 마다 저장 
# yapf:disable
log_config = dict(  # log에 대한 설정
    interval=50,    # 50마다 저장
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')    # tensorboard logger도 지원
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
```

# final config
위의 각각 따로 떨어진 config를 하나로 통합하는 config file
```
_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]
```

# 학습
```
python tools/train.py configs_to_path
# configs_to_path: final config 파일의 path
```

# Custom
## 1. backbone model 등록
* mmdet/models/backbones/에 모델 파일 생성
* 등록한 model file에서 모델 class위에 @BACKBONES.register_module()으로 등록
* mmdet/models/backbones/__init__.py에 모델 이름 등록
* config 파일 작성
* (mmdet이 이미 설치되었다면) 등록된 mmdet을 삭제
```
pip uninstall mmdet
```
* custom한 mmdet 설치
```
pip install -v -e .
```

# 자주일어나는 Error fix
## 1. 사용하려는 dataset의 classes 선언
* classes = ['a', 'b', 'c'] 추가
## 2. num_classes error
* 모델에 정의된 num_classes들을 dataset에 맞게 변경

# 과제 및 대회 수행과정
* mmdetection 정리
* swin transform 실험
* ensemble 실험

# 피어세션
* https://www.notion.so/20211005-50a305e681c14a84aa6f945f64c2ed8f
* https://www.notion.so/20211007-2b1255fb3ecb41d8962e35ab95bded75
* https://www.notion.so/20211008-9a38953c883e46efbf3d0fccd1800998

<!-- # 멘토링
## object detectoin tip
1.object detection에 효과적인 augmentation 전략은? (우선실험)
    object detection에서 multi scale이 중요
    custom aug를 한다면 박스의 geometry를 확인하는것이 중요
    object detection의 전용 augmentation은 없다고 봐야한다.
    TTA가 많이 사용


2. 특정 input에 대한 모델을 여러개 사용하여 앙상블


3. 수도라벨링
    classification에서는 prop이 작은애들을 사용해도 좋았던 경험잉 있었으나
    object detection에서는 높은 prop을 가진 box들을 사용하는것이 좋았다. -->
    

# 학습회고 
mmdetection에 대하여 정리하였고 github 블로그에서 계속 문제가 되었던 latex 에러를 해결하였다. 그리고 기본적인 model 외에 swin tranformer같은 backbone도 이용하여 실험하기 시작하였는데 모델에 대한 이해가 부족한 것 같다. mmdetection으로 이미 config 등이 다 구현되어있기 때문에 일부 파라미터만 바꾸면서 실험하는데 mmdetection이 이해한 이후에는 확실히 편하게 사용할 수 있는것 같다.