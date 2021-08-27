---
title: "[boostcamp AI Tech] 학습기록 day19 (week4)"
date: 2021-08-27 19:24:37 -0400
categories:
use_math: true
---

#
## Ensemble
여러개의 다른 모델을 사용하여 성능을 높이는 기법
* 실제는 여러개의 모델을 사용하는 것을 그만큼 시간을 많이 사용하기 때문에 트레이드오프가 있다.
* bagging, boosting
* model averageing (Voting)
    * hard voting, soft voting
* cross validation
* Test Time Augmentation

* Hyper parameter 
    * random, grid, bayesian 
    * Optuna

## Experiment Toolkits and Tips
* Training visualization
    * Tensorboard
    ```
    from torch.utils.tensorboard import SummaryWriter
    logger = SummaryWriter(log_dir=f"logs/{folder_or_proj_name})
    logger.add_scalar('Train/loss', train_loss, epoch + len(train_loader) + idx)
    logger.add_scalar('Train/accuracy', train_acc, epoch*len(train_loader) + idx)

    grid = torchvision.utils.make_grid(inputs)
    logger.add_image('image', grid, 0)
    logger.add_graph('image', inputs)
    ```
    * wandb (weight and bias)
        * 딥러닝 로그의 깃허브

* Machine Learning Project
    * Jupyter Notebook
        * cell 단위로 실행할 수 있는 장점
        * EDA에 매우 편리
    * Python IDLE
        * 구현을 한번만 하면 언제나 간편하게 코드를 재사용
        * vscode등의 강력한 디버깅툴
        * config.json 등을 이용한 실험 핸들링

* Other tips
    * 다른 사람들의 코드나 분석글에서 필자의 생각과 흐름을 읽는것이 중요하다.
    * 코드를 볼떄 응용 가능할 정도로 확인해야 나중에 좋다.
    * paper with codes 등에서 최신 논문과 코드 데이터셋을 확인
    * 자신의 내용을 공유 - 새로운것을 배울 수 있다.

# [피어세션 - 팀회고록]()

# 후기
