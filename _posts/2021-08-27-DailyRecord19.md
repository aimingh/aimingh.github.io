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

# [피어세션 - 팀회고록](https://hackmd.io/45OwzSbOSNOy0C3vCIfVfA )

# 후기
1주일이 훌쩍 지나가버린것 같다. 아쉽지만 학습횟수를 많이 사용하지 못한것이 아쉽다. 
특히 오늘은 random_split으로 나눈 validation set이 전혀 기능하지 않는것에 대해 고민했다.
최종적으로 같은 사람이 마스크를 쓰고 있는 여러 이미지가 있는것으로 train dataset과 validation dataset이 완전히 독립적으로 분리되지 않은 탓이라 결론지었다. 이부분을 해결하려고 하였는데 baseline code에 이미 관련 내용이 있는 것을 확인하였다. baseline code가 이미 여러가지 고려된 code로 다음주를 대비하여 작업하였던 code를 baseline code와 융합하는 작업을 미리 진행하는 것이 좋아보인다. 