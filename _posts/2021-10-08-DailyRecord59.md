---
title: "[boostcamp AI Tech] 학습기록 day59 (week12)"
date: 2021-10-29 20:00:00 -0400
categories:
use_math: true
---
# Segmentation
## code
멘토링을 통해 코드를 간소화 하고 앞으로 유지보수를 고려하는 리팩토링, 클린코드, 구조화된 코드(?)에 대해 듣고 고민해보았다.
필요한 함수나 코드들을 구조화하고 나누고 함수화하여 정리해보았다.
## 구조
```

|-- saved
|   `-- *.pt
|-- seg_utils
|   |-- Dataset.py
|   |-- loss.py
|   |-- models.py
|   |-- train_validation.py
|   `-- utils.py
|-- sh
|   `-- *.sh
|-- submission
|   `-- *.csv
|-- vis
|   `-- *.png
|-- test.py
|-- train.py
|-- visualize.py
|-- EDA.ipynb
|-- class_dict.csv
```
* save - 학습된 모델이 저장되는 폴더 *.pt로 저장
* seg utils - 학습에 관련된 함수, 클래스 모음
    * Dataset.py - 데이터셋을 로드하는 함수들의 모음
    * loss.py - loss function 모음
    * models.py - 학습에 사용되는 모델들 모음
    * train_validation.py - train, validation, test 등의 과정의 함수 모음
    * utils.py - 그외 기타 함수나 클래스 모음
* sh - 최종적으로 사용되는 명령어를 sh 파일로 저장 - train.sh, test.sh 등
* submission - 제출하기 위한 submission 파일을 모아두는 폴더
* vis - 결과의 일부를 이미지 파일로 출력하여 저장하는 폴더
* test.py - 테스트 코드 argparser를 이용 전체 test 코드를 작성
* train.py - 테스트 코드 argparser를 이용 전체 train 코드를 작성
* visualize.py - 테스트 코드 argparser를 이용 전체 visualize 코드를 작성
* EDA.ipynb - 간단한 eda 코드 주피터로 확인
* class_dict.csv - 시각화에서 각 클래스에 대한 색 설정 파일

## 결과물
![tree](/assets/image/level2_p_seg/code.JPG)

* 작업한 코드들을 정리 구조화

# 피어세션
[회의록1](https://www.notion.so/20211025-6a388f4302b84943b822b9fd8d51764a)
[회의록2](https://www.notion.so/20211026-42608791dc9e49b8b9fbfc612120920a)
[회의록3](https://www.notion.so/20211027-fe277d78953149c099b59cd316d1f6ea)
[회의록4](https://www.notion.so/20211028-c590a28f48d947288548e9fb7f735960)

# 회고
전체적으로 학습한 모델들을 실험하고 비교하는 한주로 시작하였지만 후반기에는 주피터 등에서 탈피하여 구조화된 코드들을 만들어보는 작업을 하였다. 전체적으로 세션된것도 아니고 다른 깃허브 구조들을 가져와서 하였지만 결과물을 보니 다음에 일부만 바꾸거나 새로운 함수나 클래스를 사용해서 어디에 추가해야될지 보이는 것 같은 느낌이었다. argparser 등을 이용하여 내가 컨트롤 할 수 있는 요소들을 넣는 부분도 꽤 재미있게 느껴졌다. object detection에서는 그냥 라이브러리를 가져다 쓰기만하고 모델이나 기존 방법들을 가져와서 실험하는거에만 집중했다면 이번주는 코드 그 자체에 조금 신경을 쓴 기분이었다. 지난주에 미처 작업하지 못하였는데 회고를 다시 늘려가며 학습 정리하는 것도 신경 써야 할것 같다. 벌써 지난 실험들이 기억이 나지 않는 느낌인데 적어놓는 것의 중요성을 다시 생각해보는 주였다.