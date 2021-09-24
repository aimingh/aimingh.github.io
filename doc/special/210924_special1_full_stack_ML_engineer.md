---
title: "[boostcamp AI Tech] 특강 Full Stack ML Engineer - 이준엽"
date: 2021-09-23 20:00:00 -0400
categories:
use_math: true
---


# Full Stack ML Engineer - 이준엽
## 부스트캠프 특강 (2021.9.24)

## Full Stack ML Engineer
### ML Engineer란?
* Machine learning (Deep learning)의 기술을 이해,연구하여 product를 만드는 engineer
* 전통적인 기술과 달리 research와 enginneering의 영역이 모호하다.

### Full stack engineer란?
* client/server software를 개발할 수 있는 사람 (w3chool 발췌)
* 코딩을 잘하고 창의적이며 다른 조직의 사람과 협업할 수 있고 새로운 기술을 배우는 것을 즐기는 개발자 (Apple)
* 자신이 세우는 product를 시간만 있다면 모두 혼자 만들 수 있는 개발자

### Full stack ML engineer란?
* ML(DL)을 이해하고 포함된 product를 만들 수 있는 Full stack engineer

### Full stack ML engineer의 장단점

| | | | | | |
|------|-------|-----------------|--------------|---------|--------------|
| 장점 | 재미! | 빠른 프로토타이핑 | 기술간 시너지 | 팀플레이 | 성장의 다각화 |
| 단점 | 얕은 깊이 | 많이 드는 시간 |  | |  |

### ML Product
* 요구사항 전달 -> 데이터 수집 -> ML 모델 개발 -> 실 서버 배포
* 요구사항 전달
    * 고객사 미팅 (B2B) or 서비스 기획 (B2C) - 추상적 단계
    * 요구사항 + 제약 사항 정리 - 상세 요구/제약 사항과 일정 정리
    * ML problem으로 회귀 - 실생활의 문제를 ML에 맞는 문제로 회귀
* 데이터 수집
    * Raw data 수집 - 요구사항에 맞는 데이터 수집 (bias가 없도록, 저작권 주의)
    * Annotation tool 기획 계발 - 데이터의 정답 입력 툴, 효율성, 모델을 고려
    * Annotation guid 작성 및 운용 - 간단하고 명확하게 문서를 작성
* ML 모델 개발
    * 기존 연구 Reaserch 및 내재화 - 논문 서치/이해, 적절한 연구를 내재화
    * 실 데이터 적용 실험 + 평가 및 피드백 - 수집된 데이터 적용, 평가/모델 수정
    * 모델 차원 경량화 작업 - 경량화, distillation, network surgery
* 실 서버 배포
    * 엔지니어링 경량화 - TensorRT등의 프레임워크, Quantization
    * 연구용 코드 수정 작업 - 배포에 맞게 수정
    * 모델 버전 관리 및 배포 자동화 - 버전관리, 배포 주기/업데이트 자동화 작업

### ML Team
* 일반적인 구성
    * 프로젝트 메니저(1)
    * 개발자(2)
    * 연구자(2)
    * 기획자(1)
    * 데이터관리자(1)

* 다른 경우...
    * 프로젝트 메니저 겸 기획자 겸 연구자 겸 ...
    * 개발자 겸 연구자 겸 데이터관리자 겸 ...
    * 개발자 겸 데이터관리자 겸 ...

### Full stack ML engineer in ML team
* 개발자 겸 연구자 겸 데이터관리자의 역할
1. 실생활 문제를 ML 문제로 formulation
2. raw data 수집
3. annotation tool 개발
4. Data version 관리 및 loader개발
5. model 개발 및 논문 작성
6. Evaluation tool 및 demo 개발
7. 모델 실 서버 배포

### Roadmap of Full stack ML engineer
* Frontend - Vue.jsm 
* Backend - django, flask
* Machine Learning - Pytorch, TensorFlow
* Database - MySQL, MariaDB
* Ops - docker, AWS, github

