---
title: "[boostcamp AI Tech] 학습기록 day36 (week8)"
date: 2021-09-24 20:00:00 -0400
categories:
use_math: true
---
# 부스트캠프 특강 (2021.9.23)
# 1. Full Stack ML Engineer - 이준엽
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

# 2. AI Ethics - 오혜연
## 1. AI & Individual
### Bias
* 데이터에 의한 출력의 편향이 윤리적으로 문제를 일으킬 수 있다.
    * 예) COMPAS - 심각한 범죄를 일으킨 백인이 row risk, 경범죄를 한 흑인 소녀가 high risk
* 사회적으로 가지는 윤리적인 문제의 편향이 들어갈 수 있다.
* 소수의 약자들이 부정당할 수 있다.
* 의도하지 않은 패턴들의 문제
* 지역 인식
* 데이터 콜렉팅에서의 편향 등
### Privacy
* 싱가폴 tracetogether app
    * 그 데이터를 가지고 누가 앱의 정보를 가져가는지 알 수 없는 문제
    * 개인의 privacy 문제

## 2. AI & Society
### AI now report
* 10년동안 AI가 미치는 영향
* 주거, 건강보험, 신용, 고용, 법률
* amplified bias
    * 인종, 성별, 학력, 지역에 따라 바이어스 될 수 있다.
* AI는 기술이 있고 다수의 지식이 있는 사람에게는 장점이 많지만, 
* 소수 인종, 기술이 없는 취약층에게 안좋을 수 있다.
* AI에 의한 고용 문제
* 범죄에 악용
    * Deepfakes
    * 보이스피싱

## 3. AI & Humanity
인류에 대한 AI의 영향
* AI for Health
* AI for climate change
    * 전기
    * 운송
    * 보일러
    * 산업
    * 농업
    * 기후예측

# 3. AI 시대의 커리어 빌딩 - 박은정

# 4. 자연어 처리를 위한 언어 모델의 학습과 평가 - 박성준

# [피어세션]()