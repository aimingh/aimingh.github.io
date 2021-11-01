---
title: "[boostcamp AI Tech] 학습기록 day45 (week10)"
date: 2021-10-01 20:00:00 -0400
categories:
use_math: true
---
# Object Detection
# 멘토링
# 놓치기 쉬운 것들 - 빠른 코드 작성하기
* multi-threadhing
* multi-processing
* 참고) 면접질문 multi-threadhing와 multi-processing의 차이?
* GIL (global interpreter lock): 파이썬에서 multi-threadhing을 막는 락
* python에서 file I/O에 대한 부분은 multi-threading이 작동, 하지만 연산에서는 작동하지 않는다.
* pytorch에서는 multi-processing을 사용 
    * DataLoader에서의 num_workers
    * pin_memory=True # gpu의 pinned memoty로 보내 빠른 작업, pageable data transfer vs pinned data trasfer
    * persistent_workers=True # worker를 유지
        * 윈도우의 경우 보안같은 문제때문에 프로세스를 만드는것이 비용이 비쌈
        * -> 만든 프로세스를 유지시키면 장점이 있다.
        * 반면에 리눅스는 비용이 조금 낮기 때문에 윈도우만큼 장점이 적지만 있다.
        * 대신 프로세스가 계속 유지되면서 메모리 부하가 있을 수 있다.
    * drop_last=True # Drop_last를 쓰는것이 좋다. batch 입력에서 문제가 생길 경우가 많아 사용하는것이 일반적으로 좋다.
* joblib
    * 병렬연산을 지원해주는 라이브러리
    * pytorch가 아닌 데이터 처리의 경우 사용하는 것을 강추
* tempfile
    * 폴더에 저장하고 다음에 다시 지우지 말자
    * Temp 폴더의 임시파일을 사용하는 라이브러리
* 머신러닝 엔지니어의 경우 go language도 참고하면 좋다.


# 시각화 마스터 클래스
## 안수빈 마스터
## 질문
* 안수빈 마스터 케글
	* https://www.kaggle.com/subinium
* 추천하는 케글노트북 스타일
	* https://www.kaggle.com/subinium/awesome-visualization-with-titanic-dataset
	* https://www.kaggle.com/subinium/tps-aug-simple-eda
* 포트폴리오 
	* https://techblog.woowahan.com/2531/
	* https://github.com/subinium/CV/blob/master/CV.pdf
* 깃허브 프로필
	* https://github.com/subinium/Misc-Cheatsheet/blob/master/web/github.md
	* https://github.com/rougier?tab=repositories