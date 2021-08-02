---
title: "학습기록 day01"
date: 2021-08-03 11:38:18 -0400
categories: Boostcamp_AI_tech
---

### 1. Python
Gudio Van Rossum - 1989년 크리스마스에 할일이 없어 파이썬 개발
* 특징
    - 인터프리터 언어
    - 플랫폼에 독립적
    - 쉽고 다양한 라이브러리
    - 동적인 데이터 타입

### 2. Variable
데이터 값을 저장되는 공간에 이름을 붙인 것
* 규칙
    - 알파벳, 숫자, _ 로 구성 (대소문자 구분)
    - 특별한 의미가 있는 예약어는 쓸 수 없다 ex) import, list

### 3. primitive data type
* primitive data type - int, float, string, boolran
* type casting - int(), float()
* list()
    * 여러 데이터를 순서대로 묶어서 관리하는 자료형
    * 특징 - 인덱싱, 슬라이싱, 리스트연산, 추가삭제, 메모리저장방식, 패킹과 언패킹, 이차원리스트

### 4. Basic Operation
+, -, *, /, **, %

### 5. Input and Output
1) input()

    사용자가 직접 문자열을 입력
    
2) print()

    입력 받은 문자열을 콘솔에 출력
    
    ```
    print("what time is it now?")
    time = input()  # 원하는 시간 입력
    print(f"it's {time} o'clock")
    ```
3) print formatting
    * %string
    * format
    * fstring
    * Padding, Aligning
    ```
    something = "water"
    print("%(a)10s"%{"a":something})
    print("{a:*>10s}".format(a=something))
    print(f'{something:-^20}')
    ```

### 6. 피어세션 정리
* Ice breaking
* [그라운드룰](https://github.com/Kangsukmin/K-AI/wiki)
